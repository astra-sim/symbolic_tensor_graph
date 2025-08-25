import os
import argparse
import copy
import sympy as sp
from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.ops import Add, PlaceHolder, Element2
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import (
    GradUpdater,
    MicroBatchReplicator,
    MicroBatchReplicatorPostProcess,
)
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.graph_distributer import GraphDistributer
from symbolic_tensor_graph.graph.convert_chakra import (
    BundledConvertChakra,
    ConvertChakra,
)
from models.transformer import (
    transformer_stack as transformer_stack_fn,
    transformer as transformer_fn,
)
import re

mixprecision = False


def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")


def _create_pipeline_tensor_map_mix_precision(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
):
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]

    # Determine how many transformer blocks belong to each pipeline stage
    num_stacks_each_stage = [num_stacks // range_] * range_
    for i in range(num_stacks % range_):
        num_stacks_each_stage[i] += 1  # distribute remainder to early stages
    # Cumulative upper bounds for easy stage lookup
    cumulative = []
    acc = 0
    for v in num_stacks_each_stage:
        acc += v
        cumulative.append(acc)

    for tensor in _tensors:
        tid = tensor.id
        # ------------------------------------------------------------------
        # 1) Transformer block tensors
        # ------------------------------------------------------------------
        m = re.search(r"transformer\.(\d+)", tid)
        if m:
            block_idx = int(m.group(1))
            # Find the first cumulative upper bound that exceeds block_idx
            stage = next(i for i, up in enumerate(cumulative) if block_idx < up)
            _tensor_map[tid] = {parallel_dim: stage}
            continue

        # ------------------------------------------------------------------
        # 2) Special tensors (embeddings, loss etc.)
        # ------------------------------------------------------------------
        if "in_emb" in tid:
            _tensor_map[tid] = {parallel_dim: 0}
        elif "out_emb" in tid or "loss" in tid:
            _tensor_map[tid] = {parallel_dim: (range_ - 1)}
        else:
            # Any tensor that doesn't match the above categories should be
            # impossible – raise explicit error to catch new patterns early.
            raise ValueError(f"Unrecognized tensor id for pipeline mapping: {tid}")

    return _tensor_map


def _create_pipeline_tensor_map(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
):
    if mixprecision:
        return _create_pipeline_tensor_map_mix_precision(
            _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
        )
    _tensor_map = dict()
    assert len(_temporal_parallel_dims) == 1
    parallel_dim = _temporal_parallel_dims[0]
    range_ = _symbol_map_value[parallel_dim]
    num_stacks_each_stage = list()
    for i in range(range_):
        num_stacks_each_stage.append(num_stacks // range_)
    for i in range(num_stacks - range_ * (num_stacks // range_)):
        num_stacks_each_stage[i] += 1
    for i in range(range_):
        if i == 0:
            continue
        num_stacks_each_stage[i] += num_stacks_each_stage[i - 1]
    # num_stacks_each_stage.append(num_stacks_each_stage[-1]+100000)

    for tensor in _tensors:
        if tensor.id == "transformer.18._sharded_weight@1":
            pass
        found = False
        for num_stack in range(num_stacks):
            if f"transformer.{num_stack}." in tensor.id:
                for stage, upper_bound in enumerate(num_stacks_each_stage):
                    if num_stack < upper_bound:
                        _tensor_map[tensor.id] = {parallel_dim: stage}
                        found = True
                        break
                if found:
                    break
        if found:
            pass
        elif "in_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: 0}
        elif "out_emb" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: (num_stacks - 1) % range_}
        elif "loss" in tensor.id:
            _tensor_map[tensor.id] = {parallel_dim: (num_stacks - 1) % range_}
        else:
            assert False, tensor.name
    return _tensor_map


# ------------------------------------------------------------------
# Helper utilities for VRAM accounting
# ------------------------------------------------------------------
def _tensor_mem_class(tensor):
    """Classify tensor for persistent VRAM footprint.

    weight : parameter shards stored persistently (exclude *_assembled_weight*)
    grad   : persistent gradient shards (only *_sharded_grad*)
    act    : forward activations that are kept (exclude backward/temp).
    None   : skip (temporary tensors not expected to occupy persistent VRAM)
    """

    name = tensor.name or ""

    # There are a few tensor variants that should NOT be counted towards
    # persistent VRAM because they are ephemeral (assembled buffers used
    # inside FSDP, micro-batch intermediate shards, etc.).
    # We filter those out *before* the normal classification.

    # 0) Explicitly ignore assembled / temporary buffers
    tmp_keywords = [
        "_assembled_weight",  # FSDP full-weight buffer
        "_assembled_weight_backward",  # its backward shadow
        "_assembled_grad",  # full gradient before shard/RS
    ]
    for kw in tmp_keywords:
        if kw in name:
            return None

    # 1) Parameters (persistent weight shards)
    # Note: Only the sharded parameters (require_grads == True) are stored
    # permanently. The full-precision assembled weights have been excluded
    # above.
    if tensor.require_grads:
        return "weight"

    # 2) Gradients (persistent)
    # We only keep the per-rank sharded gradients (post reduce-scatter).
    # Micro-batch private gradients (prefixed with "mb<i>.") are local
    # temporaries and should be skipped.
    if tensor.grad_of is not None and "_sharded_grad" in name:
        return "grad"

    # Heuristics to exclude large backward / other temporaries
    lower_name = name.lower()
    if "_backward" in lower_name or lower_name.split(".")[-1].startswith(
        ("d", "dw", "dq", "dk", "dv")
    ):
        return None  # skip temp

    # Everything else => activation
    return "act"


def _weight_and_opt_sizes(tensor, symbol_map, mixed_precision=False):
    """Return (weight_bytes, optimizer_state_bytes) for a parameter tensor.

    The logic mirrors ConvertChakra._create_IOInfo so reported numbers add
    up to the same total size but we split the optimizer state out.
    """
    from symbolic_tensor_graph.tensor import Tensor as _Tensor

    elem_cnt = _Tensor.eval_expr(_Tensor.eval_size(tensor.y_shape), symbol_map)

    # Weight representation bytes
    if mixed_precision:
        # 2 bytes (fp16) + 4 bytes (fp32 master) = 6 bytes per element
        weight_bytes = int(elem_cnt * 1.5) * 4
    else:
        # standard fp32 => 4 bytes per element
        weight_bytes = elem_cnt * 4

    # Optimizer state bytes (Adam: m & v, fp32): 2 * 4 bytes
    opt_bytes = (
        elem_cnt * 4 * 2
    )  # multiplier 2 already folded as 4 in original? wait orig used *4 only.
    # Note: ConvertChakra used *4 ( not *8 ) – but that code path is inconsistent wrt bytes.
    # For consistency with their total we keep the same (single replica) so /2.
    opt_bytes = elem_cnt * 4  # to match _create_IOInfo logic

    return weight_bytes, opt_bytes


def _tensor_size_bytes(tensor, symbol_map, mixed_precision=False):
    """Total size as computed by ConvertChakra (weight+opt if param)."""
    info = ConvertChakra._create_IOInfo(
        tensor, symbol_map, mixed_precision, fsdp_enabled=symbol_map.get("fsdp", 0) > 1
    )
    return info["size"]


def _print_gpu_vram(bundle_graph, symbol_map, mixed_precision=False, header=""):
    GiB = 1024**3
    for rank_key, tg in bundle_graph.graphs.items():
        stats = {"weight": 0, "opt": 0, "act": 0, "grad": 0}
        tensor_details = []  # Store details for sorting
        for tensor in tg.tensors:
            cls = _tensor_mem_class(tensor)
            if cls is None:
                continue
            if cls == "weight":
                w_b, opt_b = _weight_and_opt_sizes(tensor, symbol_map, mixed_precision)
                stats["weight"] += w_b
                stats["opt"] += opt_b
                tensor_details.append(
                    (cls, w_b + opt_b, tensor.id, Tensor.stringfy_shape(tensor.y_shape))
                )
            else:
                size_b = _tensor_size_bytes(tensor, symbol_map, mixed_precision)
                stats[cls] += size_b
                tensor_details.append(
                    (cls, size_b, tensor.id, Tensor.stringfy_shape(tensor.y_shape))
                )
        total = sum(stats.values())
        rk_str = ",".join([f"{d[0]}={d[1]}" for d in rank_key])
        print(
            f"{header}[GPU {rk_str}] total={total / GiB:.3f} GiB | "
            f"weights={stats['weight'] / GiB:.3f} | "
            f"opt={stats['opt'] / GiB:.3f} | "
            f"acts={stats['act'] / GiB:.3f} | "
            f"grads={stats['grad'] / GiB:.3f}"
        )
        # Print top 5 largest tensors by size for this rank
        # tensor_details.sort(key=lambda x: x[1], reverse=True)
        # print(f"    Top 5 Tensors for GPU {rk_str}:")
        # for i in range(min(5, len(tensor_details))):
        #     cls, size_b, t_id, shape_str = tensor_details[i]
        #     print(f"      {i+1}. Type={cls}, Size={size_b / GiB:.4f} GiB, ID={t_id}, Shape={shape_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, help="dir where stores output traces", required=True
    )
    parser.add_argument(
        "--output_name", type=str, help="name of output traces", required=True
    )
    parser.add_argument(
        "--dp", type=int, help="data parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--tp", type=int, help="tensor parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--sp", type=int, help="sequence parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--ep", type=int, help="expert parallel degree", required=False, default=1
    )
    parser.add_argument(
        "--pp", type=int, default=1, help="pipeline parallel degree", required=False
    )
    parser.add_argument(
        "--weight_sharded",
        type=str_to_bool,
        help="whether weight sharded",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--activation_recompute",
        type=str_to_bool,
        help="whether recompute activation",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--tpsp",
        type=str_to_bool,
        help="use tp+sp or tp only",
        required=False,
        default=True,
    )
    parser.add_argument("--dvocal", type=int, default=32000, required=False)
    parser.add_argument("--dmodel", type=int, default=8192, required=False)
    parser.add_argument("--dff", type=int, default=28672, required=False)
    parser.add_argument("--batch", type=int, default=64, required=False)
    parser.add_argument("--micro_batch", type=int, default=-1, required=False)
    parser.add_argument("--seq", type=int, default=1024, required=False)
    parser.add_argument("--head", type=int, default=64, required=False)
    parser.add_argument("--kvhead", type=int, default=8, required=False)
    parser.add_argument("--num_stacks", type=int, default=80, required=False)
    parser.add_argument("--experts", type=int, default=8, required=False)
    parser.add_argument("--kexperts", type=int, default=2, required=False)
    parser.add_argument(
        "--chakra_schema_version", type=str, default="v0.0.4", required=False
    )
    parser.add_argument("--model_type", type=str, default="dense", required=False)
    parser.add_argument(
        "--mixed_precision", type=str_to_bool, default=False, required=False
    )
    parser.add_argument(
        "--print_gpu_vram",
        type=str_to_bool,
        default=False,
        required=False,
        help="Whether to print per-GPU VRAM footprint (total / params / acts / grads) in GiB",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not "%d" in args.output_name:
        args.output_name = f"{args.output_name}.%d.et"
    generated_filename = os.path.join(args.output_dir, args.output_name)
    dp, tp, pp, spp, ep, fsdp = sp.symbols("dp tp pp cp ep fsdp")
    (
        Din,
        Dout,
        Dmodel,
        Dff,
        Batch,
        Seq,
        Head,
        KVHead,
        Experts,
        KExperts,
        Dvocal,
        MicroBatch,
    ) = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head KVHead Experts KExperts Dvocal MicroBatch"
    )
    if args.micro_batch == -1:
        args.micro_batch = args.batch
    symbol_map_value = {
        Dvocal: args.dvocal,
        Dmodel: args.dmodel,
        Dff: args.dff,
        Batch: args.batch,
        MicroBatch: args.micro_batch,
        Seq: args.seq,
        Head: args.head,
        KVHead: args.kvhead,
        Experts: args.experts,
        KExperts: args.kexperts,
        dp: args.dp,
        tp: args.tp,
        pp: args.pp,
        spp: args.sp,
        ep: args.ep,
    }
    num_stacks = args.num_stacks
    temporal_parallel_dims = [pp]
    if args.weight_sharded:
        symbol_map_value[fsdp] = args.dp if args.dp != 0 else 1
        symbol_map_value["fsdp"] = args.dp if args.dp != 0 else 1
    else:
        symbol_map_value[fsdp] = 1
        symbol_map_value["fsdp"] = 1

    hook = 1
    global mixprecision
    if args.mixed_precision:
        mixprecision = True

    if args.model_type == "llama" or args.model_type == "dense":
        if mixprecision:
            from models.stage1.llama_model import llama as transformer_dense
        else:
            from models.stage1.gpt_model import gpt as transformer_dense

        print("Assembling dense model")
        transformer_dense = transformer_dense(
            num_stacks, regenerate=True, tpsp=args.tpsp
        )
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
            transformer_dense = MicroBatchReplicator.apply(
                transformer_dense, symbol_map_value
            )
        else:
            print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"Batch": "MicroBatch"},
            )

        if args.weight_sharded:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_dense.visualize("dense")
        # transformer_dense.save_tensor_graph("llama.csv")

        transformer_dense = GradUpdater.apply(transformer_dense, inplace=True)
        spatial_parallel_dims_dense = [dp, tp, spp]

        symbol_map_value[tp] *= symbol_map_value[ep]
        # dense model
        pipeline_tensor_map = _create_pipeline_tensor_map(
            transformer_dense.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
        )

        print("Dense model: Distributing")
        distributed_tensor_graph_dense = GraphDistributer.apply(
            transformer_dense,
            symbol_map_value,
            spatial_parallel_dims_dense,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        if args.print_gpu_vram:
            _print_gpu_vram(
                distributed_tensor_graph_dense,
                symbol_map_value,
                mixed_precision=args.mixed_precision,
                header="[Dense] ",
            )

        print("Dense model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_dense = BundledConvertChakra.apply(
            distributed_tensor_graph_dense,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") != "0":
            distributed_chakra_graph_dense = MicroBatchReplicatorPostProcess.apply(
                distributed_chakra_graph_dense, args.batch // args.micro_batch
            )

        print("Dense model: reading out")
        distributed_chakra_graph_dense.readout(
            generated_filename, backend=ReadoutBackend
        )
    elif args.model_type == "gpt":
        from models.stage1.gpt_model import gpt as transformer_dense

        print("Assembling dense model")
        transformer_dense = transformer_dense(
            num_stacks, regenerate=True, tpsp=args.tpsp
        )
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
            transformer_dense = MicroBatchReplicator.apply(
                transformer_dense, symbol_map_value
            )
        else:
            print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"Batch": "MicroBatch"},
            )

        if args.weight_sharded:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_dense = ReplicateGraph.apply(
                transformer_dense, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_dense.visualize("dense")
        # transformer_dense.save_tensor_graph("gpt.csv")

        transformer_dense = GradUpdater.apply(transformer_dense, inplace=True)
        spatial_parallel_dims_dense = [dp, tp, spp]

        symbol_map_value[tp] *= symbol_map_value[ep]
        # dense model
        pipeline_tensor_map = _create_pipeline_tensor_map(
            transformer_dense.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
        )

        print("Dense model: Distributing")
        distributed_tensor_graph_dense = GraphDistributer.apply(
            transformer_dense,
            symbol_map_value,
            spatial_parallel_dims_dense,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        if args.print_gpu_vram:
            _print_gpu_vram(
                distributed_tensor_graph_dense,
                symbol_map_value,
                mixed_precision=args.mixed_precision,
                header="[GPT] ",
            )

        print("Dense model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_dense = BundledConvertChakra.apply(
            distributed_tensor_graph_dense,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("Dense model: reading out")
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") != "0":
            distributed_chakra_graph_dense = MicroBatchReplicatorPostProcess.apply(
                distributed_chakra_graph_dense, args.batch // args.micro_batch
            )
        distributed_chakra_graph_dense.readout(
            generated_filename, backend=ReadoutBackend
        )

    elif args.model_type == "moe":
        from models.stage1.moe_model import transformer as transformer_moe

        assert args.tpsp
        print("Assembling moe model")
        transformer_moe = transformer_moe(num_stacks, symbol_map_value, regenerate=True)
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") == "0":
            transformer_moe = MicroBatchReplicator.apply(
                transformer_moe, symbol_map_value
            )
        else:
            print("[Warning] MICROBATCH OPTIMIZE sometimes generate incorrect graphs, use with caution!")
            assert False, "disable for now"
        transformer_moe = ReplicateGraph.apply(
            transformer_moe,
            inplace=True,
            old_symbol_map_new_symbol={"Batch": "MicroBatch"},
        )

        if args.weight_sharded:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_moe.visualize("moe")
        # transformer_moe.save_tensor_graph("moe.csv")
        transformer_moe = GradUpdater.apply(transformer_moe, inplace=True)
        spatial_parallel_dims_moe = [dp, tp, spp, ep]

        # moe model
        pipeline_tensor_map = _create_pipeline_tensor_map(
            transformer_moe.tensors,
            temporal_parallel_dims,
            symbol_map_value,
            num_stacks,
        )

        print("MoE model: Distributing")
        distributed_tensor_graph_moe = GraphDistributer.apply(
            transformer_moe,
            symbol_map_value,
            spatial_parallel_dims_moe,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        if args.print_gpu_vram:
            _print_gpu_vram(
                distributed_tensor_graph_moe,
                symbol_map_value,
                mixed_precision=args.mixed_precision,
                header="[MoE] ",
            )

        print("MoE model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_moe = BundledConvertChakra.apply(
            distributed_tensor_graph_moe,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
            mixed_precision=args.mixed_precision,
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("MoE model: reading out")
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") != "0":
            distributed_chakra_graph_moe = MicroBatchReplicatorPostProcess.apply(
                distributed_chakra_graph_moe, args.batch // args.micro_batch
            )
        distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)

    elif args.model_type == "debug":
        transformer_moe = TensorGraph.load_tensor_graph(
            "./sharding_spreadsheets/module3/tpsp/embedding.csv"
        )
        transformer_moe = ReplicateGraph.apply(
            transformer_moe,
            inplace=True,
            old_symbol_map_new_symbol={
                "Batch": "MicroBatch",
                "Din": "Dvocal",
                "Dout": "Dvocal",
            },
        )

        if args.weight_sharded:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe,
                inplace=True,
                old_symbol_map_new_symbol={"fsdp": "dp"},
            )
        else:
            transformer_moe = ReplicateGraph.apply(
                transformer_moe, inplace=True, old_symbol_map_new_symbol={"fsdp": 1}
            )

        # transformer_moe.visualize("moe")
        # transformer_moe.save_tensor_graph("moe.csv")
        transformer_moe = GradUpdater.apply(transformer_moe, inplace=True)
        spatial_parallel_dims_moe = [dp, tp, spp, ep]

        # moe model
        assert args.pp == 1
        pipeline_tensor_map = {
            "x@0": {pp: 0},
            "w@0": {pp: 0},
            "y@0": {pp: 0},
            "dy@0": {pp: 0},
            "dw@0": {pp: 0},
            "dx@0": {pp: 0},
            "w@1": {pp: 0},
        }

        print("MoE model: Distributing")
        distributed_tensor_graph_moe = GraphDistributer.apply(
            transformer_moe,
            symbol_map_value,
            spatial_parallel_dims_moe,
            temporal_parallel_dims,
            pipeline_tensor_map,
        )

        print("MoE model: Converting Chakra")
        comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
        distributed_chakra_graph_moe = BundledConvertChakra.apply(
            distributed_tensor_graph_moe,
            symbol_map_value,
            os.path.join(args.output_dir, comm_group_file),
        )

        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
            Chakra004Backend as ReadoutBackend,
        )

        print("MoE model: reading out")
        if os.environ.get("STAGE_MICROBATCH_OPTIMIZE", "0") != "0":
            distributed_chakra_graph_moe = MicroBatchReplicatorPostProcess.apply(
                distributed_chakra_graph_moe, args.batch // args.micro_batch
            )
        distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
