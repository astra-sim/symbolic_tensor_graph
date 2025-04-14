import os
import argparse
import copy
import sympy as sp
from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.ops import Add, PlaceHolder, Element2
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import GradUpdater, MicroBatchReplicator
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.graph_distributer import GraphDistributer
from symbolic_tensor_graph.graph.convert_chakra import BundledConvertChakra
from models.transformer import (
    transformer_stack as transformer_stack_fn,
    transformer as transformer_fn,
)


def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")


def _create_pipeline_tensor_map(
    _tensors, _temporal_parallel_dims, _symbol_map_value, num_stacks
):
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
        found = False
        for num_stack in range(num_stacks):
            if f"transformer.{num_stack}" in tensor.id:
                for stage, upper_bound in enumerate(num_stacks_each_stage):
                    if num_stack < upper_bound:
                        _tensor_map[tensor.id] = {parallel_dim: (num_stack) % range_}
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
    parser.add_argument("--mixed_precision", type=str_to_bool, default=False, required=False)

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

    hook = 1

    if args.model_type == "llama" or args.model_type == "dense":
        from models.stage1.llama_model import llama as transformer_dense

        print("Assembling dense model")
        transformer_dense = transformer_dense(num_stacks, regenerate=True)
        transformer_dense = MicroBatchReplicator.apply(
            transformer_dense, symbol_map_value
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
        transformer_dense.save_tensor_graph("llama.csv")

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
        distributed_chakra_graph_dense.readout(
            generated_filename, backend=ReadoutBackend
        )
    elif args.model_type == "gpt":
        from models.stage1.gpt_model import gpt as transformer_dense

        print("Assembling dense model")
        transformer_dense = transformer_dense(num_stacks, regenerate=True)
        transformer_dense = MicroBatchReplicator.apply(
            transformer_dense, symbol_map_value
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
        transformer_dense.save_tensor_graph("gpt.csv")

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
        distributed_chakra_graph_dense.readout(
            generated_filename, backend=ReadoutBackend
        )

    elif args.model_type == "moe":
        from models.stage1.moe_model import transformer as transformer_moe

        print("Assembling moe model")
        transformer_moe = transformer_moe(num_stacks, symbol_map_value, regenerate=True)
        transformer_moe = MicroBatchReplicator.apply(transformer_moe, symbol_map_value)
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
        transformer_moe.save_tensor_graph("moe.csv")
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
        distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
