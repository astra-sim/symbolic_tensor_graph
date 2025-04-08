import os
import argparse
import copy
import sympy as sp
from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.ops import Add, PlaceHolder, Element2
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import GradUpdater
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
    parser.add_argument("--batch", type=int, default=1024, required=False)
    parser.add_argument("--seq", type=int, default=1024, required=False)
    parser.add_argument("--head", type=int, default=64, required=False)
    parser.add_argument("--kvhead", type=int, default=8, required=False)
    parser.add_argument("--num_stacks", type=int, default=80, required=False)
    parser.add_argument("--experts", type=int, default=8, required=False)
    parser.add_argument("--kexperts", type=int, default=2, required=False)
    parser.add_argument(
        "--chakra_schema_version", type=str, default="v0.0.4", required=False
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not "%d" in args.output_name:
        args.output_name = f"{args.output_name}.%d.et"
    generated_filename = os.path.join(args.output_dir, args.output_name)
    dp, tp, pp, spp, ep = sp.symbols("dp tp pp sp ep")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head, KVHead, Experts, KExperts, Dvocal = (
        sp.symbols("Din Dout Dmodel Dff Batch Seq Head KVHead Experts KExperts Dvocal")
    )
    symbol_map_value = {
        Dvocal: args.dvocal,
        Dmodel: args.dmodel,
        Dff: args.dff,
        Batch: args.batch,
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

    from models.stage1.dense_model import transformer as transformer_dense

    print("Assembling dense model")
    transformer_dense = transformer_dense(num_stacks)
    transformer_dense = GradUpdater.apply(transformer_dense, inplace=True)
    spatial_parallel_dims_dense = [dp, tp, spp]

    from models.stage1.moe_model import transformer as transformer_moe

    print("Assembling moe model")
    transformer_moe = transformer_moe(num_stacks, symbol_map_value)
    transformer_moe = GradUpdater.apply(transformer_moe, inplace=True)
    spatial_parallel_dims_moe = [dp, tp, spp, ep]

    hook = 1

    # dense model
    pipeline_tensor_map = dict()
    for tensor in transformer_dense.tensors:
        pipeline_tensor_map[tensor.id] = {pp: 0}

    print("Dense model: Distributing")
    distributed_tensor_graph_dense = GraphDistributer.apply(
        transformer_dense,
        symbol_map_value,
        spatial_parallel_dims_dense,
        temporal_parallel_dims,
        pipeline_tensor_map,
    )

    print("Dense model: Converting Chakra")
    args.output_name = "testdense.%d.et"
    comm_group_file = args.output_name.replace(".%d", "").replace(".et", ".json")
    distributed_chakra_graph_dense = BundledConvertChakra.apply(
        distributed_tensor_graph_dense,
        symbol_map_value,
        os.path.join(args.output_dir, comm_group_file),
    )

    from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
        Chakra004Backend as ReadoutBackend,
    )

    print("Dense model: reading out")
    distributed_chakra_graph_dense.readout(generated_filename, backend=ReadoutBackend)

    # moe model
    pipeline_tensor_map = dict()
    for tensor in transformer_moe.tensors:
        pipeline_tensor_map[tensor.id] = {pp: 0}

    print("MoE model: Distributing")
    distributed_tensor_graph_moe = GraphDistributer.apply(
        transformer_moe,
        symbol_map_value,
        spatial_parallel_dims_moe,
        temporal_parallel_dims,
        pipeline_tensor_map,
    )

    print("MoE model: Converting Chakra")
    args.output_name = "moe.%d.et"
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
    distributed_chakra_graph_moe.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
