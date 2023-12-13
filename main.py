import os
import argparse
import sympy as sp
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import GradUpdater
from symbolic_tensor_graph.graph.pipeline_parallel import GraphDistributer
from symbolic_tensor_graph.graph.convert_chakra import BundledConvertChakra
from models.transformer import (
    transformer_stack as transformer_stack_fn, 
    transformer as transformer_fn
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="dir where stores output traces", required=True)
    parser.add_argument("--output_name", type=str, help="name of output traces", required=True)
    parser.add_argument("--dp", type=int, help="data parallel degree", required=True)
    parser.add_argument("--mp", type=int, help="model parallel degree", required=True)
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel degree", required=False)
    parser.add_argument("--weight_sharded", type=bool, help="whether weight sharded", required=True)
    parser.add_argument("--din", type=int, default=51200, required=False)
    parser.add_argument("--dout", type=int, default=25600, required=False)
    parser.add_argument("--dmodel", type=int, default=25600, required=False)
    parser.add_argument("--dff", type=int, default=25600*4, required=False)
    parser.add_argument("--batch", type=int, default=1024, required=False)
    parser.add_argument("--seq", type=int, default=1024, required=False)
    parser.add_argument("--head", type=int, default=1024, required=False)
    parser.add_argument("--num_stacks", type=int, default=32, required=False)
    parser.add_argument("--chakra_schema_version", type=str, default="v0.0.1", required=False)
    
    args = parser.parse_args()

    if not "%d" in args.output_name:
        args.output_name = f"{args.output_name}.%d.eg"
    generated_filename = os.path.join(args.output_dir, args.output_name)
    dp, mp, pp = sp.symbols("dp mp pp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: args.din,
        Dout: args.dout,
        Dmodel: args.dmodel,
        Dff: args.dff,
        Batch: args.batch,
        Seq: args.seq,
        Head: args.head,
        dp: args.dp,
        mp: args.mp,
        pp: args.pp,
    }
    num_stacks = args.num_stacks
    spatial_parallel_dims = [dp, mp]
    temporal_parallel_dims = [pp]
    
    module_template_dir = "./sharding_spreadsheets/module/divya"
    if args.weight_sharded:
        module_template_dir = "./sharding_spreadsheets/module/fully_sharded_divya"
        
    # build the tensor graph
    mha = TensorGraph.load_tensor_graph(
        os.path.join(module_template_dir, "multi_head_attention.csv")
    )
    ffn = TensorGraph.load_tensor_graph(
        os.path.join(module_template_dir, "feed_forward_network.csv")
    )
    in_emb = TensorGraph.load_tensor_graph(
        os.path.join(module_template_dir, "embedding.csv")
    )
    out_emb = TensorGraph.load_tensor_graph(
        os.path.join(module_template_dir, "embedding.csv")
    )
    stack = transformer_stack_fn(mha, ffn)
    transformer = transformer_fn(in_emb, out_emb, stack, num_stacks)
    transformer_updated_grad = GradUpdater.apply(transformer)
    
    # distribute tensor graph to machines
    def _create_pipeline_tensor_map(_tensors, _temporal_parallel_dims, _symbol_map_value):
        _tensor_map = dict()
        assert len(_temporal_parallel_dims) == 1
        parallel_dim = _temporal_parallel_dims[0]
        range_ = _symbol_map_value[parallel_dim]
        for tensor in _tensors:
            for num_stack in range(num_stacks):
                if f"stack_{num_stack}" in tensor.id:
                    _tensor_map[tensor.id] = {parallel_dim: (num_stack+1) % range_}
                    break
            if "in_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: 0}
            elif "out_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: (num_stacks+1) % range_}
        return _tensor_map
    pipeline_tensor_map = _create_pipeline_tensor_map(transformer_updated_grad.tensors, temporal_parallel_dims, symbol_map_value)
    distributed_tensor_graph = GraphDistributer.apply(
        transformer_updated_grad,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        pipeline_tensor_map
    )
    
    # readout to chakra
    distributed_chakra_graph = BundledConvertChakra.apply(distributed_tensor_graph, symbol_map_value)
    if args.chakra_schema_version == "v0.0.1":
        from symbolic_tensor_graph.chakra.backends.chakra_00_1_backend import Chakra001Backend as ReadoutBackend
    elif args.chakra_schema_version == "v0.0.4":
        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import Chakra004Backend as ReadoutBackend
    elif args.chakra_schema_version == "json":
        from symbolic_tensor_graph.chakra.backends.json_backend import JsonBackend as ReadoutBackend
    else:
        assert False
    distributed_chakra_graph.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
    