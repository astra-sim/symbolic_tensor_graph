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
import pickle


def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="dir where stores output traces", required=True)
    parser.add_argument("--output_name", type=str, help="name of output traces", required=True)
    parser.add_argument("--comm_group_file", type=str, help="name of comm_group_file", required=True)
    parser.add_argument("--dp", type=int, help="data parallel degree", required=True)
    parser.add_argument("--mp", type=int, help="model parallel degree", required=True)
    parser.add_argument("--sp", type=int, help="token parallel degree", required=False, default=1)
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel degree", required=False)
    parser.add_argument("--weight_sharded", type=str_to_bool, help="whether weight sharded", required=True)
    parser.add_argument("--din", type=int, default=51200, required=False)
    parser.add_argument("--dout", type=int, default=25600, required=False)
    parser.add_argument("--dmodel", type=int, default=25600, required=False)
    parser.add_argument("--dff", type=int, default=25600*4, required=False)
    parser.add_argument("--batch", type=int, default=1024, required=False)
    parser.add_argument("--seq", type=int, default=1024, required=False)
    parser.add_argument("--head", type=int, default=1024, required=False)
    parser.add_argument("--num_stacks", type=int, default=32, required=False)
    parser.add_argument("--chakra_schema_version", type=str, default="v0.0.4", required=False)
    parser.add_argument("--layer_each_stage", type=int, default=1, required=False)
    parser.add_argument("--store_rank", type=str, default=None, required=False)
    parser.add_argument("--load_rank_map", type=str, default=None, required=False)
    parser.add_argument("--generate_io_info", type=str_to_bool, help="whether include io infos")
    
    args = parser.parse_args()

    if not "%d" in args.output_name:
        args.output_name = f"{args.output_name}.%d.eg"
    generated_filename = os.path.join(args.output_dir, args.output_name)
    dp, mp, pp, spp = sp.symbols("dp mp pp sp")
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
        spp: args.sp
    }
    num_stacks = args.num_stacks
    spatial_parallel_dims = [dp, mp, spp]
    temporal_parallel_dims = [pp]
    
    module_template_dir = os.path.join(
            os.path.split(
                os.path.abspath(__file__)
            )[0],
            "./sharding_spreadsheets/module/fullset"  
    )
    if args.weight_sharded:
        module_template_dir = os.path.join(
                os.path.split(
                    os.path.abspath(__file__)
                )[0],
                "./sharding_spreadsheets/module/fully_sharded_fullset"
        )
        
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
    def _create_pipeline_tensor_map(_tensors, _temporal_parallel_dims, _symbol_map_value, layer_each_stage=-1):
        # TODO: bug, num_stage=16, num_stacks=96, and pipeline 14, 15 is empty.
        _tensor_map = dict()
        assert len(_temporal_parallel_dims) == 1
        parallel_dim = _temporal_parallel_dims[0]
        num_stages = _symbol_map_value[parallel_dim]
        if layer_each_stage <= 0:
            # hotfix
            # layer_each_stage = (num_stacks+2+num_stages-1) // num_stages
            layer_each_stage = (num_stacks+2) // num_stages
        for tensor in _tensors:
            for num_stack in range(num_stacks):
                if f"stack_{num_stack}_" in tensor.id:
                    _tensor_map[tensor.id] = {parallel_dim: ((num_stack+1)//layer_each_stage) % num_stages}
                    break
            if "in_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: 0}
            elif "out_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: ((num_stacks+1)//layer_each_stage) % num_stages}
        return _tensor_map
    pipeline_tensor_map = _create_pipeline_tensor_map(transformer_updated_grad.tensors, temporal_parallel_dims, symbol_map_value, args.layer_each_stage)
    distributed_tensor_graph = GraphDistributer.apply(
        transformer_updated_grad,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        pipeline_tensor_map
    )
            
    if args.store_rank is not None:
        readable_ranks = list(distributed_tensor_graph.graphs.keys())
        f = open(args.store_rank, "wb")
        pickle.dump(readable_ranks, f)
        f.close()
    readable_rank_map_number_rank = None
    if args.load_rank_map is not None:
        f = open(args.load_rank_map, 'rb')
        readable_rank_map_number_rank = pickle.load(f)
        f.close()
    # readout to chakra
    if args.generate_io_info:
        BundledConvertChakra._ConvertChakra.with_comm_info = True
    distributed_chakra_graph = BundledConvertChakra.apply(distributed_tensor_graph, symbol_map_value, os.path.join(args.output_dir, args.comm_group_file), readable_rank_map_number_rank=readable_rank_map_number_rank)
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
    
