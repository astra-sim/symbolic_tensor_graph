import os
import argparse
import sympy as sp
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import GradUpdater
from symbolic_tensor_graph.graph.graph_distributer import GraphDistributer
from symbolic_tensor_graph.graph.convert_chakra import BundledConvertChakra
from models.transformer import (
    transformer_stack as transformer_stack_fn, 
    transformer as transformer_fn
)
from models.transformer_forward_only import (
    transformer_stack as inf_transformer_stack_fn, 
    transformer as inf_transformer_fn
)
from symbolic_tensor_graph.graph.pipeline_parallel import naive_pipeline_emb_separate_n_layer_each_stage, gpipe_pipeline_prepare
import pickle


def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="dir where stores output traces", required=True)
    parser.add_argument("--output_name", type=str, help="name of output traces", required=True)
    parser.add_argument("--comm_group_file", type=str, help="name of comm_group_file", required=True)
    parser.add_argument("--dp", type=int, help="data parallel degree", required=False, default=1)
    parser.add_argument("--tp", type=int, help="tensor parallel degree", required=False, default=1)
    parser.add_argument("--sp", type=int, help="sequence parallel degree", required=False, default=1)
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel degree", required=False)
    parser.add_argument("--weight_sharded", type=str_to_bool, help="whether weight sharded", required=False, default=False)
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
    parser.add_argument("--generate_io_info", type=str_to_bool, help="whether include io infos(false knob, for now it will always be generated)", default=True, required=False)
    parser.add_argument("--templates", type=str, default="training", choices=["training", "prefilling", "decoding"], required=False)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not "%d" in args.output_name:
        args.output_name = f"{args.output_name}.%d.et"
    generated_filename = os.path.join(args.output_dir, args.output_name)
    dp, tp, pp, spp = sp.symbols("dp tp pp sp")
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
        tp: args.tp,
        pp: args.pp,
        spp: args.sp
    }
    num_stacks = args.num_stacks
    spatial_parallel_dims = [dp, tp, spp]
    temporal_parallel_dims = [pp]
    
    TEMPLATE_DIR_MAP = {
        ("training", True): "./sharding_spreadsheets/module/fully_sharded_fullset",
        ("training", False): "./sharding_spreadsheets/module/fullset",
        ("prefilling", True): "./sharding_spreadsheets/module/prefilling_fully_sharded_fullset",
        ("prefilling", False): "./sharding_spreadsheets/module/prefilling_fullset",
        ("decoding", True): "./sharding_spreadsheets/module/decoding_fully_sharded_fullset",
        ("decoding", False): "./sharding_spreadsheets/module/decoding_fullset",
    }
    template_dir = TEMPLATE_DIR_MAP[(args.templates, args.weight_sharded)]
    
    module_template_dir = os.path.join(
        os.path.split(
            os.path.abspath(__file__)
        )[0],
        template_dir
    )
    # if args.weight_sharded:
    #     module_template_dir = os.path.join(
    #             os.path.split(
    #                 os.path.abspath(__file__)
    #             )[0],
    #             "./sharding_spreadsheets/module/fully_sharded_fullset"
    #     )
        
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
    if args.templates in {"prefilling", "decoding"}:
        stack = inf_transformer_stack_fn(mha, ffn)
        transformer = inf_transformer_fn(in_emb, out_emb, stack, num_stacks)
        transformer_updated_grad = transformer
    elif args.templates in {"training"}:
        stack = transformer_stack_fn(mha, ffn)
        transformer = transformer_fn(in_emb, out_emb, stack, num_stacks)
    else:
        assert False
        
    # # distribute tensor graph to machines
    # def _create_pipeline_tensor_map(_tensors, _temporal_parallel_dims, _symbol_map_value, layer_each_stage=1):
    #     _tensor_map = dict()
    #     assert len(_temporal_parallel_dims) == 1
    #     parallel_dim = _temporal_parallel_dims[0]
    #     range_ = _symbol_map_value[parallel_dim]
    #     for tensor in _tensors:
    #         for num_stack in range(num_stacks):
    #             if f"stack_{num_stack}_" in tensor.id:
    #                 _tensor_map[tensor.id] = {parallel_dim: ((num_stack+1)//layer_each_stage) % range_}
    #                 break
    #         if "in_emb" in tensor.id:
    #             _tensor_map[tensor.id] = {parallel_dim: 0}
    #         elif "out_emb" in tensor.id:
    #             _tensor_map[tensor.id] = {parallel_dim: ((num_stacks+1)//layer_each_stage) % range_}
    #     return _tensor_map
    
    gpipe_symbol_map_value = symbol_map_value.copy()
    gpipe_symbol_map_value[sp.symbols("MicroBatch")] = 2
    merged = gpipe_pipeline_prepare(transformer, gpipe_symbol_map_value)
    if args.templates in {"training"}:
        transformer = GradUpdater.apply(transformer)
        merged = GradUpdater.apply(merged)
    # merged = gpipe_pipeline_prepare(transformer, gpipe_symbol_map_value)
    # merged.save_tensor_graph("merged.csv")
    # hook = 1
    
    _create_pipeline_tensor_map = naive_pipeline_emb_separate_n_layer_each_stage
    
    transformer, pipeline_tensor_map = _create_pipeline_tensor_map(transformer, temporal_parallel_dims, symbol_map_value, args.num_stacks, args.layer_each_stage)
    merged, pipeline_merged_tensor_map = _create_pipeline_tensor_map(merged, temporal_parallel_dims, gpipe_symbol_map_value, args.num_stacks, args.layer_each_stage)

    distributed_tensor_graph = GraphDistributer.apply(
        transformer,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        pipeline_tensor_map
    )
    distributed_gpipe_tensor_graph = GraphDistributer.apply(
        merged,
        gpipe_symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        pipeline_merged_tensor_map
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
    distributed_chakra_graph = BundledConvertChakra.apply(distributed_tensor_graph, symbol_map_value, os.path.join(args.output_dir, args.comm_group_file), readable_rank_map_number_rank=readable_rank_map_number_rank)
    distributed_gpipe_chakra_graph = BundledConvertChakra.apply(distributed_gpipe_tensor_graph, gpipe_symbol_map_value, os.path.join(args.output_dir, args.comm_group_file), readable_rank_map_number_rank=readable_rank_map_number_rank)
    if args.chakra_schema_version == "v0.0.1":
        from symbolic_tensor_graph.chakra.backends.chakra_00_1_backend import Chakra001Backend as ReadoutBackend
    elif args.chakra_schema_version == "v0.0.4":
        from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import Chakra004Backend as ReadoutBackend
    elif args.chakra_schema_version == "json":
        from symbolic_tensor_graph.chakra.backends.json_backend import JsonBackend as ReadoutBackend
    else:
        assert False
    distributed_chakra_graph.readout(generated_filename, backend=ReadoutBackend)
    # distributed_gpipe_chakra_graph.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
    
