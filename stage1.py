import os
import argparse
import copy
import sympy as sp
from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.ops import Add, PlaceHolder
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
    parser.add_argument("--din", type=int, default=32000, required=False)
    parser.add_argument("--dout", type=int, default=32000, required=False)
    parser.add_argument("--dmodel", type=int, default=8192, required=False)
    parser.add_argument("--dff", type=int, default=28672, required=False)
    parser.add_argument("--batch", type=int, default=1024, required=False)
    parser.add_argument("--seq", type=int, default=1024, required=False)
    parser.add_argument("--head", type=int, default=64, required=False)
    parser.add_argument("--kvhead", type=int, default=8, required=False)
    parser.add_argument("--num_stacks", type=int, default=80, required=False)
    parser.add_argument(
        "--chakra_schema_version", type=str, default="v0.0.4", required=False
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not "%d" in args.output_name:
        args.output_name = f"{args.output_name}.%d.et"
    generated_filename = os.path.join(args.output_dir, args.output_name)
    dp, tp, pp, spp = sp.symbols("dp tp pp sp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head, KVHead = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head KVHead"
    )
    symbol_map_value = {
        Din: args.din,
        Dout: args.dout,
        Dmodel: args.dmodel,
        Dff: args.dff,
        Batch: args.batch,
        Seq: args.seq,
        Head: args.head,
        KVHead: args.kvhead,
        dp: args.dp,
        tp: args.tp,
        pp: args.pp,
        spp: args.sp,
    }
    num_stacks = args.num_stacks
    spatial_parallel_dims = [dp, tp, spp]
    temporal_parallel_dims = [pp]

    def group_query_attention():
        GQA_surrounding_path = (
            "./sharding_spreadsheets/module3/group_query_attention_surrounding.csv"
        )
        GQA_kernel_path = (
            "./sharding_spreadsheets/module3/group_query_attention_kernel.csv"
        )
        GQA_surrounding = TensorGraph.load_tensor_graph(GQA_surrounding_path)
        GQA_kernel = TensorGraph.load_tensor_graph(GQA_kernel_path)
        GQA_kernel = ReplicateGraph.apply(GQA_kernel, "attn_kernel_%s")
        links = dict()
        links["q"] = "attn_kernel_q"
        links["k"] = "attn_kernel_k"
        links["v"] = "attn_kernel_v"
        links["attn_kernel_dq"] = "dq"
        links["attn_kernel_dk"] = "dk"
        links["attn_kernel_dv"] = "dv"

        links["attn_kernel_qkv"] = "attn"
        links["dattn"] = "attn_kernel_dqkv"

        GQA = ConnectGraph.apply([GQA_surrounding, GQA_kernel], links)
        return GQA

    def transformer_decoder_block():
        ffn_path = "./sharding_spreadsheets/module3/llama_feed_forward_network.csv"
        layernorm_path = "./sharding_spreadsheets/module3/layer_norm.csv"
        residual_path = "./sharding_spreadsheets/module3/residual.csv"
        loss_path = "./sharding_spreadsheets/module3/loss.csv"

        input_layernorm = ReplicateGraph.apply(
            TensorGraph.load_tensor_graph(layernorm_path), "input_norm_%s"
        )
        mha = ReplicateGraph.apply(group_query_attention(), "mha_%s")
        mha_res = ReplicateGraph.apply(
            TensorGraph.load_tensor_graph(residual_path), "mha_res_%s"
        )

        post_layernorm = ReplicateGraph.apply(
            TensorGraph.load_tensor_graph(layernorm_path), "post_attn_norm_%s"
        )
        ffn = ReplicateGraph.apply(TensorGraph.load_tensor_graph(ffn_path), "ffn_%s")
        ffn_res = ReplicateGraph.apply(
            TensorGraph.load_tensor_graph(residual_path), "ffn_res_%s"
        )

        links = dict()
        # input_layernorm
        links["input_norm_y"] = "mha_x"
        # links["mha_dx"] = "input_norm_dy"

        # mha
        links["mha_o"] = "mha_res_x1"
        links["input_norm_x"] = "mha_res_x2"
        links["mha_res_dx1"] = "mha_do"
        # links["mha_res_dx2"] = "input_norm_dy"

        # mha res
        links["mha_res_y"] = "post_attn_norm_x"
        links["post_attn_norm_dx"] = "mha_res_dy"

        # post_layer_norm
        links["post_attn_norm_y"] = "ffn_x0"
        # links["ffn_dx0"] = "post_layer_norm_dy"

        # ffn
        links["ffn_xdown"] = "ffn_res_x1"
        links["post_attn_norm_x"] = "ffn_res_x2"
        links["ffn_res_dx1"] = "ffn_dxdown"
        # links["ffn_res_dx2"] = "post_layer_norm_dy"

        decoder_block = ConnectGraph.apply(
            [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res], links
        )

        tensor_id_map_tensor = decoder_block.get_tensor_id_map_tensor()

        input_norm_dy = tensor_id_map_tensor["input_norm_dy@0"]
        assert input_norm_dy.op_type == PlaceHolder.type_name
        input_norm_dy.op_type = Add.type_name
        input_norm_dy.x1 = tensor_id_map_tensor["mha_dx@0"]
        input_norm_dy.x2 = tensor_id_map_tensor["mha_res_dx2@0"]
        input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
        input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
        decoder_block.in_tensors.remove(input_norm_dy)
        decoder_block.out_tensors.remove(input_norm_dy.x1)
        decoder_block.out_tensors.remove(input_norm_dy.x2)

        post_attn_norm_dy = tensor_id_map_tensor["post_attn_norm_dy@0"]
        assert post_attn_norm_dy.op_type == PlaceHolder.type_name
        post_attn_norm_dy.op_type = Add.type_name
        post_attn_norm_dy.x1 = tensor_id_map_tensor["ffn_dx0@0"]
        post_attn_norm_dy.x2 = tensor_id_map_tensor["ffn_res_dx2@0"]
        post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
        post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
        decoder_block.in_tensors.remove(post_attn_norm_dy)
        decoder_block.out_tensors.remove(post_attn_norm_dy.x1)
        decoder_block.out_tensors.remove(post_attn_norm_dy.x2)

        loss = ReplicateGraph.apply(TensorGraph.load_tensor_graph(loss_path), "loss_%s")
        links = dict()
        links["ffn_res_y"] = "loss_y"
        links["loss_dy"] = "ffn_res_dy"
        decoder_block = ConnectGraph.apply([decoder_block, loss], links)

        return decoder_block

    decode_block = transformer_decoder_block()
    decode_block.visualize("decoder")
    decode_block.save_tensor_graph("decoder.csv")
    hook = 1

    pipeline_tensor_map = dict()
    for tensor in decode_block.tensors:
        pipeline_tensor_map[tensor.id] = {pp: 0}

    distributed_tensor_graph = GraphDistributer.apply(
        decode_block,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        pipeline_tensor_map,
    )

    comm_group_file = args.output_name.replace(".%d", "").replace(".et", "")
    distributed_chakra_graph = BundledConvertChakra.apply(
        distributed_tensor_graph,
        symbol_map_value,
        os.path.join(args.output_dir, comm_group_file),
    )

    from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend import (
        Chakra004Backend as ReadoutBackend,
    )

    distributed_chakra_graph.readout(generated_filename, backend=ReadoutBackend)


if __name__ == "__main__":
    main()
