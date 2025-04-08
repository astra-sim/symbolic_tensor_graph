import copy

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.ops import Add, PlaceHolder


def group_query_attention(GQA_surrounding_path=None, GQA_kernel_path=None):
    if GQA_surrounding_path is None:
        GQA_surrounding_path = (
            "./sharding_spreadsheets/module3/group_query_attention_surrounding.csv"
        )
    if GQA_kernel_path is None:
        GQA_kernel_path = (
            "./sharding_spreadsheets/module3/group_query_attention_kernel.csv"
        )
    GQA_surrounding = TensorGraph.load_tensor_graph(GQA_surrounding_path)
    GQA_kernel = TensorGraph.load_tensor_graph(GQA_kernel_path)
    GQA_kernel = ReplicateGraph.apply(GQA_kernel, "attn_kernel.%s")
    links = dict()
    links["q"] = "attn_kernel.q"
    links["k"] = "attn_kernel.k"
    links["v"] = "attn_kernel.v"
    links["attn_kernel.dq"] = "dq"
    links["attn_kernel.dk"] = "dk"
    links["attn_kernel.dv"] = "dv"

    links["attn_kernel.qkv"] = "attn"
    links["dattn"] = "attn_kernel.dqkv"

    GQA = ConnectGraph.apply([GQA_surrounding, GQA_kernel], links)
    return GQA


def feed_forward_network(ffn_path=None):
    if ffn_path is None:
        ffn_path = "./sharding_spreadsheets/module3/llama_feed_forward_network.csv"
    ffn = ReplicateGraph.apply(TensorGraph.load_tensor_graph(ffn_path), "ffn.%s")
    return ffn


def transformer_decoder_block(ffn_path=None, layernorm_path=None, residual_path=None):
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module3/layer_norm.csv"
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module3/residual.csv"

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "input_norm.%s"
    )
    mha = ReplicateGraph.apply(group_query_attention(), "mha.%s")
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path), "mha_res.%s"
    )

    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path), "post_attn_norm.%s"
    )

    ffn = feed_forward_network(ffn_path)

    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path), "ffn_res.%s"
    )

    links = dict()
    # input_layernorm
    links["input_norm.y"] = "mha.x"
    # links["mha_dx"] = "input_norm_dy"

    # mha
    links["mha.o"] = "mha_res.x1"
    links["input_norm.x"] = "mha_res.x2"
    links["mha_res.dx1"] = "mha.do"
    # links["mha_res_dx2"] = "input_norm_dy"

    # mha res
    links["mha_res.y"] = "post_attn_norm.x"
    links["post_attn_norm.dx"] = "mha_res.dy"

    # post_layer_norm
    links["post_attn_norm.y"] = "ffn.x0"
    # links["ffn_dx0"] = "post_layer_norm_dy"

    # ffn
    links["ffn.xdown"] = "ffn_res.x1"
    links["post_attn_norm.x"] = "ffn_res.x2"
    links["ffn_res.dx1"] = "ffn.dxdown"
    # links["ffn_res_dx2"] = "post_layer_norm_dy"

    decoder_block = ConnectGraph.apply(
        [input_layernorm, mha, mha_res, post_layernorm, ffn, ffn_res], links
    )

    tensor_id_map_tensor = decoder_block.get_tensor_id_map_tensor()

    input_norm_dy = tensor_id_map_tensor["input_norm.dy@0"]
    assert input_norm_dy.op_type == PlaceHolder.type_name
    input_norm_dy.op_type = Add.type_name
    input_norm_dy.x1 = tensor_id_map_tensor["mha.dx@0"]
    input_norm_dy.x2 = tensor_id_map_tensor["mha_res.dx2@0"]
    input_norm_dy.x2_shape = copy.deepcopy(input_norm_dy.x1_shape)
    input_norm_dy.x2_hidden = copy.deepcopy(input_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(input_norm_dy)
    decoder_block.out_tensors.remove(input_norm_dy.x1)
    decoder_block.out_tensors.remove(input_norm_dy.x2)

    post_attn_norm_dy = tensor_id_map_tensor["post_attn_norm.dy@0"]
    assert post_attn_norm_dy.op_type == PlaceHolder.type_name
    post_attn_norm_dy.op_type = Add.type_name
    post_attn_norm_dy.x1 = tensor_id_map_tensor["ffn.dx0@0"]
    post_attn_norm_dy.x2 = tensor_id_map_tensor["ffn_res.dx2@0"]
    post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
    post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(post_attn_norm_dy)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x1)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x2)

    return decoder_block


def transformer_decoders(num_layers, decoder_template):
    links = dict()
    decoders = list()
    for i in range(num_layers):
        decoder = ReplicateGraph.apply(decoder_template, f"transformer.{i}.%s")
        decoders.append(decoder)
        if i > 0:
            links[f"transformer.{i-1}.ffn_res.y"] = f"transformer.{i}.input_norm.x"
            links[f"transformer.{i}.input_norm.dx"] = f"transformer.{i-1}.ffn_res.dy"

    decoders = ConnectGraph.apply(decoders, links)
    return decoders


def transformer(num_layers, embedding_path=None, regenerate=False):
    from . import CACHE_DIR
    import os

    cache_filename = os.path.join(CACHE_DIR, f"dense_{num_layers}.csv")
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    if embedding_path is None:
        embedding_path = "./sharding_spreadsheets/module3/embedding.csv"
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal"},
    )

    decoder_template = transformer_decoder_block()
    decoders = transformer_decoders(num_layers, decoder_template)

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links["transformer.0.input_norm.dx"] = "in_emb.dy"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"
    links["out_emb.dx"] = f"transformer.{num_layers-1}.ffn_res.dy"

    transformer = ConnectGraph.apply([decoders, in_emb, out_emb], links)

    transformer.save_tensor_graph(cache_filename)
    return transformer
