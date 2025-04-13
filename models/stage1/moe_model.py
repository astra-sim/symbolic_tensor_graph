import copy
import sympy as sp

from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.grad_updater import FSDPWeightGradManager
from symbolic_tensor_graph.ops import Add, PlaceHolder
from .llama_model import group_query_attention, transformer_decoders
from .utils import reduce_chain


def expert_branch(ffn_path=None, moe_wrapper_path=None):
    if ffn_path is None:
        ffn_path = "./sharding_spreadsheets/module3/tpsp/llama_feed_forward_network.csv"
    if moe_wrapper_path is None:
        moe_wrapper_path = "./sharding_spreadsheets/module3/tpsp/expert_wrapper.csv"

    ffn = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(ffn_path),
        "ffn.%s",
        old_symbol_map_new_symbol={"Seq": "Seq*KExperts/(Experts*ep)"},
    )
    moe_wrapper = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_wrapper_path),
        "ldis.%s",
    )

    expert = ConnectGraph.apply(
        [moe_wrapper, ffn],
        {
            "ldis.x_expert": "ffn.x0",
            "ffn.xdown": "ldis.y_expert",
            "ldis.dy_expert": "ffn.dxdown",
            "ffn.dx0": "ldis.dx_expert",
        },
    )
    return expert


def feed_forward_network(
    symbol_map_value, ffn_path=None, expert_wrapper_path=None, moe_frame_path=None
):
    if moe_frame_path is None:
        moe_frame_path = "./sharding_spreadsheets/module3/tpsp/moe_frame.csv"
    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    kexperts = symbol_map_value[kexperts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    assert experts_each_group == int(experts_each_group)
    experts_each_group = int(experts_each_group)

    expert = expert_branch(ffn_path, expert_wrapper_path)
    moe_frame = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(moe_frame_path), "moe.%s"
    )

    links = dict()
    branches = list()

    for i in range(experts_each_group):
        branches.append(ReplicateGraph.apply(expert, f"moe.{i}.%s"))
        # links["moe.xrouted"] = f"moe.{i}.ldis.x"        # one to multiple, need link multiple times
        # links["moe.dyrouted"] = f"moe.{i}.ldis.dy"
        # links[f"moe.{i}.ldis.dx"] = "moe.dxrouted"      # multiple to one, need reduce nodes
        # links[f"moe.{i}.ldis.y"] = "moe.yrouted"

    moe = ConnectGraph.apply([moe_frame] + branches, links)
    tensor_id_map_tensor = moe.get_tensor_id_map_tensor()

    # one to multiple
    moe_xrouted = tensor_id_map_tensor["moe.xrouted@0"]
    moe_dyrouted = tensor_id_map_tensor["moe.dyrouted@0"]
    for i in range(experts_each_group):
        links = dict()
        links["moe.xrouted"] = f"moe.{i}.ldis.x"
        links["moe.dyrouted"] = f"moe.{i}.ldis.dy"
        moe = ConnectGraph.apply([moe], links, inplace=True)
        moe.out_tensors.append(moe_xrouted)
        moe.out_tensors.append(moe_dyrouted)

    moe.out_tensors.remove(moe_xrouted)
    moe.out_tensors.remove(moe_dyrouted)

    # multiple to one
    to_be_reduce_moe_dxrouted = list()
    to_be_reduce_moe_yrouted = list()

    for i in range(experts_each_group):
        branch_ldis_dx = tensor_id_map_tensor[f"moe.{i}.ldis.dx@0"]
        to_be_reduce_moe_dxrouted.append(branch_ldis_dx)
        moe.out_tensors.remove(branch_ldis_dx)

        branch_ldis_y = tensor_id_map_tensor[f"moe.{i}.ldis.y@0"]
        to_be_reduce_moe_yrouted.append(branch_ldis_y)
        moe.out_tensors.remove(branch_ldis_y)

    # merge those reduce in a chain with 0 ops, which equavilent to a single node
    merged_dxrouted = reduce_chain(to_be_reduce_moe_dxrouted, "moe.dxrouted_r%d", amp=0)
    moe.tensors.extend(merged_dxrouted)
    if len(merged_dxrouted) > 0:
        merged_dxrouted[-1].op_attr = (
            "1"  # last node counts a whole elementwise op, which equalient to a single node
        )
        merged_dxrouted_last = merged_dxrouted[-1]
    else:
        assert len(to_be_reduce_moe_dxrouted) == 1
        merged_dxrouted_last = to_be_reduce_moe_dxrouted[0]
    moe.out_tensors.append(
        merged_dxrouted_last
    )  # add last node as output for future linkage

    merged_yrouted = reduce_chain(to_be_reduce_moe_yrouted, "moe.yrouted_r%d", amp=0)
    moe.tensors.extend(merged_yrouted)
    if len(merged_yrouted) > 0:
        merged_yrouted[-1].op_attr = "1"
        merged_yrouted_last = merged_yrouted[-1]
    else:
        assert len(to_be_reduce_moe_yrouted) == 1
        merged_yrouted_last = to_be_reduce_moe_yrouted[0]
    moe.out_tensors.append(merged_yrouted_last)

    links = {
        merged_dxrouted_last.name: "moe.dxrouted",
        merged_yrouted_last.name: "moe.yrouted",
    }
    moe = ConnectGraph.apply([moe], links)
    return moe


def transformer_decoder_block(
    symbol_map_value, layernorm_path=None, residual_path=None
):
    if layernorm_path is None:
        layernorm_path = "./sharding_spreadsheets/module3/tpsp/layer_norm.csv"
    if residual_path is None:
        residual_path = "./sharding_spreadsheets/module3/tpsp/residual.csv"

    input_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "input_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp*ep"},
    )
    mha = ReplicateGraph.apply(
        group_query_attention(), "mha.%s", old_symbol_map_new_symbol={"tp": "tp*ep"}
    )
    mha_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "mha_res.%s",
        old_symbol_map_new_symbol={"tp": "tp*ep"},
    )

    post_layernorm = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(layernorm_path),
        "post_attn_norm.%s",
        old_symbol_map_new_symbol={"tp": "tp*ep"},
    )

    ffn = feed_forward_network(symbol_map_value)

    ffn_res = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(residual_path),
        "ffn_res.%s",
        old_symbol_map_new_symbol={"tp": "tp*ep"},
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
    links["post_attn_norm.y"] = "moe.x"
    # links["moe_dx"] = "post_layer_norm_dy"

    # ffn
    links["moe.y"] = "ffn_res.x1"
    links["post_attn_norm.x"] = "ffn_res.x2"
    links["ffn_res.dx1"] = "moe.dy"
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
    post_attn_norm_dy.x1 = tensor_id_map_tensor["moe.dx@0"]
    post_attn_norm_dy.x2 = tensor_id_map_tensor["ffn_res.dx2@0"]
    post_attn_norm_dy.x2_shape = copy.deepcopy(post_attn_norm_dy.x1_shape)
    post_attn_norm_dy.x2_hidden = copy.deepcopy(post_attn_norm_dy.x1_hidden)
    decoder_block.in_tensors.remove(post_attn_norm_dy)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x1)
    decoder_block.out_tensors.remove(post_attn_norm_dy.x2)

    decoder_block = FSDPWeightGradManager.apply(decoder_block)

    return decoder_block


def transformer(num_layers, symbol_map_value, embedding_path=None, regenerate=False):
    from . import CACHE_DIR
    import os

    experts, kexperts, ep = sp.symbols("Experts KExperts ep")
    experts = symbol_map_value[experts]
    kexperts = symbol_map_value[kexperts]
    ep = symbol_map_value[ep]
    experts_each_group = experts / ep
    cache_filename = os.path.join(
        CACHE_DIR, f"moe_{num_layers}_{experts_each_group}.csv"
    )
    if os.path.exists(cache_filename) and not regenerate:
        return TensorGraph.load_tensor_graph(cache_filename)

    if embedding_path is None:
        embedding_path = "./sharding_spreadsheets/module3/tpsp/embedding.csv"
    in_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "in_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dvocal", "Dout": "Dmodel", "tp": "tp*ep"},
    )
    out_emb = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph(embedding_path),
        "out_emb.%s",
        old_symbol_map_new_symbol={"Din": "Dmodel", "Dout": "Dvocal", "tp": "tp*ep"},
    )

    decoder_template = transformer_decoder_block(symbol_map_value)
    decoders = transformer_decoders(num_layers, decoder_template)

    links = dict()
    links["in_emb.y"] = "transformer.0.input_norm.x"
    links["transformer.0.input_norm.dx"] = "in_emb.dy"
    links[f"transformer.{num_layers-1}.ffn_res.y"] = "out_emb.x"
    links["out_emb.dx"] = f"transformer.{num_layers-1}.ffn_res.dy"

    transformer = ConnectGraph.apply([decoders, in_emb, out_emb], links)

    loss = ReplicateGraph.apply(
        TensorGraph.load_tensor_graph("./sharding_spreadsheets/module3/tpsp/loss.csv"),
        "loss.%s",
        old_symbol_map_new_symbol={"tp": "tp*ep"},
    )
    links = dict()
    links["out_emb.y"] = "loss.y"
    links["loss.dy"] = "out_emb.dy"
    transformer = ConnectGraph.apply([transformer, loss], links)

    transformer.save_tensor_graph(cache_filename)
    return transformer
