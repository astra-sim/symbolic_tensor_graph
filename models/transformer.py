from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.connect_graph import ConnectGraph


def transformer_stack(mha, ffn):
    mha = ReplicateGraph.apply(mha, "mha_%s")
    ffn = ReplicateGraph.apply(ffn, "ffn_%s")

    stack = ConnectGraph.apply(
        [mha, ffn], {"mha_norm": "ffn_x0", "ffn_dx0": "mha_d_norm"}
    )
    return stack


def transformer_stacks(stack, num_stacks):
    graphs = list()
    links = dict()
    for num_stack in range(num_stacks):
        graphs.append(ReplicateGraph.apply(stack, f"stack_{num_stack}_%s"))
        if num_stack == 0:
            pass
        elif num_stack > num_stacks - 1:
            links[f"stack_{num_stack-1}_ffn_norm"] = f"stack_{num_stack}_mha_x"
            links[f"stack_{num_stack}_mha_d_x"] = f"stack_{num_stack-1}_ffn_dnorm"
        else:
            links[f"stack_{num_stack-1}_ffn_norm"] = f"stack_{num_stack}_mha_x"
            links[f"stack_{num_stack}_mha_d_x"] = f"stack_{num_stack-1}_ffn_dnorm"
    transformer_graph = ConnectGraph.apply(graphs, links)
    return transformer_graph


def transformer(in_emb, out_emb, stack, num_stacks):
    graphs = list()
    links = dict()
    in_emb = ReplicateGraph.apply(
        in_emb, "in_emb_%s", old_symbol_map_new_symbol={"Dout": "Dmodel"}
    )
    out_emb = ReplicateGraph.apply(
        out_emb, "out_emb_%s", old_symbol_map_new_symbol={"Din": "Dmodel"}
    )
    graphs.append(in_emb)
    graphs.append(out_emb)
    for num_stack in range(num_stacks):
        graphs.append(ReplicateGraph.apply(stack, f"stack_{num_stack}_%s"))
        if num_stack == 0:
            links["in_emb_y"] = "stack_0_mha_x"
            links["stack_0_mha_d_x"] = "in_emb_dy"
        elif num_stack > num_stacks - 1:
            links[f"stack_{num_stack-1}_ffn_norm"] = f"stack_{num_stack}_mha_x"
            links[f"stack_{num_stack}_mha_d_x"] = f"stack_{num_stack-1}_ffn_dnorm"
        else:
            links[f"stack_{num_stack-1}_ffn_norm"] = f"stack_{num_stack}_mha_x"
            links[f"stack_{num_stack}_mha_d_x"] = f"stack_{num_stack-1}_ffn_dnorm"
            links[f"stack_{num_stack}_ffn_norm"] = "out_emb_x"
            links["out_emb_dx"] = f"stack_{num_stack}_ffn_dnorm"
    transformer_graph = ConnectGraph.apply(graphs, links)
    return transformer_graph
