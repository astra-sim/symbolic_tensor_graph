import os, sys

from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.graph_linker import GraphLinker
from symbolic_tensor_graph.grad_updater import GradUpdater
from symbolic_tensor_graph.offload_strategy import OffloadStrategy


def transformer_stack(root="transformer/sharding_spreadsheets/dp", visualize=False):
    mha_fwd = Tensor.parse_records(
        os.path.join(root, "graphs/multiHeadAttentionFwd.csv")
    )
    mha_bwd = Tensor.parse_records(
        os.path.join(root, "graphs/multiHeadAttentionBwd.csv")
    )
    ffn_fwd = Tensor.parse_records(
        os.path.join(root, "graphs/feedForwardNetworkFwd.csv")
    )
    ffn_bwd = Tensor.parse_records(
        os.path.join(root, "graphs/feedForwardNetworkBwd.csv")
    )

    mha_fwd = GraphLinker.prefix_graph_impl(mha_fwd, "mha")
    mha_bwd = GraphLinker.prefix_graph_impl(mha_bwd, "mha")
    ffn_fwd = GraphLinker.prefix_graph_impl(ffn_fwd, "ffn")
    ffn_bwd = GraphLinker.prefix_graph_impl(ffn_bwd, "ffn")

    if visualize:
        Tensor.visualize(mha_fwd, os.path.join(root, f"visualization/mha_fwd"))
        Tensor.visualize(mha_bwd, os.path.join(root, f"visualization/mha_bwd"))
        Tensor.visualize(ffn_fwd, os.path.join(root, f"visualization/ffn_fwd"))
        Tensor.visualize(ffn_bwd, os.path.join(root, f"visualization/ffn_bwd"))

    fwd = GraphLinker.link_graph_impl([mha_fwd, ffn_fwd], {"mha_norm": "ffn_x0"})
    bwd = GraphLinker.link_graph_impl([mha_bwd, ffn_bwd], {"ffn_d_x0": "mha_d_norm"})

    if visualize:
        Tensor.visualize(fwd, os.path.join(root, "visualization/stack_fwd"))
        Tensor.visualize(bwd, os.path.join(root, "visualization/stack_bwd"))

    Tensor.to_records(fwd, os.path.join(root, "processed_graphs/stackFwd.csv"))
    Tensor.to_records(bwd, os.path.join(root, "processed_graphs/stackBwd.csv"))
    return


def transformer_stacks(num_stacks, root="sharding_spreadsheets/dp", visualize=False):
    fwd_graphs = list()
    bwd_graphs = list()
    for i in range(num_stacks):
        if not os.path.exists(os.path.join(root, "processed_graphs/stackFwd.csv")):
            transformer_stack(root=root, visualize=visualize)
        assert os.path.exists(os.path.join(root, "processed_graphs/stackBwd.csv"))
        fwd = Tensor.parse_records(os.path.join(root, "processed_graphs/stackFwd.csv"))
        bwd = Tensor.parse_records(os.path.join(root, "processed_graphs/stackBwd.csv"))

        fwd = GraphLinker.prefix_graph_impl(fwd, f"stack{i}")
        bwd = GraphLinker.prefix_graph_impl(bwd, f"stack{i}")
        fwd_graphs.append(fwd)
        bwd_graphs.append(bwd)

    fwd_links = dict()
    bwd_links = dict()
    for i in range(num_stacks - 1):
        fwd_links[f"stack{i}_ffn_norm"] = f"stack{i+1}_mha_x"
        bwd_links[f"stack{i+1}_mha_d_x"] = f"stack{i}_ffn_d_norm"
    fwd = GraphLinker.link_graph_impl(fwd_graphs, fwd_links)
    bwd = GraphLinker.link_graph_impl(bwd_graphs, bwd_links)

    if visualize:
        Tensor.visualize(
            fwd, os.path.join(root, f"visualization/stack_{num_stacks}_fwd_stacks")
        )
        Tensor.visualize(
            bwd, os.path.join(root, f"visualization/stack_{num_stacks}_bwd_stacks")
        )
    Tensor.to_records(
        fwd, os.path.join(root, f"processed_graphs/stack{num_stacks}Fwd.csv")
    )
    Tensor.to_records(
        bwd, os.path.join(root, f"processed_graphs/stack{num_stacks}Bwd.csv")
    )


def transformer(num_stacks, root="sharding_spreadsheets/dp", visualize=False):
    os.makedirs(os.path.join(root, f"processed_graphs"), exist_ok=True)
    print(f"transformer {num_stacks} {root} {visualize}")
    if not os.path.exists(
        os.path.join(root, f"processed_graphs/stack{num_stacks}Fwd.csv")
    ):
        transformer_stacks(num_stacks, root=root, visualize=visualize)
    assert os.path.exists(
        os.path.join(root, f"processed_graphs/stack{num_stacks}Bwd.csv")
    )
    stacks_fwd = Tensor.parse_records(
        os.path.join(root, f"processed_graphs/stack{num_stacks}Fwd.csv")
    )
    stacks_bwd = Tensor.parse_records(
        os.path.join(root, f"processed_graphs/stack{num_stacks}Bwd.csv")
    )
    in_embed_fwd = Tensor.parse_records(os.path.join(root, "graphs/inEmbedFwd.csv"))
    in_embed_bwd = Tensor.parse_records(os.path.join(root, "graphs/inEmbedBwd.csv"))
    out_embed_fwd = Tensor.parse_records(os.path.join(root, "graphs/outEmbedFwd.csv"))
    out_embed_bwd = Tensor.parse_records(os.path.join(root, "graphs/outEmbedBwd.csv"))

    fwd_links = dict()
    bwd_links = dict()
    fwd_links["inEmbY"] = "stack0_mha_x"
    fwd_links[f"stack{num_stacks-1}_ffn_norm"] = "outEmbedX"
    bwd_links["d_outEmbedX"] = f"stack{num_stacks-1}_ffn_d_norm"
    bwd_links["stack0_mha_d_x"] = "d_inEmbY"
    fwd = GraphLinker.link_graph_impl(
        [stacks_fwd, in_embed_fwd, out_embed_fwd], fwd_links
    )
    bwd = GraphLinker.link_graph_impl(
        [stacks_bwd, in_embed_bwd, out_embed_bwd], bwd_links
    )

    if visualize:
        Tensor.visualize(
            fwd, os.path.join(root, f"visualization/transformer_{num_stacks}_fwd")
        )
        Tensor.visualize(
            bwd, os.path.join(root, f"visualization/transformer_{num_stacks}_bwd")
        )
    Tensor.to_records(
        fwd, os.path.join(root, f"processed_graphs/transformer_{num_stacks}_fwd.csv")
    )
    Tensor.to_records(
        bwd, os.path.join(root, f"processed_graphs/transformer_{num_stacks}_bwd.csv")
    )

    grad_updater = GradUpdater(fwd, bwd)
    update_tensors = grad_updater.update_tensors()
    loop_links = dict()
    loop_links["outEmbedY"] = "d_outEmbedY"
    # loop = GraphLinker.link_graph_impl([fwd, bwd, update_tensors], loop_links)
    loop = GraphLinker.link_graph_impl([fwd, bwd], loop_links)
    if visualize:
        # Tensor.visualize(loop, os.path.join(root, f"visualization/transformer_{num_stacks}"))
        pass
    Tensor.to_records(
        loop, os.path.join(root, f"processed_graphs/transformer_{num_stacks}.csv")
    )
    print(f"transformer done {num_stacks} {root} {visualize}")
    return


def transformer_offload_strategy(
    num_stacks,
    root="sharding_spreadsheets/dp",
    weight_offload=1,
    leaf_offload=1,
    input_offload=0,
):
    print(
        f"transformer_offload_strategy {num_stacks} {root} {weight_offload} {leaf_offload} {input_offload}"
    )
    transformer_csv = os.path.join(
        root, "processed_graphs", f"transformer_{num_stacks}.csv"
    )
    if not os.path.exists(transformer_csv):
        transformer(num_stacks, root=root)
    os.makedirs(os.path.join(root, "offload_strategy"), exist_ok=True)
    offload_csv = os.path.join(
        root,
        "offload_strategy",
        f"transformer_{num_stacks}_w{weight_offload}_l{leaf_offload}_i{input_offload}.csv",
    )
    tensors = Tensor.parse_records(transformer_csv)
    offload_strategy = OffloadStrategy.create_blank(tensors)
    if weight_offload > 0:
        offload_strategy.set_all_weight_offload(tensors, weight_offload)
    if leaf_offload > 0:
        offload_strategy.set_all_leaf_offload(tensors, leaf_offload)
    if input_offload > 0:
        offload_strategy.set_all_intermediate_offload(tensors, input_offload)
    offload_strategy.to_records(offload_csv)
    print(
        f"transformer_offload_strategy done {num_stacks} {root} {weight_offload} {leaf_offload} {input_offload}"
    )
    return


if __name__ == "__main__":
    transformer(2)
    transformer(8)
    transformer(16)
    transformer(32)
    transformer(64)
    transformer(96)
