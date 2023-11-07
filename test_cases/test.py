import sympy as sp
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.coll_comm_matcher import CommunicationMatcher
from symbolic_tensor_graph.graph.grad_updater import GradUpdater
from symbolic_tensor_graph.graph.convert_chakra import ConvertChakra
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.chakra.node import Node
from symbolic_tensor_graph.chakra.backends.json_backend import JsonBackend
from models.transformer import (
    # transformer as transformer_fn,
    transformer_stack as transformer_stack_fn,
    transformer_stacks as transformer_fn,
)


def test1():
    paths = [
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv",
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv",
        "./sharding_spreadsheets/module/divya/reshape.csv",
        "./sharding_spreadsheets/module/divya/linear.csv",
    ]

    mp, dp = sp.symbols("mp dp")
    parallel_dims = [mp, dp]

    for path in paths:
        print(path)
        graph = TensorGraph.load_tensor_graph(path)
        for tensor in graph.tensors:
            print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
            if tensor.x1 is not None:
                from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
                to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
                print(
                    f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
                )
            if tensor.x2 is not None:
                from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
                to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
                print(
                    f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
                )
        updated_graph = GradUpdater.apply(graph)
        for tensor in updated_graph.tensors:
            print(tensor)


def test2():
    dp, mp = sp.symbols("dp mp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 32,
        mp: 32,
    }

    parallel_dims = [dp, mp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, 2)
    transformer = transformer_fn(stack, 2)
    transformer_updated_grad = GradUpdater.apply(transformer)
    for tensor in transformer_updated_grad.tensors:
        print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
        if tensor.x1 is not None:
            from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
            to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
            print(
                f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )
        if tensor.x2 is not None:
            from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
            to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
            print(
                f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )

    chakra_nodes = ConvertChakra.apply(
        transformer_updated_grad, symbol_map_value, parallel_dims
    )

    nodes = chakra_nodes.get_nodes()
    Node.readout_nodes(nodes, "test.0.eg")
    Node.readout_nodes(nodes, "test.json", backend=JsonBackend)


def test3():
    dp, mp = sp.symbols("dp mp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 1024,
        mp: 1,
    }

    parallel_dims = [dp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, 2)
    transformer = transformer_fn(stack, 2)
    transformer_updated_grad = GradUpdater.apply(transformer)
    for tensor in transformer_updated_grad.tensors:
        print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
        if tensor.x1 is not None:
            from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
            to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
            print(
                f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )
        if tensor.x2 is not None:
            from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
            to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
            print(
                f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )

    chakra_nodes = ConvertChakra.apply(
        transformer_updated_grad, symbol_map_value, parallel_dims
    )

    nodes = chakra_nodes.get_nodes()
    Node.readout_nodes(nodes, "test.0.eg")
    Node.readout_nodes(nodes, "test.json", backend=JsonBackend)


def test4():
    dp, mp = sp.symbols("dp mp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 32,
        mp: 32,
    }

    parallel_dims = [dp, mp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, 2)
    transformer = transformer_fn(stack, 2)
    transformer_updated_grad = GradUpdater.apply(transformer)
    transformer_updated_grad_c2 = ReplicateGraph.apply(
        transformer_updated_grad, new_revision=lambda old_revision: f"c2&{old_revision}"
    )
    transformer_updated_grad_2 = ConnectGraph.apply(
        [transformer_updated_grad, transformer_updated_grad_c2], {}
    )
    for tensor in transformer_updated_grad_2.tensors:
        print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
        if tensor.x1 is not None:
            from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
            to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
            print(
                f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )
        if tensor.x2 is not None:
            from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
            to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
            print(
                f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )

    chakra_nodes = ConvertChakra.apply(
        transformer_updated_grad_2, symbol_map_value, parallel_dims
    )

    nodes = chakra_nodes.get_nodes()
    Node.readout_nodes(nodes, "test.0.eg")
    Node.readout_nodes(nodes, "test.json", backend=JsonBackend)


if __name__ == "__main__":
    test4()
