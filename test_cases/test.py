import sympy as sp
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.coll_comm_matcher import CommunicationMatcher
from symbolic_tensor_graph.graph.grad_updater import GradUpdater


if __name__ == "__main__":
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
