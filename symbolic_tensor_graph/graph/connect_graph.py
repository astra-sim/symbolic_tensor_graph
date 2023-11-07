import copy
from .graph import TensorGraph
from ..ops import PlaceHolder, Identical


class ConnectGraph:
    @classmethod
    def apply(cls, graphs, links, inplace=False):
        if not inplace:
            graphs = copy.deepcopy(graphs)
        connected_tensors = list()
        connected_in_tensors = list()
        connected_out_tensors = list()
        connected_graph = None
        for graph in graphs:
            connected_tensors.extend(graph.tensors)
            connected_in_tensors.extend(graph.in_tensors)
            connected_out_tensors.extend(graph.out_tensors)
            connected_graph = TensorGraph(
                connected_tensors, connected_in_tensors, connected_out_tensors
            )
        assert connected_graph is not None
        connected_graph_id_map_graph = connected_graph.get_tensor_id_map_tensor()
        for from_, to_ in links.items():
            if isinstance(from_, str):
                assert isinstance(to_, str)
                if not "@" in from_:
                    from_ += "@0"
                if not "@" in to_:
                    to_ += "@0"
                from_ = connected_graph_id_map_graph[from_]
                to_ = connected_graph_id_map_graph[to_]
            assert to_.op_type == PlaceHolder.type_name
            # should have some machnisum to ensure they have same shape except parallel shardings
            # assert from_.y_shape == to_.x1_shape
            to_.op_type = Identical.type_name
            to_.x1 = from_
            connected_graph.in_tensors.remove(to_)
            connected_graph.out_tensors.remove(from_)
        return connected_graph
