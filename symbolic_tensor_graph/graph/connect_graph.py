import copy
from .graph import TensorGraph
from ..ops import PlaceHolder


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
        connected_parent_map_child = connected_graph.get_tensor_parent_to_child_link()
        for from_, to_ in links:
            if isinstance(from_, str):
                assert isinstance(to_, str)
                from_ = connected_graph_id_map_graph[from_]
                to_ = connected_graph_id_map_graph[to_]
            assert to_.op_type == PlaceHolder.type_name
            for child in connected_parent_map_child[to_]:
                if child.x1 == to_:
                    child.x1 = from_
                elif child.x2 == to_:
                    child.x2 = from_
                else:
                    assert False
            connected_graph.tensors.remove(to_)
            connected_graph.in_tensors.remove(to_)
            connected_graph.out_tensors.remove(from_)
        return connected_graph
