import copy
from .graph import TensorGraph
from ..ops import PlaceHolder, Identical


class ConnectGraph:
    @classmethod
    def apply(cls, graphs, links, inplace=False, force_connect=False):
        if not inplace:
            graphs = copy.deepcopy(graphs)
        connected_tensors = dict()
        connected_in_tensors = list()
        connected_out_tensors = list()
        connected_graph = None
        # for tensors with same name, preserve only one, and reupdate the links
        for graph in graphs:
            for tensor in graph.tensors:
                if tensor.id in connected_tensors:
                    continue
                connected_tensors[tensor.id] = tensor
                if tensor in graph.in_tensors:
                    connected_in_tensors.append(tensor)
                if tensor in graph.out_tensors:
                    connected_out_tensors.append(tensor)
        for tensor in connected_tensors.values():
            if tensor.x1 is not None and tensor.x1.id in connected_tensors:
                tensor.x1 = connected_tensors[tensor.x1.id]
            if tensor.x2 is not None and tensor.x2.id in connected_tensors:
                tensor.x2 = connected_tensors[tensor.x2.id]
        connected_graph = TensorGraph(
            list(connected_tensors.values()), connected_in_tensors, connected_out_tensors
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
            assert to_.op_type == PlaceHolder.type_name or force_connect
            # should have some machnisum to ensure they have same shape except parallel shardings
            # assert from_.y_shape == to_.x1_shape
            to_.op_type = Identical.type_name
            to_.x1 = from_
            connected_graph.in_tensors.remove(to_)
            connected_graph.out_tensors.remove(from_)
        return connected_graph
