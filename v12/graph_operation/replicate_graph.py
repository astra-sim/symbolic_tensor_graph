import copy
from .graph_operation_base import GraphOPBase
from .apply_id import id_cnt
from symbolic_graph import SymbolicGraph


class ReplicateGraph(GraphOPBase):
    def __init__(self, name_id_map=None):
        super(ReplicateGraph, self).__init__(("tensor_id", "x1", "x2"), (), ())
        self.name_id_map = name_id_map

    def process(self, graph: SymbolicGraph):
        old_new_id_map = dict()
        keys = list()
        tensors = list()
        for key in graph.keys():
            keys.append(key)
        for tensor in graph.tensors:
            tensor = copy.deepcopy(tensor)
            old_id = tensor.tensor_id
            if tensor.tensor_name in self.name_id_map:
                new_id = self.name_id_map[tensor.tensor_name]
            else:
                new_id = id_cnt
                id_cnt += 1
            old_new_id_map[old_id] = new_id
            tensor.tensor_id = new_id
            tensors.append(tensor)
        for tensor in tensors:
            if tensor.x1 in old_new_id_map:
                tensor.x1 = old_new_id_map[tensor.x1]
            if tensor.x2 in old_new_id_map:
                tensor.x2 = old_new_id_map[tensor.x2]
        # create empty graph
        new_graph = SymbolicGraph(csv_filename=None)
        new_graph.keys = keys
        new_graph.tensors = tensors
        return new_graph
