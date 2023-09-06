from symbolic_graph import SymbolicGraph
from .graph_operation_base import GraphOPBase

id_cnt = 0


class GraphApplyID(GraphOPBase):
    def __init__(self, name_id_map=None):
        super(GraphApplyID, self).__init__(("x1_name", "x2_name"), (), ("tensor_id"))
        if name_id_map is None:
            name_id_map = dict()
        self.name_id_map = name_id_map

    def process(self, graph: SymbolicGraph):
        for tensor in graph.tensors:
            if tensor.name in self.name_id_map:
                id_ = self.name_id_map[tensor.name]
            else:
                id_ = id_cnt
                id_cnt += 1
            tensor["tensor_id"] = id_cnt
        return
