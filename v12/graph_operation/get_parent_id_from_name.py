from .graph_operation_base import GraphOPBase
from symbolic_graph import SymbolicGraph


class GetParentIdFromName(GraphOPBase):
    def __init__(self):
        super(GetParentIdFromName, self).__init__(
            ("tensor_name", "tensor_id", "x1_name", "x2_name"), (), ("x1", "x2")
        )

    def process(self, graph: SymbolicGraph):
        name_id_map = dict()
        for tensor in graph.tensors:
            tensor_id, tensor_name = tensor.tensor_id, tensor.tensor_name
            assert tensor_name not in name_id_map  # no tensor with same name
            name_id_map[tensor_name] = tensor_id

        for tensor in graph.tensors:
            if tensor.x1_name is not None:
                assert tensor.x1_name in name_id_map
                tensor["x1"] = name_id_map[tensor.x1_name]
            if tensor.x2_name is not None:
                assert tensor.x2_name in name_id_map
                tensor["x2"] = name_id_map[tensor.x2_name]
        return
