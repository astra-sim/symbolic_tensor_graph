from .graph_operation_base import GraphOPBase
from symbolic_graph import SymbolicGraph


class ChangeTensorName(GraphOPBase):
    def __init__(self, template):
        super(ChangeTensorName, self).__init__(("tensor_name",), (), ())
        self.template = template

    def process(self, graph: SymbolicGraph):
        for tensor in graph.tensors:
            tensor.tensor_name = self.template % (tensor.tensor_name,)
        return
