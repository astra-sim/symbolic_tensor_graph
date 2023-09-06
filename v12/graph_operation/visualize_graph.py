import graphviz

from symbolic_graph import SymbolicGraph
from .graph_operation_base import GraphOPBase


class GraphVisualizer(GraphOPBase):
    def __init__(self, filename: str, format="pdf"):
        super(GraphVisualizer, self).__init__(
            ("tensor_id", "tensor_name", "x1", "x2"), (), ()
        )
        self.filename = filename
        self.format = format

    def process(self, graph: SymbolicGraph):
        f = graphviz.Digraph()
        for tensor in graph.tensors:
            name = f"{tensor.tensor_id}: {tensor.name}"
            label = f"{tensor.tensor_id}: {tensor.name}"
            if "revision" in tensor.keys():
                label += f" rev{tensor.revision}"
            f.node(name=name, lable=label, id=tensor.tensor_id, shape="box")
            if tensor.x1 is not None:
                f.edge(tensor.x1, tensor.tensor_id)
            if tensor.x2 is not None:
                f.edge(tensor.x2, tensor.tensor_id)
        f.render(self.filename, format=self.format, cleanup=True)
