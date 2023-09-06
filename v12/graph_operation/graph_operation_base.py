from symbolic_graph import SymbolicGraph


class GraphOPBase:
    def __init__(self, requried_signature, removed_signature, added_signature):
        self.required_signature = list(requried_signature)
        self.removed_signature = list(removed_signature)
        self.added_signature = list(added_signature)

    def check_signature(self, graph: SymbolicGraph):
        graph.check_signature(keys=self.required_signature)

    def change_signature(self, graph: SymbolicGraph):
        for key in self.removed_signature:
            graph.keys.remove(key)
        for key in self.added_signature:
            graph.keys.append(key)
        return

    def apply(self, graph: SymbolicGraph):
        self.check_signature(graph)
        self.process(graph)
        self.added_signature(graph)
        return

    def process(self, graph: SymbolicGraph):
        raise NotImplementedError()
