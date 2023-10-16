from .scheduler import SchedulerBase


class GreedyScheduler(SchedulerBase):
    def __init__(self, eg_nodes, node_runtime=None, num_queue=1, inplace=False):
        super(GreedyScheduler, self).__init__(
            eg_nodes, node_runtime, num_queue, inplace
        )
        self.pending_nodes = list()
        self.finished_nodes = list()
        self.issuable_nodes = list()
        self.node_map_unresolved_parents = dict()

        for node in self.nodes:
            self.pending_nodes.append(node)
            if len(node.parent) == 0:
                self.issuable_nodes.append(node)
            self.node_map_unresolved_parents[node.id] = list()
            for parent in node.parent:
                self.node_map_unresolved_parents[node.id].append(parent)

    def issue_node(self, node):
        super(GreedyScheduler, self).issue_node(node)

        self.pending_nodes.remove(node)
        self.issuable_nodes.remove(node)
        self.finished_nodes.append(node)
        for child_id in self.parent_child_map[node.id]:
            child = self.id_node_map[child_id]
            assert node.id in self.node_map_unresolved_parents[child_id]
            self.node_map_unresolved_parents[child_id].remove(node.id)
            if len(self.node_map_unresolved_parents[child_id]) == 0:
                if not child in self.issuable_nodes:
                    self.issuable_nodes.append(child)

    def try_issue_node(self, node):
        assert node in self.pending_nodes
        for parent_id in node.parent:
            parent_node = self.id_node_map[parent_id]
            if not parent_node in self.finished_nodes:
                # try fail
                return False
        # try success
        self.issue_node(node)
        return True

    def resolve_queue(self):
        while len(self.pending_nodes) > 0:
            success = False
            for node in self.issuable_nodes:
                success = success or self.try_issue_node(node)
            if not success:
                # should be issuable nodes anyway, if not, the state do not change,
                # next iter is the same, and trap in infinite loop
                assert False
        assert len(self.finished_nodes) == len(self.nodes)
