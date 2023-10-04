import copy


class Scheduler:
    def __init__(self, eg_nodes, node_runtime=None, num_queue=1, inplace=False):
        if inplace:
            self.nodes = eg_nodes
        else:
            self.nodes = copy.deepcopy(eg_nodes)
        self.node_runtime = node_runtime
        self.num_queue = num_queue

        self.queues = list()
        self.queues_tick = list()
        for _ in range(self.num_queue):
            self.queues.append(list())
            self.queues_tick.append(0)

        self._id_node_map = None
        self._parent_child_map = None

    @property
    def id_node_map(self):
        if self._id_node_map is not None:
            if len(self._id_node_map) != len(self.nodes):
                self._id_node_map = None
        if self._id_node_map is None:
            self._id_node_map = dict()
            for node in self.nodes:
                self._id_node_map[node.id] = node
        return self._id_node_map

    @property
    def parent_child_map(self):
        if self._parent_child_map is not None:
            if len(self._parent_child_map) != len(self.nodes):
                self._parent_child_map = None
        if self._parent_child_map == None:
            self._parent_child_map = dict()
            for node in self.nodes:
                self._parent_child_map[node.id] = list()
            for node in self.nodes:
                for parent in node.parent:
                    self._parent_child_map[parent].append(node.id)
        return self._parent_child_map

    def resolve_queue(self):
        raise NotImplementedError()

    def apply(self):
        self.resolve_queue()
        for queue in self.queues:
            for i in range(len(queue) - 1):
                from_ = queue[i]
                to_ = queue[i + 1]
                if not from_.id in to_.parent:
                    to_.parent.append(from_.id)
        return self.nodes


class GreedyScheduler(Scheduler):
    def __init__(self, eg_nodes, node_runtime=None, num_queue=1, inplace=False):
        super(GreedyScheduler, self).__init__(
            eg_nodes, node_runtime, num_queue, inplace
        )
        self.issuable_nodes = list()
        self.finished_nodes = list()

        for node in self.nodes:
            self.issuable_nodes.append(node)

    def issue_node(self, node):
        if self.node_runtime == None:
            tick = 1
        else:
            assert node.id in self.node_runtime
            tick = self.node_runtime[node.id]

        min_tick = 1e90
        min_queue = None
        for i, (queue, tick) in enumerate(zip(self.queues, self.queues_tick)):
            if tick < min_tick:
                min_queue = queue

        min_queue.append(node)
        self.queues_tick[i] += tick

        self.issuable_nodes.remove(node)
        self.finished_nodes.append(node)

    def try_issue_node(self, node):
        assert node in self.issuable_nodes
        for parent_id in node.parent:
            parent_node = self.id_node_map[parent_id]
            if not parent_node in self.finished_nodes:
                # try fail
                return False
        # try success
        self.issue_node(node)
        return True

    def resolve_queue(self):
        while len(self.issuable_nodes) > 0:
            success = False
            issuable_nodes = copy.copy(self.issuable_nodes)
            for node in issuable_nodes:
                success = success or self.try_issue_node(node)
            if not success:
                # should be issuable nodes anyway, if not, trap in infinite loop
                assert False
        assert len(self.finished_nodes) == len(self.nodes)
