import copy, os, sys
import bisect

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "../../chakra/et_def/"))
sys.path.append(os.path.join(file_dir, "../../chakra/third_party/utils/"))

from et_def_pb2 import *
from protolib import *


class Scheduler:
    class TaskQueue:
        def __init__(self, function=None):
            if function is None:
                function = (
                    NodeType.COMP_NODE,
                    NodeType.COMM_COLL_NODE,
                    NodeType.COMM_SEND_NODE,
                    NodeType.COMM_RECV_NODE,
                    NodeType.MEM_LOAD_NODE,
                    NodeType.MEM_STORE_NODE,
                )
            self.function = function
            self.tasks = list()  # for each entity (node_id, begin_tick, end_tick)

        def issuable(
            self,
            node,
            earlist_begin_tick=None,
            latest_finish_tick=None,
            duration_tick=None,
        ):
            if not node.node_type in self.function:
                return False
            if earlist_begin_tick is not None:
                assert latest_finish_tick is not None
                assert duration_tick is not None
                _, begin_tick = self.find_first_available_gap(
                    earlist_begin_tick, duration_tick
                )
                end_tick = begin_tick + duration_tick
                if end_tick > latest_finish_tick:
                    return False
            return True

        def insert_task(self, node, begin_tick, duration_tick):
            # do not allow task with duration of 0
            assert duration_tick > 0

            gap_index, begin_tick = self.find_first_available_gap(
                begin_tick, duration_tick
            )
            end_tick = begin_tick + duration_tick
            self.tasks.insert(gap_index, (node.id, begin_tick, end_tick))
            return begin_tick, end_tick

        def find_first_available_gap(self, begin_tick, duration_tick):
            # in tasks:
            # task1 | gap | task2
            # b1  e1        b2  e
            #               gap_index
            # the available gap should be b >= e1 and e <= b2
            gap_index = bisect.bisect(self.tasks, begin_tick, key=lambda t: t[2])
            while True:
                if gap_index > 0:
                    gap_begin = self.tasks[gap_index - 1][2]
                else:
                    gap_begin = 0

                if gap_index < len(self.tasks):
                    gap_end = self.tasks[gap_index][1]
                else:
                    gap_end = float("inf")

                if gap_end >= gap_begin + duration_tick:
                    # gap has enough space
                    break

                if gap_index >= len(self.tasks):
                    # find to end of queue, and the gap after the last task should be large enough so shouldnt be here
                    assert False
                gap_index += 1
            return gap_index, gap_begin

        def sanity_check(self):
            end_tick = -float("inf")
            for task in self.tasks:
                assert task[1] >= end_tick
                assert task[2] > task[1]
                end_tick = task[2]

        @property
        def latest_task_tick(self):
            if len(self.tasks) > 0:
                return self.tasks[-1][2]
            else:
                return 0

    def __init__(
        self, eg_nodes, node_runtime=None, queues_function=None, inplace=False
    ):
        if inplace:
            self.nodes = eg_nodes
        else:
            self.nodes = copy.deepcopy(eg_nodes)
        self.node_runtime = node_runtime

        self.queues = list()
        if queues_function is None:
            queues_function = [None]
        for queue_function in queues_function:
            self.queues.append(Scheduler.TaskQueue(queue_function))

        self._node_id_map_node = None
        self._parent_map_child = None

        self._pending_issue_parents = dict()
        self._node_id_map_end_time = dict()

    @property
    def node_id_map_node(self):
        if self._node_id_map_node is not None:
            if len(self._node_id_map_node) != len(self.nodes):
                self._node_id_map_node = None
        if self._node_id_map_node is None:
            self._node_id_map_node = dict()
            for node in self.nodes:
                self._node_id_map_node[node.id] = node
        return self._node_id_map_node

    @property
    def parent_map_child(self):
        if self._parent_map_child is not None:
            if len(self._parent_map_child) != len(self.nodes):
                self._parent_map_child = None
        if self._parent_map_child == None:
            self._parent_map_child = dict()
            for node in self.nodes:
                self._parent_map_child[node.id] = list()
            for node in self.nodes:
                for parent in node.parent:
                    self._parent_map_child[parent].append(node.id)
        return self._parent_map_child

    def get_node_runtime(self, node):
        if self.node_runtime is None:
            return 1
        return self.node_runtime[node.id]

    def _resolve_queue_impl(self):
        raise NotImplementedError()

    def resolve_queue(self):
        for queue in self.queues:
            queue.tasks.clear()
        self._pending_issue_parents.clear()
        self._node_id_map_end_time.clear()
        for node in self.nodes:
            self._pending_issue_parents[node.id] = list()
            for parent in node.parent:
                self._pending_issue_parents[node.id].append(parent)

        self._resolve_queue_impl()

        assert len(self._node_id_map_end_time) == len(self.nodes)

    def apply(self):
        self.resolve_queue()
        for queue in self.queues:
            for i in range(len(queue.tasks) - 1):
                from_id = queue.tasks[i][0]
                to_id = queue.tasks[i + 1][0]
                to_ = self.node_id_map_node[to_id]
                if not from_id in to_.parent:
                    to_.parent.append(from_id)
        return self.nodes
