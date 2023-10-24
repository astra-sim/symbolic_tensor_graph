import random
from .scheduler import *


class MonotonousNearstGreedyScheduler(Scheduler):
    def __init__(
        self, eg_nodes, node_runtime=None, queues_function=None, inplace=False
    ):
        raise NotImplementedError()
        super(MonotonousNearstGreedyScheduler, self).__init__(
            eg_nodes, node_runtime, queues_function, inplace
        )

    def _resolve_queue_impl(self):
        dep_free_nodes = list()
        for node in self.nodes:
            if len(self._pending_issue_parents[node.id]) == 0:
                dep_free_nodes.append(node)

        while not len(dep_free_nodes) == 0:
            freed_nodes_this_round = list()
            for node in dep_free_nodes:
                latest_parent_end_time = 0
                for parent in node.parent:
                    assert (
                        parent in self._node_id_map_end_time
                    )  # as it is dep free, all parent should finished and have end time
                    latest_parent_end_time = max(
                        latest_parent_end_time, self._node_id_map_end_time[parent]
                    )
                duration_time = self.get_node_runtime(node)
                begin_time = latest_parent_end_time
                # select a queue, here just find the one with shortest latest tick
                shortest_queue_tick = float("inf")
                picked_queue = None
                issuable = False
                for queue in self.queues:
                    if queue.issuable(node):
                        issuable = True
                        break
                assert issuable

                for queue in self.queues:
                    if queue.latest_task_tick < begin_time:
                        continue
                    if queue.latest_task_tick < shortest_queue_tick:
                        shortest_queue_tick = queue.latest_task_tick
                        picked_queue = queue
                if picked_queue is None:
                    # all queue has latest task tick before the begin time
                    longest_queue_tick = -float("inf")
                    for queue in self.queues:
                        if not queue.issuable(node):
                            continue
                        if queue.latest_task_tick > longest_queue_tick:
                            longest_queue_tick = queue.latest_task_tick
                            picked_queue = queue

                begin_time, end_time = picked_queue.insert_task(
                    node, begin_time, duration_time
                )
                print(
                    f"insert {node.name} in queue {id(queue)} and with {begin_time}:{end_time}"
                )
                self._node_id_map_end_time[node.id] = end_time
                for child in self.parent_map_child[node.id]:
                    assert node.id in self._pending_issue_parents[child]
                    self._pending_issue_parents[child].remove(node.id)
                    if len(self._pending_issue_parents[child]) == 0:
                        freed_nodes_this_round.append(self.node_id_map_node[child])
            dep_free_nodes = freed_nodes_this_round
