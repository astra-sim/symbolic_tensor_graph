import multiprocessing
import copy, tempfile


class NodeRunner:
    def __init__(self, node_executor, database, num_workers=-1):
        self.node_executor = node_executor
        self.database = database
        if num_workers == -1:
            num_workers = int(multiprocessing.cpu_count() * 0.5)
        self.num_workers = num_workers
        self.pool = multiprocessing.Pool(self.num_workers)

    @staticmethod
    def _task(database, node_executor, nodes):
        nodes_runtime = list()
        new_records = dict()
        for node in nodes:
            node = database.node_remove_extra_attr(node)
            runtime = database.lookup(node)
            if runtime == None:
                node_executor.update_workload([node])
                runtime = node_executor.run()
                database.update(node, runtime)
                new_records[database.stringfy_node(node)] = runtime
            nodes_runtime.append(runtime)
        return nodes, nodes_runtime, new_records

    def run_nodes(self, nodes):
        rets = list()
        num_workers = min(len(nodes), self.num_workers)
        node_id_start, node_id_end = 0, 0
        nodes_per_worker = (len(nodes) + num_workers - 1) // num_workers
        for worker_id in range(num_workers):
            node_id_start = node_id_end
            node_id_end = node_id_start + nodes_per_worker
            node_id_end = min(node_id_end, len(nodes))
            nodes_this_worker = nodes[node_id_start:node_id_end]
            database = copy.deepcopy(self.database)
            executor = copy.deepcopy(self.node_executor)
            rets.append(
                self.pool.apply_async(
                    NodeRunner._task, (database, executor, nodes_this_worker)
                )
            )
        nodes_runtime = list()
        for _ in range(len(rets)):
            rets[_] = rets[_].get()
            nodes_worker, nodes_runtime_worker, new_records_worker = rets[_]
            nodes_runtime.extend(nodes_runtime_worker)
            for key in new_records_worker.keys():
                value = new_records_worker[key]
                self.database.runtime_dict[key] = value
        assert len(nodes_runtime) == len(nodes)
        return nodes_runtime

    def run_node(self, node):
        return self.run_nodes([node])
