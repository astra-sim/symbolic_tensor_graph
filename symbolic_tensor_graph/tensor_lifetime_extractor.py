from collections import OrderedDict
import json


class TensorLifetimeExtractor:
    def __init__(self, tensors, tensor_name_map_node, node_in_queue):
        self.tensors = tensors
        self.tensor_name_map_node = tensor_name_map_node

        self.tensor_name_map_tensor = dict()
        self.node_id_map_tensor_name = dict()
        self.node_in_queue = node_in_queue
        self.memory_operations = OrderedDict()

        self._alloced_memory = set()
        self._dealloced_memory = set()
        self._tensor_name_map_tid = None

        self.reset()

    def analysis_memory_alloc(self):
        for node in self.node_in_queue:
            assert node.id in self.node_id_map_tensor_name
            y_tensor_name = self.node_id_map_tensor_name[node.id]
            y_tensor = self.tensor_name_map_tensor[y_tensor_name]
            if node.id != self.tensor_name_map_node[y_tensor_name][0].id:
                # when counting allocs, only capture the first node of this tensor
                continue

            x1_tensor_name = y_tensor.x1
            if x1_tensor_name is not None:
                if not x1_tensor_name in self._alloced_memory:
                    # tensor used before generated from calc, should be leaf tensor which will load from remote mem
                    x1_tensor = self.tensor_name_map_tensor[x1_tensor_name]
                    assert (x1_tensor.x1 == None) and (x1_tensor.x2 == None)
                    self.memory_operations[node.id].append(("load", x1_tensor_name))
                    print(f"alloc_add x1 {x1_tensor_name} during node {node.id}")
                    self._alloced_memory.add(x1_tensor_name)
                # else: from calc

            x2_tensor_name = y_tensor.x2
            if x2_tensor_name is not None:
                if not x2_tensor_name in self._alloced_memory:
                    # tensor used before generated from calc, should be leaf tensor which will load from remote mem
                    x2_tensor = self.tensor_name_map_tensor[x2_tensor_name]
                    assert (x2_tensor.x1 == None) and (x2_tensor.x2 == None)
                    self.memory_operations[node.id].append(("load", x2_tensor_name))
                    print(f"alloc_add x2 {x2_tensor_name} during node {node.id}")
                    self._alloced_memory.add(x2_tensor_name)
                # else: from calc

            self.memory_operations[node.id].append(("alloc", y_tensor_name))
            print(f"alloc_add y {y_tensor_name} during node {node.id}")
            assert not y_tensor_name in self._alloced_memory
            self._alloced_memory.add(y_tensor_name)

    def analysis_memory_dealloc(self):
        for node in reversed(self.node_in_queue):
            y_tensor_name = self.node_id_map_tensor_name[node.id]
            y_tensor = self.tensor_name_map_tensor[y_tensor_name]
            if node.id != self.tensor_name_map_node[y_tensor_name][-1].id:
                # when counting allocs, only capture the first node of this tensor
                continue

            x1_tensor_name = y_tensor.x1
            if x1_tensor_name is not None:
                if not x1_tensor_name in self._dealloced_memory:
                    # last time used (first time seen in reverse)
                    self.memory_operations[node.id].append(("dealloc", x1_tensor_name))
                    self._dealloced_memory.add(x1_tensor_name)
            x2_tensor_name = y_tensor.x2
            if x2_tensor_name is not None:
                if not x2_tensor_name in self._dealloced_memory:
                    # last time user (first time seen in reverse)
                    self.memory_operations[node.id].append(("dealloc", x2_tensor_name))
                    self._dealloced_memory.add(x2_tensor_name)

            if not y_tensor_name in self._dealloced_memory:
                # from generation, the tensor is never used, should of stored
                # (assuming it is meaningful, if not used as intermediate then it must be result)
                self.memory_operations[node.id].append(("store", y_tensor_name))
                self._dealloced_memory.add(y_tensor_name)

    def analysis_memory_access(self):
        for node in self.node_in_queue:
            assert node.id in self.node_id_map_tensor_name
            y_tensor_name = self.node_id_map_tensor_name[node.id]
            y_tensor = self.tensor_name_map_tensor[y_tensor_name]

            self.memory_operations[node.id].append(("write", y_tensor_name))

            x1_tensor_name = y_tensor.x1
            if x1_tensor_name is not None:
                self.memory_operations[node.id].append(("read", x1_tensor_name))
            x2_tensor_name = y_tensor.x2
            if x2_tensor_name is not None:
                self.memory_operations[node.id].append(("read", x2_tensor_name))

    def analysis_memory_operations(self):
        self._alloced_memory.clear()
        self._dealloced_memory.clear()
        for node in self.node_in_queue:
            self.memory_operations[node.id].clear()
        self.analysis_memory_alloc()
        self.analysis_memory_dealloc()
        assert len(self._alloced_memory) == len(self._dealloced_memory)
        self.analysis_memory_access()

    def reset(self):
        self.tensor_name_map_tensor = dict()
        for tensor in self.tensors:
            self.tensor_name_map_tensor[tensor.id] = tensor

        self.node_id_map_tensor_name = dict()
        for tensor in self.tensor_name_map_node.keys():
            for node in self.tensor_name_map_node[tensor]:
                self.node_id_map_tensor_name[node.id] = tensor
        assert len(self.node_in_queue) == len(self.node_id_map_tensor_name)

        self._alloced_memory = set()
        self._dealloced_memory = set()
        self.memory_operations = OrderedDict()
        for node in self.node_in_queue:
            self.memory_operations[node.id] = list()
        self._tensor_name_map_tid = None

    @property
    def tensor_name_map_tid(self):
        if self._tensor_name_map_tid is not None:
            if len(self._tensor_name_map_tid) != len(self.tensors):
                self._tensor_name_map_tid = None
        if self._tensor_name_map_tid is None:
            self._tensor_name_map_tid = dict()
            # build tid based on their exist time
            tensor_alloc_time = dict()
            tensor_dealloc_time = dict()
            for i, node_id in enumerate(self.memory_operations.keys()):
                for op_type, tensor_name in self.memory_operations[node_id]:
                    if op_type == "load" or op_type == "alloc":
                        assert not tensor_name in tensor_alloc_time
                        tensor_alloc_time[tensor_name] = i
                    elif op_type == "store" or op_type == "dealloc":
                        assert not tensor_name in tensor_dealloc_time
                        tensor_dealloc_time[tensor_name] = i
                    elif op_type == "read" or op_type == "write":
                        pass
                    else:
                        # invalid op type
                        assert False
            assert len(tensor_alloc_time) == len(tensor_dealloc_time)
            # assert len(tensor_alloc_time) == len(self.tensors)
            duration_appear_count = (
                dict()
            )  # used to avoid two tensor have same duration
            for tensor_name in tensor_alloc_time.keys():
                alloc_time = tensor_alloc_time[tensor_name]
                dealloc_time = tensor_dealloc_time[tensor_name]
                duration = dealloc_time - alloc_time
                if not duration in duration_appear_count:
                    duration_appear_count[duration] = 0
                this_duraction_appear_count = duration_appear_count[duration]
                duration_appear_count[duration] += 1
                self._tensor_name_map_tid[tensor_name] = -(
                    duration * (1 << 16) + this_duraction_appear_count
                )
        return self._tensor_name_map_tid

    def to_records(self, filename):
        node_id_in_queue = list()
        for node in self.node_in_queue:
            node_id_in_queue.append(node.id)
        dump_dict = {
            "node_in_queue": node_id_in_queue,
            "memory_operations": self.memory_operations,
        }
        f = open(filename, "w")
        json.dump(dump_dict, f, indent=4)
        f.close()

    def parse_records(self, filename):
        self.reset()
        f = open(filename, "r")
        load_dict = json.load(f)
        f.close()
        assert len(load_dict["node_in_queue"]) == len(self.node_in_queue)
        for load_node_id, class_node in zip(
            load_dict["node_in_queue"], self.node_in_queue
        ):
            assert load_node_id == class_node.id
        load_memory_operations = load_dict["memory_operations"]
        for node_id in self.memory_operations.keys():
            load_memory_operations_this_node = load_memory_operations[str(node_id)]
            for i, operation in enumerate(load_memory_operations_this_node):
                load_memory_operations_this_node[i] = tuple(operation)
            self.memory_operations[node_id] = load_memory_operations_this_node

    def visualize(self, chrome_trace_file):
        event_interval = 10
        trace_events = list()

        tick = event_interval
        for node in self.node_in_queue:
            tensor_name = self.node_id_map_tensor_name[node.id]

            event = {
                "name": tensor_name,
                "cat": "op",
                "ph": "B",
                "ts": tick,
                "pid": 1,
                "tid": 1,
            }
            trace_events.append(event)

            event = {
                "name": tensor_name,
                "cat": "op",
                "ph": "E",
                "ts": tick + event_interval,
                "pid": 1,
                "tid": 1,
            }
            trace_events.append(event)

            tick += event_interval

        tick = event_interval
        for node in self.node_in_queue:
            node_id = node.id
            memory_operations = self.memory_operations[node_id]
            for operation_type, tensor_name in memory_operations:
                tid = self.tensor_name_map_tid[tensor_name]
                # tid = hash(tensor_name)
                if operation_type == "load" or operation_type == "alloc":
                    event = {
                        "name": tensor_name,
                        "cat": "tensor",
                        "ph": "B",
                        "ts": tick,
                        "pid": 2,
                        "tid": tid,
                    }
                    trace_events.append(event)
                elif operation_type == "store" or operation_type == "dealloc":
                    event = {
                        "name": tensor_name,
                        "cat": "tensor",
                        "ph": "E",
                        "ts": tick + event_interval,
                        "pid": 2,
                        "tid": tid,
                    }
                    trace_events.append(event)
                elif operation_type == "read" or operation_type == "write":
                    event = {
                        "name": f"{tensor_name}_{operation_type}",
                        "cat": "tensor",
                        "ph": "B",
                        "ts": tick,
                        "pid": 2,
                        "tid": tid,
                    }
                    trace_events.append(event)
                    event = {
                        "name": f"{tensor_name}_{operation_type}",
                        "cat": "tensor",
                        "ph": "E",
                        "ts": tick + event_interval,
                        "pid": 2,
                        "tid": tid,
                    }
                    trace_events.append(event)
                else:
                    assert False
            tick += event_interval

        dump_dict = {"traceEvents": trace_events}
        f = open(chrome_trace_file, "w")
        json.dump(dump_dict, f)
        f.close()
