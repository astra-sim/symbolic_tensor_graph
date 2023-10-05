from collections import OrderedDict
from enum import Enum
import json


class TensorLifetimeExtractor:
    def __init__(self, tensors, tensor_name_map_node, node_in_queue):
        self.tensors = tensors
        self.tensor_name_map_node = tensor_name_map_node

        self.node_id_map_tensor_name = dict()
        self.node_in_queue = node_in_queue
        self.memory_operations = OrderedDict()

        self._alloced_memory = set()
        self._dealloced_memory = set()

        self.reset()

    def analysis_memory_alloc(self):
        for node in self.node_in_queue:
            assert node.id in self.node_id_map_tensor_name
            y_tensor_name = self.node_id_map_tensor_name[node.id]
            y_tensor = self.tensor_name_to_tensor_map[y_tensor_name]

            x1_tensor_name = y_tensor.x1
            if x1_tensor_name is not None:
                if not x1_tensor_name in self._alloced_memory:
                    # tensor used before generated from calc, should be leaf tensor which will load from remote mem
                    x1_tensor = self.tensor_name_to_tensor_map[x1_tensor_name]
                    assert (x1_tensor.x1 == None) and (x1_tensor.x2 == None)
                    self.memory_operations[node.id].append(("load", x1_tensor_name))
                    self._alloced_memory.add(x1_tensor_name)
                # else: from calc

            x2_tensor_name = y_tensor.x2
            if x2_tensor_name is not None:
                if not x2_tensor_name in self._alloced_memory:
                    # tensor used before generated from calc, should be leaf tensor which will load from remote mem
                    x2_tensor = self.tensor_name_to_tensor_map[x2_tensor_name]
                    assert (x2_tensor.x1 == None) and (x2_tensor.x2 == None)
                    self.memory_operations[node.id].append(("load", x2_tensor_name))
                    self._alloced_memory.add(x2_tensor_name)
                # else: from calc

            self.memory_operations[node.id].append(("alloc", y_tensor_name))
            self._alloced_memory.add(y_tensor_name)

    def analysis_memory_dealloc(self):
        for node in reversed(self.node_in_queue):
            y_tensor_name = self.node_id_map_tensor_name[node.id]
            y_tensor = self.tensor_name_to_tensor_map[y_tensor_name]

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

    def analysis_memory(self):
        self._alloced_memory.clear()
        self._dealloced_memory.clear()
        for node in self.node_in_queue:
            self.memory_operations[node.id].clear()
        self.analysis_memory_alloc()
        self.analysis_memory_dealloc()
        assert len(self._alloced_memory) == len(self._dealloced_memory)

    def reset(self):
        self.tensor_name_to_tensor_map = dict()
        for tensor in self.tensors:
            self.tensor_name_to_tensor_map[tensor.id] = tensor

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
