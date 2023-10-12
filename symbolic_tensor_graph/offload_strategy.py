import pandas as pd
import numpy as np


class OffloadStrategy:
    def __init__(self, create_empty=False):
        raise PendingDeprecationWarning()
        if not create_empty:
            assert False
        self.x1_offload = dict()
        self.x2_offload = dict()
        self._y_offload = dict()
        self.tensors = list()
        self._id_tensor_map = None

    def get_input_offload(self, tensor):
        if not tensor.id in self.x1_offload:
            assert False
        else:
            x1_offload = self.x1_offload[tensor.id]
        if not tensor.id in self.x2_offload:
            assert False
        else:
            x2_offload = self.x2_offload[tensor.id]
        return x1_offload, x2_offload

    def set_input_offload(self, tensor, x1_offload=1.0, x2_offload=1.0):
        assert tensor in self.tensors
        if (tensor.x1 is None) or (tensor.x1 == ""):
            self.x1_offload[tensor.id] = 0
        else:
            self.x1_offload[tensor.id] = x1_offload

        if (tensor.x2 is None) or (tensor.x2 == ""):
            self.x2_offload[tensor.id] = 0
        else:
            self.x2_offload[tensor.id] = x2_offload

    def update_output_offload(self):
        self._y_offload.clear()

        tensor_x1_child_map = dict()
        tensor_x2_child_map = dict()
        for tensor in self.tensors:
            if not tensor.x1 in tensor_x1_child_map:
                tensor_x1_child_map[tensor.x1] = list()
            tensor_x1_child_map[tensor.x1].append(tensor.id)
            if not tensor.x2 in tensor_x2_child_map:
                tensor_x2_child_map[tensor.x2] = list()
            tensor_x2_child_map[tensor.x2].append(tensor.id)

        for tensor in tensor_x1_child_map.keys():
            self._y_offload[tensor.id] = 0
            for child in tensor_x1_child_map[tensor]:
                x1_offload, _ = self.get_input_offload(child)
                self._y_offload[tensor.id] = max(self._y_offload[tensor.id], x1_offload)

        for tensor in tensor_x2_child_map.keys():
            assert tensor.id in self._y_offload
            for child in tensor_x2_child_map[tensor]:
                _, x2_offload = self.get_input_offload(child)
                self._y_offload[tensor.id] = max(self._y_offload[tensor.id], x2_offload)
        return

    def get_output_offload(self, tensor):
        assert tensor.id in self._y_offload
        return self._y_offload[tensor.id]

    # def set_output_offload()  # output offload statues should be inferred from input, and cannot be "set"

    def get_id_tensor_map(self):
        self._id_tensor_map = dict()
        for tensor in self.tensors:
            self._id_tensor_map[tensor.id] = tensor

    @property
    def id_tensor_map(self):
        def _get_id_tensor_map(tensors):
            _id_tensor_map = dict()
            for tensor in tensors:
                _id_tensor_map[tensor.id] = tensor
            return _id_tensor_map

        if self._id_tensor_map is None:
            self._id_tensor_map = _get_id_tensor_map(self.tensors)

        if len(self._id_tensor_map) != self.tensors:
            self._id_tensor_map = _get_id_tensor_map(self.tensors)

        return self._id_tensor_map

    def set_all_weight_offload(self, offload=1.0):
        for tensor in self.tensors:
            x1_offload, x2_offload = self.get_input_offload(tensor)
            if tensor.x1 is not None:
                if self.id_tensor_map[tensor.x1].require_grads:
                    x1_offload = offload
            if tensor.x2 is not None:
                if self.id_tensor_map[tensor.x2].require_grads:
                    x2_offload = offload
            self.set_input_offload(tensor, x1_offload, x2_offload)

    def set_all_intermediate_offload(self, offload=1.0):
        for tensor in self.tensors:
            x1_offload, x2_offload = self.get_input_offload(tensor)
            if tensor.x1 is not None:
                if not self.id_tensor_map[tensor.x1].require_grads:
                    x1_offload = offload
            if tensor.x2 is not None:
                if not self.id_tensor_map[tensor.x2].require_grads:
                    x2_offload = offload
            self.set_input_offload(tensor, x1_offload, x2_offload)

    def set_all_leaf_offload(self, offload=1.0):
        for tensor in self.tensors:
            parent1_offload, parent2_offload = self.get_input_offload(tensor)
            if tensor.x1 is not None:
                parent1_tensor = self.id_tensor_map[tensor.x1]
                grandparent11, grandparent12 = parent1_tensor.x1, parent1_tensor.x2
                if (grandparent11 is None) and (grandparent12 is None):
                    parent1_offload = offload

            if tensor.x2 is not None:
                parent2_tensor = self.id_tensor_map[tensor.x2]
                grandparent21, grandparent22 = parent2_tensor.x1, parent2_tensor.x2
                if (grandparent21 is None) and (grandparent22 is None):
                    parent2_offload = offload

            self.set_input_offload(tensor, parent1_offload, parent2_offload)

    @staticmethod
    def parse_records(csv_filename, tensors):
        offload_strategy = OffloadStrategy(True)
        df = pd.read_csv(csv_filename, encoding="utf-8", header=None)
        df = df.replace({np.nan: None})
        for i in range(df.shape[0]):
            data = np.array(df[i : i + 1]).reshape(-1)
            id_ = data[0]
            x1_offload = data[1]
            x2_offload = data[2]
            offload_strategy.x1_offload[id_] = x1_offload
            offload_strategy.x2_offload[id_] = x2_offload

        for tensor in tensors:
            assert tensor.id in offload_strategy.x1_offload
            assert tensor.id in offload_strategy.x2_offload
        offload_strategy.tensors = tensors
        return offload_strategy

    def to_records(self, csv_filename):
        data = list()
        for tensor_id, x1_offload in self.x1_offload.items():
            assert tensor_id in self.x2_offload
            x2_offload = self.x2_offload[tensor_id]
            data.append([tensor_id, x1_offload, x2_offload])
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, encoding="utf-8", header=None, index=None)
        return

    @staticmethod
    def create_blank(tensors):
        offload_strategy = OffloadStrategy(True)
        for tensor in tensors:
            offload_strategy.tensors.append(tensor)
            offload_strategy.set_input_offload(tensor, 0, 0)
        return offload_strategy
