import pandas as pd
import numpy as np

from tensor import Tensor


class SymbolicGraph:
    def __init__(self, csv_filename):
        self.keys = None
        self.tensors = None
        if not csv_filename is None:
            self.load(csv_filename)
        else:
            # create empty graph
            pass

    def load(self, csv_filename):
        df = pd.read_csv(csv_filename, encoding="utf-8")
        df = df.replace({np.nan: None})
        keys = list(df.columns)
        tensors = list()
        for i in range(df.shape[0]):
            data = np.array(df[i : i + 1]).reshape(-1)
            tensors.append(Tensor.parse_record(data, keys=keys))
        self.tensors = tensors
        self.keys = keys

    def save(self, csv_filename):
        keys = self.keys
        data = list()
        for tensor in self.tensors:
            data.append(tensor.to_record(keys=keys))
        df = pd.DataFrame(data, columns=keys)
        df.to_csv(csv_filename, encoding="utf-8", index=None)

    def check_signature(self, keys=None, strict=False):
        if keys is None:
            keys = [
                "tensor_id",
                "require_grads",
                "shape",
                "x1_id",
                "x2_id",
                "op_type",
                "op_attr",
            ]
        for key in keys:
            assert key in self.keys
        if strict:
            assert len(keys) == len(self.keys)
        # for tensor in self.tensors:
        #     tensor.check_signature(keys, strict)
