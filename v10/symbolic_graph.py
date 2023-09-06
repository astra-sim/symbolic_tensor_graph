import pandas as pd
import numpy as np
import graphviz

from tensor import Tensor


class SymbolicGraph:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.keys = None
        self.tensors = None

    def load(self):
        df = pd.read_csv(self.csv_filename, encoding="utf-8")
        df = df.replace({np.nan: None})
        keys = list(df.columns)
        tensors = list()
        for i in range(df.shape[0]):
            data = np.array(df[i : i + 1]).reshape(-1)
            tensors.append(Tensor.parse_record(data, keys=keys))
        self.tensors = tensors
        self.keys = keys

    def save(self):
        keys = self.keys
        data = list()
        for tensor in self.tensors:
            data.append(tensor.to_record(keys=keys))
        df = pd.DataFrame(data, columns=keys)
        df.to_csv(self.csv_filename, encoding="utf-8", index=None)

    def visualize(self, filename, format="pdf"):
        f = graphviz.Digraph()
        for tensor in self.tensors:
            f.node(name=tensor.id_, lable=tensor.id_, id=tensor.id_, shape="box")
            if tensor.x1 is not None:
                f.edge(tensor.x1, tensor.id_)
            if tensor.x2 is not None:
                f.edge(tensor.x2, tensor.id_)
        f.render(filename, format=format, cleanup=True)

    def get_tensor_dict(self):
        ret = dict()
        for tensor in self.tensors:
            ret[(tensor.id_, tensor.num_iter)] = tensor
        return ret
