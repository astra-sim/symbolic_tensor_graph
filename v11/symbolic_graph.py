import pandas as pd
import numpy as np
import graphviz

from tensor import Tensor, get_tensor_size


def _default_key_fn(tensor):
    return tensor.tensor_id


class SymbolicGraph:
    def __init__(self, csv_filename=None):
        self.keys = None
        self.tensors = None
        if csv_filename is not None:
            self.load(csv_filename)

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

    def visualize(self, filename, format="pdf"):
        f = graphviz.Digraph()
        for tensor in self.tensors:
            f.node(name=tensor.id_, lable=tensor.id_, id=tensor.id_, shape="box")
            if tensor.x1 is not None:
                f.edge(tensor.x1, tensor.id_)
            if tensor.x2 is not None:
                f.edge(tensor.x2, tensor.id_)
        f.render(filename, format=format, cleanup=True)

    def get_tensor_dict(self, key_fn=_default_key_fn):
        ret = dict()
        for tensor in self.tensors:
            ret[key_fn(tensor)] = tensor
        return ret

    def change_name(self, template="%s"):
        old_new_name_map = dict()
        for tensor in self.tensors:
            old_name = tensor.tensor_id
            new_name = template % (old_name,)
            tensor.tensor_id = new_name
            old_new_name_map[old_name] = new_name
        for tensor in self.tensors:
            if tensor.x1 in old_new_name_map:
                tensor.x1 = old_new_name_map[tensor.x1]
            if tensor.x2 in old_new_name_map:
                tensor.x2 = old_new_name_map[tensor.x2]
        return

    def add_prefix(self, prefix):
        self.change_name(template=f"{prefix}_%s")

    def add_postfix(self, postfix):
        self.change_name(template=f"{postfix}_%s")

    @staticmethod
    def link_graph(graphs, links, key_fn=_default_key_fn):
        merged_tensors = list()
        for graph in graphs:
            merged_tensors.extend(graph.tensors)
        tensor_dict = dict()
        for tensor in merged_tensors:
            key = key_fn(tensor)
            assert not key in tensor_dict
            tensor_dict[key] = tensor
        for from_, to_ in links.items():
            from_tensor = tensor_dict[from_]
            to_tensor = tensor_dict[to_]
            assert to_tensor.op_type == "T"
            assert get_tensor_size(from_tensor.shape) == get_tensor_size(
                to_tensor.shape
            )
            assert get_tensor_size(from_tensor.hidden) == get_tensor_size(
                to_tensor.hidden
            )

            for key in from_tensor.keys():
                to_tensor[key] = from_tensor[key]
