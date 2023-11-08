import copy
import json
import tempfile
import os
import sympy as sp
from ..tensor import Tensor
from ..ops import PlaceHolder


class TensorGraph:
    def __init__(self, tensors, in_tensors=None, out_tensors=None):
        if in_tensors is None:
            in_tensors = list()
        if out_tensors is None:
            out_tensors = list()
        self.tensors = tensors
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors

    def reverse_links(self, links):
        reversed_links = dict()
        for old_from in links.keys():
            for old_to in links[old_from]:
                new_from, new_to = old_to, old_from
                if not new_from in reversed_links:
                    reversed_links[new_from] = list()
                reversed_links[new_from].append(new_to)
        return reversed_links

    def get_tensor_child_to_parent_link(self, tensors=None):
        if tensors is None:
            tensors = self.tensors
        child_to_parent = dict()
        for tensor in tensors:
            child_to_parent[tensor.id] = list()
            if tensor.x1 is not None:
                child_to_parent[tensor.id].append(tensor.x1.id)
            if tensor.x2 is not None:
                child_to_parent[tensor.id].append(tensor.x2.id)
        return child_to_parent

    def get_tensor_parent_to_child_link(self, tensors=None):
        child_to_parent = self.get_tensor_child_to_parent_link(tensors)
        parent_to_child = self.reverse_links(child_to_parent)
        return parent_to_child

    def get_tensor_id_map_tensor(self, tensors=None):
        if tensors is None:
            tensors = self.tensors
        tensor_id_map_tensor = dict()
        for tensor in tensors:
            tensor_id_map_tensor[tensor.id] = tensor
        return tensor_id_map_tensor

    def get_dimensions(self, tensors=None):
        if tensors is None:
            tensors = self.tensors
        dims = set()
        for tensor in tensors:
            if tensor.x1_shape is not None:
                for dim in tensor.x1_shape:
                    dims.add(dim)
            if tensor.x1_hidden is not None:
                for dim in tensor.x1_hidden:
                    dims.add(dim)
            if tensor.x2_shape is not None:
                for dim in tensor.x2_shape:
                    dims.add(dim)
            if tensor.x2_hidden is not None:
                for dim in tensor.x2_hidden:
                    dims.add(dim)
        return dims

    def get_symbols(self, tensors=None):
        dims = self.get_dimensions(tensors=tensors)
        symbols = set()
        for dim in dims:
            if isinstance(dim, int) or isinstance(dim, float):
                continue
            for sym in dim.free_symbols:
                symbols.add(sym)
        return symbols

    @classmethod
    def load_tensor_graph(cls, csv_filename, json_filename=None):
        tensors = Tensor.parse_records(csv_filename)
        graph = cls(tensors)
        if json_filename is None:
            assert csv_filename.endswith("csv")
            json_filename = csv_filename[:-3] + "json"
            # return graph

        tensor_id_map_tensor = graph.get_tensor_id_map_tensor()
        f = open(json_filename, "r")
        meta_data = json.load(f)
        f.close()
        for tensor_id in meta_data["in_tensors"]:
            if not "@" in tensor_id:
                tensor_id += "@0"
            assert tensor_id in tensor_id_map_tensor
            graph.in_tensors.append(tensor_id_map_tensor[tensor_id])
        for tensor_id in meta_data["out_tensors"]:
            if not "@" in tensor_id:
                tensor_id += "@0"
            assert tensor_id in tensor_id_map_tensor
            graph.out_tensors.append(tensor_id_map_tensor[tensor_id])
        symbols = set()
        for symbol in meta_data["symbols"]:
            symbols.add(sp.parse_expr(symbol))
        assert symbols == graph.get_symbols()
        return graph

    def save_tensor_graph(self, csv_filename, json_filename=None):
        Tensor.to_records(self.tensors, csv_filename)
        if json_filename is None:
            assert csv_filename.endswith("csv")
            json_filename = csv_filename[:-3] + "json"
            # return

        meta_data = dict()
        meta_data["symbols"] = list()
        for symbol in self.get_symbols():
            meta_data["symbols"].append(str(symbol))
        meta_data["in_tensors"] = list()
        meta_data["out_tensors"] = list()
        for tensor in self.in_tensors:
            meta_data["in_tensors"].append(tensor.id)
        for tensor in self.out_tensors:
            meta_data["out_tensors"].append(tensor.id)
        f = open(json_filename, "w")
        json.dump(meta_data, f)
        f.close()

    def __eq__(one, another):
        for one_tensor, another_tensor in zip(one.tensors, another.tensors):
            if not one_tensor._to_record() == another_tensor._to_record():
                return False
        for one_tensor, another_tensor in zip(one.in_tensors, another.in_tensors):
            if not one_tensor._to_record() == another_tensor._to_record():
                return False
        for one_tensor, another_tensor in zip(one.out_tensors, another.out_tensors):
            if not one_tensor._to_record() == another_tensor._to_record():
                return False
        return True

    def sanity_check(self):
        for tensor in self.in_tensors:
            assert tensor in self.tensors
            assert tensor.op_type == PlaceHolder.type_name
        for tensor in self.out_tensors:
            assert tensor in self.tensors

    def visualize(self, filename, format="pdf", tensors=None):
        if tensors is None:
            tensors = self.tensors
        Tensor.visualize(tensors, filename, format)

    def __deepcopy__(self, memo):
        copied_graph = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_file_path = os.path.join(tmp_dir, "graph.csv")
            json_file_path = os.path.join(tmp_dir, "grpah.json")
            self.save_tensor_graph(csv_file_path, json_file_path)
            copied_graph = self.__class__.load_tensor_graph(
                csv_file_path, json_file_path
            )
        return copied_graph


class HybridGraph(TensorGraph):
    class NodeType:
        COMP = "comp"
        X1_COMM = "x1_comm"
        X2_COMM = "x2_comm"

    def __init__(self, tensors, tensor_map_nodes=None, symbol_map_values=None):
        super(HybridGraph, self).__init__(tensors)
        if tensor_map_nodes is None:
            assert symbol_map_values is None
            tensor_map_nodes = dict()
            symbol_map_values = dict()
            for tensor in self.tensors:
                tensor_map_nodes[tensor] = dict()
            for symbol in self.get_symbols():
                symbol_map_values[symbol] = None

        assert symbol_map_values is not None
        self.tensor_map_nodes = tensor_map_nodes
        self.symbol_map_values = symbol_map_values
        for tensor in tensors:
            assert tensor in self.tensor_map_nodes
        for symbol in self.get_symbols():
            assert symbol in symbol_map_values.keys()

    def get_nodes(self):
        nodes = list()
        for tensor in self.tensor_map_nodes.keys():
            nodes_this_tensor = self.tensor_map_nodes[tensor]
            for node in nodes_this_tensor.values():
                nodes.append(node)
        return nodes

    def replace_nodes(self, nodes, strict=True):
        unprocessed_nodes = copy.deepcopy(nodes)
        node_id_map_node = self.get_node_id_map_node(unprocessed_nodes)
        for tensor in self.tensor_map_node.keys():
            nodes_this_tensor = self.tensor_map_nodes[tensor]
            for key in nodes_this_tensor.keys():
                old_value = nodes_this_tensor[key]
                if not old_value.id in node_id_map_node:
                    if strict:
                        assert False
                else:
                    new_value = node_id_map_node[old_value.id]
                    nodes_this_tensor[key] = new_value
                    unprocessed_nodes.remove(new_value)
        if strict:
            assert len(unprocessed_nodes) == 0

    def get_node_id_map_node(self, nodes=None):
        if nodes is None:
            nodes = self.get_nodes()
        ret = dict()
        for node in nodes:
            assert not node.id in ret
            ret[node.id] = node
        return ret

    def get_node_child_to_parent_link(self, nodes=None):
        if nodes is None:
            nodes = self.get_nodes()
        child_to_parent = dict()
        for node in nodes:
            child_to_parent[node.id] = list()
            for parent in node.parent:
                child_to_parent[node.id].append(parent)
        return child_to_parent

    def apply_node_child_to_parent_link(self, links, nodes=None):
        if nodes is None:
            nodes = self.get_nodes()
        for child_node in nodes:
            child_id = child_node.id
            while len(child_node.parent) > 0:
                child_node.parent.pop()
            for parent_id in links[child_id]:
                child_node.append(parent_id)

    def get_node_parent_to_child_link(self, nodes=None):
        child_to_parent = self.get_node_child_to_parent_link(nodes)
        parent_to_child = self.reverse_links(child_to_parent)
        return parent_to_child

    def apply_node_parent_to_child_link(self, parent_to_child, nodes=None):
        child_to_parent = self.reverse_links(parent_to_child)
        self.apply_node_child_to_parent_link(child_to_parent, nodes)

    def get_node_id_map_tensor(self, tensor_map_nodes=None):
        if tensor_map_nodes is None:
            tensor_map_nodes = self.tensor_map_nodes
        node_id_map_tensor = dict()
        for tensor in tensor_map_nodes.keys():
            for node in tensor_map_nodes[tensor].values():
                assert node.id not in node_id_map_tensor
                node_id_map_tensor[node.id] = tensor
        return node_id_map_tensor


class BundledHybridGraph:
    def __init__(self, graphs, symbol_map_value=None, parallel_dims=None):
        self.graphs = graphs
        if parallel_dims is None:
            assert symbol_map_value is None
            self.parallel_dims = list()
            self.symbol_map_value = dict()
            for symbol in graph[0].get_symbols():
                self.symbol_map_value[symbol] = None
        assert symbol_map_value is not None

        self.parallel_dims = list(parallel_dims)
        for graph in graphs:
            for symbol in graph.symbol_map_value.key():
                assert symbol in self.symbol_map_value
                assert self.symbol_map_value[symbol] == graph.symbol_map_value[symbol]
        for dim in self.parallel_dims:
            assert dim in self.symbol_map_value
