import copy
from ..tensor import Tensor


class TensorGraph:
    def __init__(self, tensors, in_nodes=None, out_nodes=None):
        if in_nodes is None:
            in_nodes = list()
        if out_nodes is None:
            out_nodes = list()
        self.tensors = tensors

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

    @classmethod
    def load_tensor_graph(cls, csv_filename):
        tensors = Tensor.parse_records(csv_filename)
        return cls(tensors)

    def save_tensor_graph(self, csv_filename):
        Tensor.to_records(self.tensors, csv_filename)

    def __eq__(one, another):
        if not one.tensors == another.tensors:
            return False
        return True


class HybridGraph(TensorGraph):
    def __init__(self, tensors, tensor_map_nodes=None):
        super(HybridGraph, self).__init__(tensors)
        if tensor_map_nodes is None:
            tensor_map_nodes = dict()
            for tensor in self.tensors:
                tensor_map_nodes[tensor] = list()
        self.tensor_map_nodes = tensor_map_nodes
        for tensor in tensors:
            assert tensor in self.tensor_map_nodes

    def get_nodes(self):
        nodes = list()
        for tensor in self.tensor_map_node.keys():
            nodes_this_tensor = self.tensor_map_node[tensor]
            for node in nodes_this_tensor.values():
                nodes.append(node)
        return nodes

    def replace_nodes(self, nodes, strict=True):
        unprocessed_nodes = copy.deepcopy(nodes)
        node_id_map_node = self.get_node_id_map_node(unprocessed_nodes)
        for tensor in self.tensor_map_node.keys():
            nodes_this_tensor = self.tensor_map_node[tensor]
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
            nodes = self.get_nodes
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
