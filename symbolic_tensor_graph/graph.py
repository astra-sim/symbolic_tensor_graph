import copy


class Graph:
    def __init__(self, create_empty=False):
        if not create_empty:
            assert False
        self.tensors = list()
        self.tensor_map_node = dict()

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

    def get_child_to_parent_link(self, nodes=None):
        if nodes is None:
            nodes = self.get_nodes()
        child_to_parent = dict()
        for node in nodes:
            child_to_parent[node.id] = list()
            for parent in node.parent:
                child_to_parent[node.id].append(parent)
        return child_to_parent

    def apply_child_to_parent_link(self, links, nodes=None):
        if nodes is None:
            nodes = self.get_nodes()
        for child_node in nodes:
            child_id = child_node.id
            while len(child_node.parent) > 0:
                child_node.parent.pop()
            for parent_id in links[child_id]:
                child_node.append(parent_id)

    def reverse_links(links):
        reversed_links = dict()
        for old_from in links.keys():
            for old_to in links[old_from]:
                new_from, new_to = old_to, old_from
                if not new_from in reversed_links:
                    reversed_links[new_from] = list()
                reversed_links[new_from].append(new_to)
        return reversed_links

    def get_parent_to_child_link(self, nodes=None):
        child_to_parent = self.get_child_to_parent_link(nodes)
        parent_to_child = self.reverse_links(child_to_parent)
        return parent_to_child

    def apply_parent_to_child_link(self, parent_to_child, nodes=None):
        child_to_parent = self.reverse_links(parent_to_child)
        self.apply_child_to_parent_link(child_to_parent, nodes)
