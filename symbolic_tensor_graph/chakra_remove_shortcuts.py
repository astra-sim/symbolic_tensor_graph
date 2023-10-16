import copy, sys

sys.setrecursionlimit(1000000)


class ChakraShortcutRemover:
    def __init__(self, eg_nodes, inplace=False):
        self.nodes = eg_nodes
        self.inplace = inplace
        self.reset()

    def extract_graph_structure_from_eg(self):
        parent_to_child = dict()
        for node in self.nodes:
            # every nodes can be parent. For those do not have child, it is empty list.
            parent_to_child[node.id] = list()
        for node in self.nodes:
            for parent in node.parent:
                parent_to_child[parent].append(node.id)
        self.parent_to_child = parent_to_child

    def remove_shortcuts(self):
        simplified_parent_to_child = dict()
        for parent in self.parent_to_child.keys():
            simplified_parent_to_child[parent] = list()
            for child in self.parent_to_child[parent]:
                is_shortcut = False
                # for each pair of parent->child1
                # find if there is a longer path, parent->child2->xxx->child1
                for another_child in self.parent_to_child[parent]:
                    if child == another_child:
                        continue
                    if self.reachable(another_child, child, self.parent_to_child):
                        # if we can reach the child1 from another child2, then parent->child1 is a short path
                        is_shortcut = True
                        break
                if not is_shortcut:
                    simplified_parent_to_child[parent].append(child)
        self.simplified_parent_to_child = simplified_parent_to_child

    def reachable(self, from_, to_, graph, memorize=True, recursive=True):
        if recursive:
            return self.reachable_recursive(from_, to_, graph, memorize)
        else:
            return self.reachable_flatten(from_, to_, graph, memorize)

    def reachable_recursive(self, from_, to_, graph, memorize=True):
        if memorize:
            if (from_, to_) in self.reachable_memory:
                return self.reachable_memory[(from_, to_)]

        reachable_ = False
        if from_ == to_:
            reachable_ = True
        else:
            for child in graph[from_]:
                if self.reachable_recursive(child, to_, graph, memorize):
                    reachable_ = True
                    break

        if memorize:
            self.reachable_memory[(from_, to_)] = reachable_
        return reachable_

    def reachable_flatten(self, from_, to_, graph, memorize=True):
        assert False
        reachable_ = False
        stack = [from_]

        while stack:
            current = stack.pop()
            if memorize:
                if (current, to_) in self.reachable_memory:
                    return self.reachable_memory[(current, to_)]

            if current == to_:
                reachable_ = True
                if memorize:
                    self.reachable_memory[(current, to_)] = reachable_
                break
            else:
                for child in graph[current]:
                    stack.insert(0, child)

        if memorize:
            self.reachable_memory[(from_, to_)] = reachable_
        return reachable_

    def verify(self):
        for parent in self.parent_to_child:
            for child in self.parent_to_child[parent]:
                assert self.reachable(
                    parent, child, self.simplified_parent_to_child, memorize=False
                )

    def apply_removal(self):
        simplified_child_to_parent = dict()
        for parent in self.simplified_parent_to_child:
            # every parent is other's child (for those parent do not have its own parent, it is empty list)
            simplified_child_to_parent[parent] = list()
        for parent in self.simplified_parent_to_child:
            for child in self.simplified_parent_to_child[parent]:
                simplified_child_to_parent[child].append(parent)
        if not self.inplace:
            self.simplified_nodes = copy.deepcopy(self.nodes)
        else:
            self.simplified_nodes = self.nodes
        for node in self.simplified_nodes:
            assert node.id in simplified_child_to_parent
            old_parents = copy.deepcopy(node.parent)
            for parent in old_parents:
                node.parent.remove(parent)
            for parent in simplified_child_to_parent[node.id]:
                node.parent.append(parent)

    def apply(self):
        self.reset()
        self.extract_graph_structure_from_eg()
        self.remove_shortcuts()
        self.verify()
        self.apply_removal()
        return self.simplified_nodes

    def reset(self):
        self.simplified_nodes = None
        self.parent_to_child = None
        self.simplified_parent_to_child = None
        self.reachable_memory = dict()
