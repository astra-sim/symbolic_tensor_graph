import os, copy, json


class AstrasimNodeRuntimeDatabase:
    def __init__(self, system, network, memory, astrasim_bin):
        self.runtime_dict = dict()
        self.system = self.load_json(system)
        self.network = self.load_json(network)
        self.memory = self.load_json(memory)
        self.astrasim_bin = self.hash_file(astrasim_bin)

    def hash_file(self, path):
        assert os.path.exists(path)
        pipe_out = os.popen(f"md5sum {os.path.abspath(path)}")
        hash = pipe_out.readlines()[0].split(" ")[0]
        pipe_out.close()
        return hash

    def load_json(self, path):
        assert os.path.exists(path)
        f = open(path, "r")
        json_dict = json.load(f)
        f.close()
        return json_dict

    def node_remove_extra_attr(self, node):
        # node contains some graph-specified attr, should remove these extra attrs
        node = copy.deepcopy(node)
        while len(node.parent) > 0:
            node.parent.pop()
        node.id = 0
        node.name = ""
        return node

    def stringfy_node(self, node):
        node = self.node_remove_extra_attr(node)
        return node.SerializeToString()

    def sanity_check(self, system=None, network=None, memory=None, astrasim_bin=None):
        if not system is None:
            assert self.load_json(system) == self.system
        if not network is None:
            assert self.load_json(network) == self.network
        if not memory is None:
            assert self.load_json(memory) == self.memory
        if not astrasim_bin is None:
            assert self.hash_file(astrasim_bin) == self.astrasim_bin

    def lookup(self, node, system=None, network=None, memory=None, astrasim_bin=None):
        self.sanity_check(system, network, memory, astrasim_bin)
        stringfied_clean_node = self.stringfy_node(node)
        if stringfied_clean_node in self.runtime_dict:
            return self.runtime_dict[stringfied_clean_node]
        return None

    def update(
        self, node, runtime, system=None, network=None, memory=None, astrasim_bin=None
    ):
        self.sanity_check(system, network, memory, astrasim_bin)
        stringfied_clean_node = self.stringfy_node(node)
        self.runtime_dict[stringfied_clean_node] = runtime
