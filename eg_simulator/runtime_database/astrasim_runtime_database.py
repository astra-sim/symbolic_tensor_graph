import os, copy


class AstrasimNodeRuntimeDatabase:
    def __init__(self, system, network, memory, astrasim_bin):
        self.runtime_dict = dict()
        self.system = system
        self.network = network
        self.memory = memory

        assert os.path.exists(astrasim_bin)
        pipe_out = os.popen(f"md5sum {os.path.abspath(astrasim_bin)}")
        self.astrasim_bin_hash = pipe_out.readlines()[0].split(" ")[0]
        pipe_out.close()

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
        if not system == None:
            assert system == self.system
        if not network == None:
            assert network == self.network
        if not memory == None:
            assert memory == self.memory
        if not astrasim_bin == None:
            assert os.path.exists(astrasim_bin)
            pipe_out = os.popen(f"md5sum {os.path.abspath(astrasim_bin)}")
            astrasim_bin_hash = pipe_out[0].split(" ")[0]
            assert astrasim_bin_hash == self.astrasim_bin_hash

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
