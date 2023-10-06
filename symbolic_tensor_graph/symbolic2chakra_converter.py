import sys, os

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "../chakra/et_def/"))
sys.path.append(os.path.join(file_dir, "../chakra/third_party/utils/"))

import sympy as sp
from .tensor import Tensor
from .offload_strategy import OffloadStrategy
from et_def_pb2 import *
from protolib import *


class Symbolic2ChakraConverter:
    def __init__(self, symbolic_file, eg_file, symbol_value_map, num_npu):
        self.tensors = Tensor.parse_records(symbolic_file)
        symbols = self.get_symbols()
        for symbol in symbols:
            assert str(symbol) in symbol_value_map.keys()
        self.symbol_value_map = symbol_value_map
        self.tensor_node_maps = dict()
        self._next_node_id = 0
        self.eg_file = eg_file
        self.num_npu = num_npu

    def get_symbols(self):
        symbols_list = list()
        for tensor in self.tensors:
            for dimension in tensor.shape:
                for symbol in dimension.free_symbols:
                    if not symbol in symbols_list:
                        symbols_list.append(symbol)
        return symbols_list

    @property
    def next_node_id(self):
        self._next_node_id += 1
        return self._next_node_id

    def parse_comm_attr(self, comm_attr):
        terms = comm_attr.split("@")
        assert len(terms) == 2
        type_, size_ = terms
        if type_ == "Scatter":
            type_ = CollectiveCommType.REDUCE_SCATTER
        elif type_ == "AllGather":
            type_ = CollectiveCommType.ALL_GATHER
        elif type_ == "AllReduce":
            type_ = CollectiveCommType.ALL_REDUCE
        else:
            assert False
            # type_ = CollectiveCommType.INVALID_COMM
        # size_ = sp.parse_expr(size_)
        size_ = Tensor.parse_expr(size_)
        # print(size_)
        size_value = Tensor.eval_expr(size_, self.symbol_value_map)
        # size_value = size_.evalf(subs=self.symbol_value_map)
        # print(size_, self.symbol_value_map, size_value)
        return type_, int(size_value)

    def get_comp_node(self, name, num_ops, tensor_size):
        node = Node()
        node.name = name
        node.id = self.next_node_id
        node.node_type = NodeType.COMP_NODE
        node.num_ops = int(num_ops)
        node.tensor_size = int(tensor_size)
        return node

    def get_coll_comm_node(
        self, name, comm_type, comm_size, involved_dims=[True, True]
    ):
        node = Node()
        node.name = name
        node.id = self.next_node_id
        node.node_type = NodeType.COMM_COLL_NODE
        node.comm_type = comm_type
        node.comm_size = int(comm_size)
        for involved_dim in involved_dims:
            node.involved_dim.append(involved_dim)
        return node

    def get_memory_load_node(self, name, tensor_size):
        node = Node()
        node.name = name
        node.id = self.next_node_id
        node.node_type = NodeType.MEM_LOAD_NODE
        node.tensor_size = int(tensor_size)
        node.tensor_loc = MemoryType.REMOTE_MEMORY
        return node

    def convert_tensor_to_nodes(self, tensor):
        nodes = list()

        if tensor.op_type in {"M", "E", "A"}:
            num_ops = int(Tensor.eval_expr(tensor.ops, self.symbol_value_map))
            # num_ops = int(tensor.ops.evalf(subs=self.symbol_value_map))
            tensor_size = 1
            for dim in tensor.shape:
                tensor_size *= dim
            tensor_size = int(Tensor.eval_expr(tensor_size, self.symbol_value_map))
            # tensor_size = int(tensor_size.evalf(subs=self.symbol_value_map))
            node = self.get_comp_node(
                name=tensor.id, num_ops=num_ops, tensor_size=tensor_size
            )
            nodes.append(node)
        elif tensor.op_type in {"C"}:
            comm_type, comm_size = self.parse_comm_attr(tensor.op_attr)
            node = self.get_coll_comm_node(
                name=tensor.id, comm_type=comm_type, comm_size=comm_size
            )
            nodes.append(node)
        elif tensor.op_type in {"T"}:
            pass
        else:
            assert False

        if tensor.post_communications == "" or tensor.post_communications is None:
            pass
        else:
            comm_type, comm_size = self.parse_comm_attr(tensor.post_communications)
            node = self.get_coll_comm_node(
                name=tensor.id + "_post_comm", comm_type=comm_type, comm_size=comm_size
            )
            nodes.append(node)

        for from_ in range(len(nodes) - 1):
            to_ = from_ + 1
            nodes[to_].parent.append(nodes[from_].id)
        return nodes

    def get_tensor_node_maps(self):
        for tensor in self.tensors:
            id_ = tensor.id
            nodes = self.convert_tensor_to_nodes(tensor)
            self.tensor_node_maps[id_] = nodes

    def connect_nodes(self):
        for tensor in self.tensors:
            assert tensor.id in self.tensor_node_maps
            nodes = self.tensor_node_maps[tensor.id]
            if len(nodes) == 0:
                # tensor type
                continue
            node_head = nodes[0]
            if tensor.x1 is not None:
                x1_nodes = self.tensor_node_maps[tensor.x1]
                if len(x1_nodes) == 0:
                    # tensor type
                    pass
                else:
                    x1_tail = x1_nodes[-1]
                    node_head.parent.append(x1_tail.id)
            if tensor.x2 is not None:
                x2_nodes = self.tensor_node_maps[tensor.x2]
                if len(x2_nodes) == 0:
                    # tensor type
                    pass
                else:
                    x2_tail = x2_nodes[-1]
                    node_head.parent.append(x2_tail.id)
        return

    def readout(self):
        def _readout_worker(packed_args):
            output_filename_, nodes_ = packed_args
            with open(output_filename_, "wb") as g:
                for node in nodes_:
                    encodeMessage(g, node)
            return True

        nodes = list()
        for tensor in self.tensors:
            for node in self.tensor_node_maps[tensor.id]:
                nodes.append(node)

        input_args = list()
        for npu_id in range(self.num_npu):
            output_filename = f"{self.eg_file}.{npu_id}.eg"
            input_args.append((output_filename, nodes))

        rets = map(_readout_worker, input_args)

        # do not remove: it is lazy map and need to iter all to ensure execution
        for ret in rets:
            assert ret

        return

    def convert(self):
        self.get_tensor_node_maps()
        self.connect_nodes()

    def convert_and_readout(self):
        self.convert()
        self.readout()

    def get_nodes(self):
        nodes = list()
        for tensor in self.tensor_node_maps:
            for node in self.tensor_node_maps[tensor]:
                nodes.append(node)
        return nodes

    def replace_nodes(self, new_nodes, strict=True):
        id_new_nodes_map = dict()
        for new_node in new_nodes:
            id_new_nodes_map[new_node.id] = new_node
        for tensor in self.tensor_node_maps:
            for i, node in enumerate(self.tensor_node_maps[tensor]):
                if not node.id in id_new_nodes_map:
                    if strict:
                        assert False
                    else:
                        continue
                new_node = id_new_nodes_map[node.id]
                self.tensor_node_maps[tensor][i] = new_node
                if strict:
                    del id_new_nodes_map[node.id]
        if strict:
            # verify if the number of nodes and num of nodes in dict is exactly the same
            assert len(id_new_nodes_map) == 0


if __name__ == "__main__":
    sys.path.append("../")
    from models.transformer import transformer

    symbol_value_map = {
        "bp": 1024,
        "mp": 1,
        "B": 32 * 1024,
        "Seq": 1024,
        "H": 256,
        "D": 100,
        "DF": 400,
        "DI": 200,
        "DO": 100,
    }
    transformer(4, "../sharding_spreadsheets/dp")
    converter = Symbolic2ChakraConverter(
        "../sharding_spreadsheets/dp/processed_graphs/transformer_4.csv",
        "../sharding_spreadsheets/dp/egs/transformer_4",
        symbol_value_map,
        1024,
    )
