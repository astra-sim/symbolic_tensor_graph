import sys, os

sys.path.append("../../../../extern/graph_frontend/chakra/et_def/")
sys.path.append("../../../../extern/graph_frontend/chakra/third_party/utils/")

import sympy as sp
from tensor import Tensor
from et_def_pb2 import *
from protolib import *


class Symbolic2ChakraConverter:
    def __init__(self, symbolic_file, eg_file, num_npu, symbol_value_map):
        self.tensors = Tensor.parse_records(symbolic_file)
        self.eg_file = eg_file
        self.num_npu = num_npu
        self.symbol_value_map = symbol_value_map

        symbols = self.get_symbols()
        for symbol in symbols:
            assert symbol in symbol_value_map

        self._next_node_id = 0

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
        size_ = sp.parse_expr(size_)
        size_value = size_.evalf(subs=self.symbol_value_map)
        print(size_, self.symbol_value_map, size_value)
        return type_, int(size_value)

    def convert_tensor_to_node(self, tensor):
        nodes = list()
        if not tensor.ops == 0:
            # create comp node
            node = Node()
            node.name = tensor.id
            node.id = self.next_node_id
            node.node_type = NodeType.COMP_NODE
            node.num_ops = int(tensor.ops.evalf(subs=self.symbol_value_map))
            nodes.append(node)
        elif tensor.op_type == "C":
            # create comm node
            node = Node()
            node.name = tensor.id
            node.id = self.next_node_id
            node.node_type = NodeType.COMM_COLL_NODE
            comm_type, comm_size = self.parse_comm_attr(tensor.op_attr)
            node.comm_type = comm_type
            node.comm_size = comm_size
            # node.involved_dim = list()
            node.involved_dim.append(True)
            node.involved_dim.append(True)
            nodes.append(node)
        elif tensor.op_type == "T":
            # tensor node, do nothing
            pass
        else:
            assert False

        if tensor.post_communications == "" or tensor.post_communications is None:
            # no post comm
            pass
        else:
            # create comm node
            node = Node()
            node.name = tensor.id + "post_comm"
            node.id = self.next_node_id
            node.node_type = NodeType.COMM_COLL_NODE
            comm_type, comm_size = self.parse_comm_attr(tensor.post_communications)
            node.comm_type = comm_type
            node.comm_size = comm_size
            # node.involved_dim = list()
            node.involved_dim.append(True)
            node.involved_dim.append(True)
            if len(nodes) > 0:
                node.parent.append(nodes[-1].id)
            nodes.append(node)
        return nodes

    def get_tensor_node_maps(self):
        for tensor in self.tensors:
            id_ = tensor.id
            nodes = self.convert_tensor_to_node(tensor)
            if len(nodes) == 0:
                # tensor type skip
                continue
            self.tensor_node_maps[id_] = nodes

    def connect_nodes(self):
        for tensor in self.tensors:
            if not tensor.id in self.tensor_node_maps:
                # tensor type
                continue
            node_head = self.tensor_node_maps[tensor.id][0]
            if tensor.x1 is not None:
                if tensor.x1 in self.tensor_node_maps:
                    x1_tail = self.tensor_node_maps[tensor.x1][-1]
                    node_head.parent.append(x1_tail.id)
                # else: tensor type
            if tensor.x2 is not None:
                if tensor.x2 in self.tensor_node_maps:
                    x2_tail = self.tensor_node_maps[tensor.x2][-1]
                    node_head.parent.append(x2_tail.id)
                # else tensor type
            # self.tensor_node_maps[tensor.id][0] = node_head
        return

    def readout(self):
        for npu_id in range(self.num_npu):
            output_filename = f"{self.eg_file}.{npu_id}.eg"
            with open(output_filename, "wb") as g:
                for tensor_id in self.tensor_node_maps.keys():
                    nodes = self.tensor_node_maps[tensor_id]
                    for node in nodes:
                        encodeMessage(g, node)
        return

    def convert(self):
        self.get_tensor_node_maps()
        self.connect_nodes()
        self.readout()
