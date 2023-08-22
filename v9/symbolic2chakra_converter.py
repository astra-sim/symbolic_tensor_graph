import sys
sys.path.append("../../chakra/et_def/")
sys.path.append("../../chakra/third_party/utils/")

import sympy as sp
from tensor import Tensor
from et_def_pb2 import *
from protolib import *

class Symbolic2ChakraConverter:
    def __init__(self, symbolic_file, eg_file, num_npu):
        self.tensors = Tensor.parse_records(symbolic_file)
        self.symbol_value_map = dict()
        symbols = self.get_symbols()
        for symbol in symbols:
            self.symbol_value_map[symbol] = None
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
        if type_ == 'Scatter':
            type_ = CollectiveCommType.REDUCE_SCATTER
        elif type_ == 'AllGather':
            type_ = CollectiveCommType.All_GATHER
        elif type_ == 'AllReduce':
            type_ = CollectiveCommType.ALL_REDUCE
        else:
            type_ = CollectiveCommType.INVALID_COMM
        size_ = sp.parse_expr(size_)
        size_ = size_.evalf(subs=self.symbol_value_map)
        return type_, int(size_)
    
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
        elif tensor.op_type == 'C':
            # create comm node
            node = node()
            node.name = tensor.id
            node.id = self.next_node_id
            node.node_type = NodeType.COMM_COLL_NODE
            comm_type, comm_size = self.parse_comm_attr(tensor.op_attr)
            node.comm_type = comm_type
            node.comm_size = comm_size
            # node.involved_dim = list()
            node.involved_dim.append(1)
            nodes.append(node)
        elif tensor.op_type == 'T':
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
            node.name = tensor.id + "post"
            node.id = self.next_node_id
            node.node_type = NodeType.COMM_COLL_NODE
            comm_type, comm_size = self.parse_comm_attr(tensor.post_communications)
            node.comm_type = comm_type
            node.comm_size = comm_size
            # node.involved_dim = list()
            node.involved_dim.append(1)
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
            if not tensor in self.tensor_node_maps:
                # tensor type
                continue
            node_head = self.tensor_node_maps[tensor.id][0]
            if tensor.x1 is not None:
                if tensor.x1 in self.tensor_node_maps:
                    x1_tail = self.tensor_node_maps[tensor.x1][-1]
                    node_head.parent.append(x1_tail)
                # else: tensor type
            if tensor.x2 is not None:
                if tensor.x2 in self.tensor_node_maps:
                    x2_tail = self.tensor_node_maps[tensor.x2][-1]
                    node_head.parent.append(x2_tail)
                # else tensor type
        return
    
    def readout(self):
        for npu_id in range(self.num_npu):
            output_filename = f"{self.eg_file}.{npu_id}.et"
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
        
    
if __name__ == '__main__':
    converter = Symbolic2ChakraConverter(
                    'sharding_spreadsheets/dp/processed_graphs/transformer_8.csv', 
                    'sharding_spreadsheets/dp/ets/transformer_8', 
                    256)
    symbol_value_map = {
        'bp': 2, 'B': 8, 'Seq': 1024, 'H': 32, 'D': 32, 'DF': 128, 'DI': 1024, 'DO': 32
    }
    converter.symbol_value_map = symbol_value_map
    symbols = converter.convert()
    hook = 0
    
