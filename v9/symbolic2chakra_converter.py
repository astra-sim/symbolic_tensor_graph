import sys
sys.path.append("../../chakra/et_def/")
sys.path.append("../../chakra/third_party/utils/")

import sympy as sp
from tensor import Tensor
from offload_strategy import OffloadStrategy
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
       
        
class Symbolic2ChakraConverterWithOffload:
    def __init__(self, symbolic_file, offload_file, eg_file, num_npu):
        tensors = Tensor.parse_records(symbolic_file)
        self.tensors = dict()
        for tensor in tensors:
            self.tensors[tensor.id] = tensor
        self.symbol_value_map = dict()
        symbols = self.get_symbols()
        for symbol in symbols:
            self.symbol_value_map[symbol] = None
        self.tensor_node_maps = dict()
        self._next_node_id = 0
        self.eg_file = eg_file
        self.num_npu = num_npu
        self.offload_strategy = OffloadStrategy.parse_records(offload_file)
        self.offload_nodes = list()
    
    def get_symbols(self):
        symbols_list = list()
        for tensor in self.tensors.values():
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
            node = Node()
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
            
        if self.offload_strategy.get_offload(tensor) > 0:
            if tensor.op_type != 'T':
                node = Node()
                node.name = tensor.id + "post_save"
                node.id = self.next_node_id
                node.node_type = NodeType.MEM_STORE_NODE
                size = 1
                for s in tensor.shape:
                    size *= s
                size = int(size.evalf(subs=self.symbol_value_map) * self.offload_strategy.get_offload(tensor))
                node.tensor_size = size
                node.tensor_loc = MemoryType.REMOTE_MEMORY
                if len(nodes) > 0:
                    node.parent.append(nodes[-1].id)
                nodes.append(node)
                
            # node = Node()
            # node.name = tensor.id + "post_load"
            # node.id = self.next_node_id
            # node.node_type = NodeType.MEM_LOAD_NODE
            # size = 1
            # for s in tensor.shape:
            #     size *= s
            # size = int(size.evalf(subs=self.symbol_value_map) * self.offload_strategy.get_offload(tensor))
            # node.tensor_size = size
            # node.tensor_loc = MemoryType.REMOTE_MEMORY
            # if len(nodes) > 0:
            #     node.parent.append(nodes[-1].id)
            # nodes.append(node)
            
        return nodes
    
    def get_tensor_node_maps(self):
        for tensor in self.tensors.values():
            id_ = tensor.id
            nodes = self.convert_tensor_to_node(tensor)
            if len(nodes) == 0:
                # tensor type skip
                continue
            self.tensor_node_maps[id_] = nodes
    
    def connect_nodes(self):
        for tensor in self.tensors.values():
            if not tensor.id in self.tensor_node_maps:
                # tensor type
                continue
            node_head = self.tensor_node_maps[tensor.id][0]
            if tensor.x1 is not None:
                x1 = self.tensors[tensor.x1]
                if self.offload_strategy.get_offload(x1):
                    load_node = Node()
                    load_node.name = tensor.id + "_x1_pre_load"
                    load_node.id = self.next_node_id
                    load_node.node_type = NodeType.MEM_LOAD_NODE
                    size = 1
                    for s in x1.shape:
                        size *= s
                    size = int(size.evalf(subs=self.symbol_value_map) * self.offload_strategy.get_offload(x1))
                    load_node.tensor_size = size
                    load_node.tensor_loc = MemoryType.REMOTE_MEMORY
                    if tensor.x1 in self.tensor_node_maps:
                        assert self.tensor_node_maps[tensor.x1][-1].node_type == NodeType.MEM_STORE_NODE
                        load_node.parent.append(self.tensor_node_maps[tensor.x1][-1].id)
                    node_head.parent.append(load_node.id)
                    self.offload_nodes.append(load_node)
                else:
                    if tensor.x1 in self.tensor_node_maps:
                        x1_tail = self.tensor_node_maps[tensor.x1][-1]
                        node_head.parent.append(x1_tail.id)
                    # else: tensor type and no offload
            if tensor.x2 is not None:
                x2 = self.tensors[tensor.x2]
                if self.offload_strategy.get_offload(x2):
                    load_node = Node()
                    load_node.name = tensor.id + "_x2_pre_load"
                    load_node.id = self.next_node_id
                    load_node.node_type = NodeType.MEM_LOAD_NODE
                    size = 1
                    for s in x2.shape:
                        size *= s
                    size = int(size.evalf(subs=self.symbol_value_map) * self.offload_strategy.get_offload(x2))
                    load_node.tensor_size = size
                    load_node.tensor_loc = MemoryType.REMOTE_MEMORY
                    if tensor.x2 in self.tensor_node_maps:
                        assert self.tensor_node_maps[tensor.x2][-1].node_type == NodeType.MEM_STORE_NODE
                        load_node.parent.append(self.tensor_node_maps[tensor.x2][-1].id)
                    node_head.parent.append(load_node.id)
                    self.offload_nodes.append(load_node)
                else:
                    if tensor.x2 in self.tensor_node_maps:
                        x2_tail = self.tensor_node_maps[tensor.x2][-1]
                        node_head.parent.append(x2_tail.id)
                    # else tensor type and no offload
            # self.tensor_node_maps[tensor.id][0] = node_head
        return
    
    def readout(self):
        for npu_id in range(self.num_npu):
            output_filename = f"{self.eg_file}.{npu_id}.et"
            with open(output_filename, "wb") as g:
                for tensor_id in self.tensor_node_maps.keys():
                    nodes = self.tensor_node_maps[tensor_id]
                    for node in nodes:
                        encodeMessage(g, node)
                for node in self.offload_nodes:
                    encodeMessage(g, node)
        return
    
    def convert(self):
        self.get_tensor_node_maps()
        self.connect_nodes()
        self.readout()
   
if __name__ == '__main__':
    converter = Symbolic2ChakraConverter(
                    'sharding_spreadsheets/dp/processed_graphs/transformer_2.csv', 
                    'sharding_spreadsheets/dp/ets/transformer_2', 
                    256)
    symbol_value_map = {
        'bp': 2, 'B': 8, 'Seq': 1024, 'H': 32, 'D': 32, 'DF': 128, 'DI': 1024, 'DO': 32
    }
    converter.symbol_value_map = symbol_value_map
    symbols = converter.convert()
    hook = 0
    
    converter = Symbolic2ChakraConverterWithOffload(
                    'sharding_spreadsheets/dp/processed_graphs/transformer_2.csv', 
                    'sharding_spreadsheets/dp/offload_strategy/weight_1_leaf_1/transformer_2.csv',
                    'sharding_spreadsheets/dp/ets/transformer_2_offload_w1l1', 
                    256)
    symbol_value_map = {
        'bp': 2, 'B': 8, 'Seq': 1024, 'H': 32, 'D': 32, 'DF': 128, 'DI': 1024, 'DO': 32
    }
    converter.symbol_value_map = symbol_value_map
    symbols = converter.convert()
    hook = 0
    
