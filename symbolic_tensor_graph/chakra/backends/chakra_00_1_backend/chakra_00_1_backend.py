import os
from .et_def.et_def_pb2 import Node, NodeType, CollectiveCommType, MemoryType
from .protolib import *

from ..backend import FrontendNode, NodeBackendBase


class Chakra001Backend(NodeBackendBase):
    SCHEMA = "Chakra v0.0.1"
    DEFAULT_NETWORK_DIM = 3

    @classmethod
    def serialize_nodes(cls, backend_nodes, file):
        os.makedirs(os.path.split(file)[0], exist_ok=True)
        file = open(file, "wb")
        for node in backend_nodes:
            encodeMessage(file, node)
        file.close()

    @classmethod
    def alloc_backend_node(cls):
        return Node()

    @classmethod
    def set_node_common_attrs(cls, id, name, node_type, backend_node):
        def _get_backend_node_type(_frontend_node_type):
            if _frontend_node_type == FrontendNode.NodeType.COLL_COMM_NODE:
                return NodeType.COMM_COLL_NODE
            elif _frontend_node_type == FrontendNode.NodeType.COMM_RECV_NODE:
                return NodeType.COMM_RECV_NODE
            elif _frontend_node_type == FrontendNode.NodeType.COMM_SEND_NODE:
                return NodeType.COMM_SEND_NODE
            elif _frontend_node_type == FrontendNode.NodeType.COMP_NODE:
                return NodeType.COMP_NODE
            elif _frontend_node_type == FrontendNode.NodeType.MEM_LOAD_NODE:
                return NodeType.MEM_LOAD_NODE
            elif _frontend_node_type == FrontendNode.NodeType.MEM_STORE_NODE:
                return NodeType.MEM_STORE_NODE
            else:
                assert False

        backend_node.id = id
        backend_node.name = name
        backend_node.node_type = _get_backend_node_type(node_type)

    @classmethod
    def set_data_deps(cls, data_deps, backend_node):
        for dep in data_deps:
            if not dep in backend_node.parent:
                backend_node.parent.append(dep)

    @classmethod
    def set_ctrl_deps(cls, ctrl_deps, backend_node):
        for dep in ctrl_deps:
            if not dep in backend_node.parent:
                backend_node.parent.append(dep)

    @classmethod
    def set_comp_attrs(cls, num_ops, tensor_size, backend_node):
        backend_node.num_ops = int(num_ops)
        backend_node.tensor_size = int(tensor_size)

    @classmethod
    def set_coll_comm_attrs(cls, comm_size, comm_type, comm_group, backend_node, inputs, outputs):
        # IO should be none as v0.0.1 do not support those
        assert inputs is None
        assert outputs is None
        def _get_backend_comm_type(_frontend_comm_type):
            if _frontend_comm_type == FrontendNode.CollectiveType.ALL_GATHER:
                return CollectiveCommType.ALL_GATHER
            elif _frontend_comm_type == FrontendNode.CollectiveType.ALL_REDUCE:
                return CollectiveCommType.ALL_REDUCE
            elif _frontend_comm_type == FrontendNode.CollectiveType.ALL_TO_ALL:
                return CollectiveCommType.ALL_TO_ALL
            elif _frontend_comm_type == FrontendNode.CollectiveType.REDUCE_SCATTER:
                return CollectiveCommType.REDUCE_SCATTER
            else:
                assert False

        backend_node.comm_size = int(comm_size)
        backend_node.comm_type = _get_backend_comm_type(comm_type)
        backend_node.communicator_id = int(comm_group)
        for _ in range(cls.DEFAULT_NETWORK_DIM):
            backend_node.involved_dim.append(True)

    @classmethod
    def set_comm_send_attrs(cls, comm_size, comm_tag, comm_dst, backend_node):
        backend_node.comm_size = int(comm_size)
        backend_node.comm_tag = int(comm_tag)
        backend_node.comm_dst = int(comm_dst)

    @classmethod
    def set_comm_recv_attrs(cls, comm_size, comm_tag, comm_src, backend_node):
        backend_node.comm_size = int(comm_size)
        backend_node.comm_tag = int(comm_tag)
        backend_node.comm_src = int(comm_src)

    @classmethod
    def set_mem_attrs(cls, tensor_size, backend_node):
        backend_node.tensor_siize = int(tensor_size)
        backend_node.tensor_loc = MemoryType.REMOTE_MEMORY
