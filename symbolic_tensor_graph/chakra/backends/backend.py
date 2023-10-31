from ..node import Node as FrontendNode
from typing import List


class NodeBackendBase:
    SCHEMA = "UNDEFINED"

    @classmethod
    def readout(cls, node: FrontendNode):
        frontend_node = node
        frontend_node.sanity_check()
        backend_node = cls.alloc_backend_node()
        cls.set_node_common_attrs(
            frontend_node.id, frontend_node.name, frontend_node.node_type, backend_node
        )
        cls.set_data_deps(frontend_node.data_deps, backend_node)
        cls.set_ctrl_deps(frontend_node.ctrl_deps, backend_node)

        if frontend_node.node_type == FrontendNode.NodeType.COLL_COMM_NODE:
            cls.set_coll_comm_attrs(
                frontend_node.comm_size, frontend_node.comm_type, backend_node
            )
        elif frontend_node.node_type == FrontendNode.NodeType.COMM_RECV_NODE:
            cls.set_comm_recv_attrs(
                frontend_node.comm_size,
                frontend_node.comm_tag,
                frontend_node.comm_src,
                backend_node,
            )
        elif frontend_node.node_type == FrontendNode.NodeType.COMM_SEND_NODE:
            cls.set_comm_send_attrs(
                frontend_node.comm_size,
                frontend_node.comm_tag,
                frontend_node.comm_dsr,
                backend_node,
            )
        elif frontend_node.node_type == FrontendNode.NodeType.COMP_NODE:
            cls.set_comp_attrs(
                frontend_node.num_ops, frontend_node.tensor_size, backend_node
            )
        elif frontend_node.node_type == FrontendNode.NodeType.MEM_LOAD_NODE:
            cls.set_mem_attrs(frontend_node.tensor_size, backend_node)
        elif frontend_node.node_type == FrontendNode.NodeType.MEM_STORE_NODE:
            cls.set_mem_attrs(frontend_node.tensor_size, backend_node)
        else:
            raise NotImplementedError()
        return backend_node

    @classmethod
    def readout_nodes(cls, frontend_nodes: List[FrontendNode], file):
        backend_nodes = list()
        for node in frontend_nodes:
            backend_nodes.append(cls.readout(node))
        cls.serialize_nodes(backend_nodes, file)

    @classmethod
    def serialize_nodes(cls, backend_nodes, file):
        raise NotImplementedError()

    @classmethod
    def alloc_backend_node(cls):
        raise NotImplementedError()

    @classmethod
    def set_node_common_attrs(cls, id, name, node_type, backend_node):
        raise NotImplementedError()

    @classmethod
    def set_data_deps(cls, data_deps, backend_node):
        raise NotImplementedError()

    @classmethod
    def set_ctrl_deps(cls, ctrl_deps, backend_node):
        raise NotImplementedError()

    @classmethod
    def set_comp_attrs(cls, num_ops, tensor_size, backend_node):
        raise NotImplementedError()

    @classmethod
    def set_coll_comm_attrs(cls, comm_size, comm_type, backend_node):
        raise NotImplementedError()

    @classmethod
    def set_comm_send_attrs(cls, comm_size, comm_tag, comm_dst, backend_node):
        raise NotImplementedError()

    @classmethod
    def set_comm_recv_attrs(cls, comm_size, comm_tag, comm_src, backend_node):
        raise NotImplementedError()

    @classmethod
    def set_mem_attrs(cls, tensor_size, backend_node):
        raise NotImplementedError()
