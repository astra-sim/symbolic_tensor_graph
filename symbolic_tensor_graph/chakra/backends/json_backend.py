import json
import os
from .backend import FrontendNode, NodeBackendBase


class JsonBackend(NodeBackendBase):
    SCHEMA = "Json"

    @classmethod
    def serialize_nodes(cls, backend_nodes, file):
        os.makedirs(os.path.split(file)[0], exist_ok=True)
        data = {"nodes": backend_nodes}
        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def alloc_backend_node(cls):
        return {}

    @classmethod
    def set_node_common_attrs(cls, id_, name, node_type, backend_node, inputs, outputs):
        def _get_backend_node_type(_frontend_node_type):
            frontend_node_type_map_string = {
                FrontendNode.NodeType.COLL_COMM_NODE: "coll_comm_node",
                FrontendNode.NodeType.COMM_RECV_NODE: "comm_recv_node",
                FrontendNode.NodeType.COMM_SEND_NODE: "comm_send_node",
                FrontendNode.NodeType.COMP_NODE: "comp_node",
                FrontendNode.NodeType.MEM_LOAD_NODE: "mem_load_node",
                FrontendNode.NodeType.MEM_STORE_NODE: "mem_store_node",
            }
            assert _frontend_node_type in frontend_node_type_map_string
            return frontend_node_type_map_string[_frontend_node_type]

        backend_node["id"] = id_
        backend_node["name"] = name
        backend_node["node_type"] = _get_backend_node_type(node_type)
        if inputs is not None:
            backend_node["inputs"] = inputs
        if outputs is not None:
            backend_node["outputs"] = outputs

    @classmethod
    def set_data_deps(cls, data_deps, backend_node):
        backend_node["data_deps"] = []
        for dep in data_deps:
            backend_node["data_deps"].append(dep)

    @classmethod
    def set_ctrl_deps(cls, ctrl_deps, backend_node):
        backend_node["ctrl_deps"] = []
        for dep in ctrl_deps:
            backend_node["ctrl_deps"].append(dep)

    @classmethod
    def set_comp_attrs(cls, num_ops, tensor_size, backend_node):
        backend_node["num_ops"] = int(num_ops)
        backend_node["tensor_size"] = int(tensor_size)

    @classmethod
    def set_coll_comm_attrs(cls, comm_size, comm_type, comm_group, backend_node):
        def _get_backend_comm_type(_frontend_comm_type):
            if _frontend_comm_type == FrontendNode.CollectiveType.ALL_GATHER:
                return "ALL_GATHER"
            elif _frontend_comm_type == FrontendNode.CollectiveType.ALL_REDUCE:
                return "ALL_REDUCE"
            elif _frontend_comm_type == FrontendNode.CollectiveType.ALL_TO_ALL:
                return "ALL_TO_ALL"
            elif _frontend_comm_type == FrontendNode.CollectiveType.REDUCE_SCATTER:
                return "REDUCE_SCATTER"
            else:
                assert False

        backend_node["comm_size"] = int(comm_size)
        backend_node["comm_type"] = _get_backend_comm_type(comm_type)
        backend_node["comm_group"] = int(comm_group)

    @classmethod
    def set_comm_recv_attrs(cls, comm_size, comm_tag, comm_src, backend_node):
        backend_node["comm_size"] = int(comm_size)
        backend_node["comm_tag"] = int(comm_tag)
        backend_node["comm_src"] = int(comm_src)

    @classmethod
    def set_comm_send_attrs(cls, comm_size, comm_tag, comm_dst, backend_node):
        backend_node["comm_size"] = int(comm_size)
        backend_node["comm_tag"] = int(comm_tag)
        backend_node["comm_dst"] = int(comm_dst)

    @classmethod
    def set_mem_attrs(cls, tensor_size, backend_node):
        backend_node["tensor_size"] = int(tensor_size)
