import random
import copy
import json
import sympy as sp
from ..tensor import Tensor
from .graph import HybridGraph, TensorGraph, BundledTensorGraph, BundledHybridGraph
from .coll_comm_matcher import CommunicationMatcherV2
from ..chakra.node import Node
from ..ops import Shadow, Identical, PlaceHolder


class ConvertChakra:
    with_comm_info = True

    @classmethod
    def _create_IOInfo(cls, tensor, symbol_map_value):
        if tensor.op_type == Identical.type_name:
            tensor = tensor.x1
        name = tensor.id
        shape = tensor.y_shape
        shape_str = Tensor.stringfy_shape(shape)
        name = name + shape_str
        size = Tensor.eval_expr(Tensor.eval_size(shape), symbol_map_value)
        IOInfo = {"name": name, "size": size}
        return IOInfo

    @classmethod
    def _insert_comp(cls, tensor, symbol_map_value, nodes_this_tensor):
        tensor_ops = Tensor.eval_expr(tensor.ops, symbol_map_value)
        ## TODO: change implementation
        ## some tensors with identical op will translate no comp/comms, and leading issues when connecting tensor's nodes as the connection cannot skip the empty tensor.
        ## to avoid this, new insert zero size comp as place holder, and remove them later.
        ## should be fixed
        # if tensor.op > 0:
        if True:
            comp_node = Node()
            comp_node.node_type = Node.NodeType.COMP_NODE
            comp_node.name = f"{tensor.id}_COMP"
            comp_node.num_ops = tensor_ops
            y_tensor_size = Tensor.eval_expr(
                Tensor.eval_size(tensor.y_shape), symbol_map_value
            )
            x1_tensor_size = 0
            if tensor.x1_shape is not None:
                x1_tensor_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x1_shape), symbol_map_value
                )
            x2_tensor_size = 0
            if tensor.x2_shape is not None:
                x2_tensor_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x2_shape), symbol_map_value
                )
            tensor_size = y_tensor_size + x1_tensor_size + x2_tensor_size
            comp_node.tensor_size = tensor_size
            comp_node.y_tensor_size = y_tensor_size
            comp_node.op_type = tensor.op_type
            nodes_this_tensor[HybridGraph.NodeType.COMP] = comp_node

    @classmethod
    def _insert_comm_x1(
        cls, tensor, symbol_map_value, parallel_syms, nodes_this_tensor
    ):
        if tensor.x1 is not None:
            matched_comms = CommunicationMatcherV2.match_comms(
                tensor.x1.y_shape,
                tensor.x1.y_hidden,
                tensor.x1_shape,
                tensor.x1_hidden,
                parallel_syms,
            )

            comm_nodes = list()
            for comm in matched_comms:
                parallel_dim = comm[3]
                if symbol_map_value[parallel_dim] == 1:
                    continue
                comm_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x1.y_shape), symbol_map_value
                )
                x1_comm_node = Node()
                x1_comm_node.node_type = Node.NodeType.COLL_COMM_NODE
                x1_comm_node.name = f"{tensor.id}_X1COMM"
                x1_comm_node.comm_size = comm_size
                x1_comm_node._comm_meta_data = comm
                if comm[0] == CommunicationMatcherV2.CommType.ALL_REDUCE:
                    x1_comm_node.comm_type = Node.CollectiveType.ALL_REDUCE
                elif comm[0] == CommunicationMatcherV2.CommType.ALL_GATHER:
                    x1_comm_node.comm_type = Node.CollectiveType.ALL_GATHER
                elif comm[0] == CommunicationMatcherV2.CommType.ALL_TO_ALL:
                    x1_comm_node.comm_type = Node.CollectiveType.ALL_TO_ALL
                elif comm[0] == CommunicationMatcherV2.CommType.REDUCE_SCATTER:
                    x1_comm_node.comm_type = Node.CollectiveType.REDUCE_SCATTER
                else:
                    assert False
                if len(comm_nodes) > 0:
                    x1_comm_node.data_deps.append(comm_nodes[-1].id)
                output_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x1_shape), symbol_map_value
                )
                x1_comm_node.y_tensor_size = output_size
                comm_nodes.append(x1_comm_node)
            if HybridGraph.NodeType.COMP in nodes_this_tensor and len(comm_nodes) > 0:
                nodes_this_tensor[HybridGraph.NodeType.COMP].data_deps.append(
                    comm_nodes[-1].id
                )
            for i, comm_node in enumerate(comm_nodes):
                if i == 0:
                    nodes_this_tensor[HybridGraph.NodeType.X1_COMM] = comm_node
                else:
                    nodes_this_tensor[f"{HybridGraph.NodeType.X1_COMM}{i}"] = comm_node

    @classmethod
    def _insert_comm_x2(
        cls, tensor, symbol_map_value, parallel_syms, nodes_this_tensor
    ):
        if tensor.x2 is not None:
            matched_comms = CommunicationMatcherV2.match_comms(
                tensor.x2.y_shape,
                tensor.x2.y_hidden,
                tensor.x2_shape,
                tensor.x2_hidden,
                parallel_syms,
            )

            comm_nodes = list()
            for comm in matched_comms:
                parallel_dim = comm[3]
                if symbol_map_value[parallel_dim] == 1:
                    continue
                comm_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x2.y_shape), symbol_map_value
                )
                x2_comm_node = Node()
                x2_comm_node.node_type = Node.NodeType.COLL_COMM_NODE
                x2_comm_node.name = f"{tensor.id}_X2_COMM"
                x2_comm_node.comm_size = comm_size
                x2_comm_node._comm_meta_data = comm
                if comm[0] == CommunicationMatcherV2.CommType.ALL_REDUCE:
                    x2_comm_node.comm_type = Node.CollectiveType.ALL_REDUCE
                elif comm[0] == CommunicationMatcherV2.CommType.ALL_GATHER:
                    x2_comm_node.comm_type = Node.CollectiveType.ALL_GATHER
                elif comm[0] == CommunicationMatcherV2.CommType.ALL_TO_ALL:
                    x2_comm_node.comm_type = Node.CollectiveType.ALL_TO_ALL
                elif comm[0] == CommunicationMatcherV2.CommType.REDUCE_SCATTER:
                    x2_comm_node.comm_type = Node.CollectiveType.REDUCE_SCATTER
                else:
                    assert False
                if len(comm_nodes) > 0:
                    x2_comm_node.data_deps.append(comm_nodes[-1].id)
                output_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x2_shape), symbol_map_value
                )
                x2_comm_node.y_tensor_size = output_size
                comm_nodes.append(x2_comm_node)
            if HybridGraph.NodeType.COMP in nodes_this_tensor and len(comm_nodes) > 0:
                nodes_this_tensor[HybridGraph.NodeType.COMP].data_deps.append(
                    comm_nodes[-1].id
                )
            for i, comm_node in enumerate(comm_nodes):
                if i == 0:
                    nodes_this_tensor[HybridGraph.NodeType.X2_COMM] = comm_node
                else:
                    nodes_this_tensor[f"{HybridGraph.NodeType.X2_COMM}{i}"] = comm_node

    @classmethod
    def _get_output_node(cls, nodes_this_tensor):
        if HybridGraph.NodeType.COMP in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.COMP]
        elif HybridGraph.NodeType.X1_COMM in nodes_this_tensor:
            x1_keys = list()
            for key in nodes_this_tensor.keys():
                if HybridGraph.NodeType.X1_COMM in key:
                    x1_keys.append(key)
            x1_comm_id = -1
            for key in x1_keys:
                key = key.replace(HybridGraph.NodeType.X1_COMM, "")
                if key == "":
                    id_ = 0
                else:
                    id_ = int(key)
                if id_ > x1_comm_id:
                    x1_comm_id = id_
            if x1_comm_id == 0:
                return nodes_this_tensor[HybridGraph.NodeType.X1_COMM]
            else:
                return nodes_this_tensor[f"{HybridGraph.NodeType.X1_COMM}{x1_comm_id}"]
        else:
            assert False

    @classmethod
    def _get_x1_input_node(cls, nodes_this_tensor):
        if HybridGraph.NodeType.X1_COMM in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.X1_COMM]
        elif HybridGraph.NodeType.COMP in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.COMP]
        else:
            assert False

    @classmethod
    def _get_x2_input_node(cls, nodes_this_tensor):
        if HybridGraph.NodeType.X2_COMM in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.X2_COMM]
        elif HybridGraph.NodeType.COMP in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.COMP]
        else:
            assert False

    @classmethod
    def _tensor_to_nodes(cls, tensor, symbol_map_value, parallel_syms):
        nodes_this_tensor = dict()
        cls._insert_comp(tensor, symbol_map_value, nodes_this_tensor)
        cls._insert_comm_x1(tensor, symbol_map_value, parallel_syms, nodes_this_tensor)
        cls._insert_comm_x2(tensor, symbol_map_value, parallel_syms, nodes_this_tensor)

        return nodes_this_tensor

    @classmethod
    def _connect_tensors_node(cls, tensor_map_nodes):
        for tensor in tensor_map_nodes.keys():
            if not tensor.x1 is None:
                x1_to = cls._get_x1_input_node(tensor_map_nodes[tensor])
                x1_from = cls._get_output_node(tensor_map_nodes[tensor.x1])
                if not x1_from is None:
                    x1_to.data_deps.append(x1_from.id)
            if not tensor.x2 is None:
                x2_to = cls._get_x2_input_node(tensor_map_nodes[tensor])
                x2_from = cls._get_output_node(tensor_map_nodes[tensor.x2])
                if not x2_from is None:
                    x2_to.data_deps.append(x2_from.id)

    @classmethod
    def _sanity_check(cls, tensor_graph, symbol_map_value, parallel_syms):
        assert isinstance(tensor_graph, TensorGraph)
        for symbol in tensor_graph.get_symbols():
            assert symbol in symbol_map_value
        for parallel_sym in parallel_syms:
            assert parallel_sym in symbol_map_value

    @classmethod
    def _clean_empty_comp(cls, hybrid_graph):
        # TODO: might buggy when there are multiple nodes with 0 ops.
        while True:
            nodes = hybrid_graph.get_nodes()
            assert all(len(node.ctrl_deps) == 0 for node in nodes)
            empty_comp_nodes = list()
            node_parent_to_child_link = hybrid_graph.get_node_parent_to_child_link()
            node_id_map_node = hybrid_graph.get_node_id_map_node()
            node_id_map_tensor = hybrid_graph.get_node_id_map_tensor()
            for node in nodes:
                if node.node_type == Node.NodeType.COMP_NODE:
                    if node.num_ops == 0:
                        empty_comp_nodes.append(node)
            if len(empty_comp_nodes) == 0:
                break
            print(f"empty_comp_nodes: {len(empty_comp_nodes)}")
            for empty_comp_node in empty_comp_nodes:
                # if len(empty_comp_node.data_deps) >= 2:
                if False:  # hotfix: reduce chain
                    assert False
                elif len(empty_comp_node.data_deps) != 0:
                    # if the empty comp node's parent is also another empty node, give up this round and try another
                    # otherwise its child will to connected to another empty node, which will be deleted this round again.
                    has_empty_parent = False
                    for parent_id in empty_comp_node.data_deps:
                        parent_node = node_id_map_node[parent_id]
                        if parent_node in empty_comp_nodes:
                            has_empty_parent = True
                            break
                            # continue
                    if has_empty_parent:
                        continue
                tensor = node_id_map_tensor[empty_comp_node.id]
                nodes_this_tensor = hybrid_graph.tensor_map_nodes[tensor]
                nodes_this_tensor_keys = copy.copy(list(nodes_this_tensor.keys()))
                for key in nodes_this_tensor_keys:
                    if nodes_this_tensor[key].id == empty_comp_node.id:
                        del nodes_this_tensor[key]
                for child_id in node_parent_to_child_link[empty_comp_node.id]:
                    child = node_id_map_node[child_id]
                    child.data_deps.remove(empty_comp_node.id)
                    # if len(empty_comp_node.data_deps) != 0:
                    for parent_id in empty_comp_node.data_deps:
                        child.data_deps.append(parent_id)

    @classmethod
    def apply(cls, tensor_graph, symbol_map_value, parallel_syms):
        cls._sanity_check(tensor_graph, symbol_map_value, parallel_syms)
        assert hasattr(tensor_graph, "comm_groups")
        for sym in copy.copy(parallel_syms):
            if symbol_map_value[sym] == 1:
                parallel_syms.remove(sym)
        tensor_map_nodes = dict()
        for tensor in tensor_graph.tensors:
            nodes_this_tensor = cls._tensor_to_nodes(
                tensor, symbol_map_value, parallel_syms
            )
            for node in nodes_this_tensor.values():
                if node.node_type == Node.NodeType.COLL_COMM_NODE:
                    parallel_dim = node._comm_meta_data[3]
                    assert parallel_dim in tensor_graph.comm_groups
                    node.comm_group = tensor_graph.comm_groups[parallel_dim][0]
            tensor_map_nodes[tensor] = nodes_this_tensor
        cls._connect_tensors_node(tensor_map_nodes)
        cls._comm_info_post_process(tensor_map_nodes, symbol_map_value)
        graph = HybridGraph(tensor_graph.tensors, tensor_map_nodes, symbol_map_value)
        cls._clean_empty_comp(graph)
        return graph

    @classmethod
    def _comm_info_post_process(cls, tensor_map_nodes, symbol_map_value):
        if not cls.with_comm_info:
            return
        for tensor in tensor_map_nodes.keys():
            nodes_this_tensor = tensor_map_nodes[tensor]
            for node_type in nodes_this_tensor.keys():
                if node_type == HybridGraph.NodeType.COMP:
                    inputs = list()
                    if tensor.x1 is not None:
                        inputs.append(cls._create_IOInfo(tensor.x1, symbol_map_value))
                    if tensor.x2 is not None:
                        inputs.append(cls._create_IOInfo(tensor.x2, symbol_map_value))
                    outputs = list()
                    outputs.append(cls._create_IOInfo(tensor, symbol_map_value))
                    nodes_this_tensor[node_type].inputs = inputs
                    nodes_this_tensor[node_type].outputs = outputs
                elif HybridGraph.NodeType.X1_COMM in node_type:
                    inputs = list()
                    inputs.append(cls._create_IOInfo(tensor.x1, symbol_map_value))
                    outputs = list()
                    nodes_this_tensor[node_type].inputs = inputs
                    nodes_this_tensor[node_type].outputs = outputs
                elif HybridGraph.NodeType.X2_COMM in node_type:
                    inputs = list()
                    inputs.append(cls._create_IOInfo(tensor.x2, symbol_map_value))
                    outputs = list()
                    nodes_this_tensor[node_type].inputs = inputs
                    nodes_this_tensor[node_type].outputs = outputs
                elif node_type == HybridGraph.NodeType.Y_RECV:
                    inputs = list()
                    outputs = list()
                    outputs.append(cls._create_IOInfo(tensor, symbol_map_value))
                    nodes_this_tensor[node_type].inputs = inputs
                    nodes_this_tensor[node_type].outputs = outputs
                elif HybridGraph.NodeType.Y_SEND in node_type:
                    inputs = list()
                    outputs = list()
                    inputs.append(cls._create_IOInfo(tensor, symbol_map_value))
                    nodes_this_tensor[node_type].inputs = inputs
                    nodes_this_tensor[node_type].outputs = outputs
                else:
                    assert False


class BundledConvertChakra:
    class _ConvertChakra(ConvertChakra):
        @classmethod
        def _get_output_node(cls, nodes_this_tensor):
            if HybridGraph.NodeType.Y_RECV in nodes_this_tensor:
                return nodes_this_tensor[HybridGraph.NodeType.Y_RECV]
            else:
                return super()._get_output_node(nodes_this_tensor)

        @classmethod
        def _insert_send_node(
            cls, tensor, nodes_this_tensor, dst_rank, tag, symbol_map_value
        ):
            node = Node()
            node.node_type = Node.NodeType.COMM_SEND_NODE
            node.name = tensor.id + "_Y_SEND"
            node.data_deps.append(cls._get_output_node(nodes_this_tensor).id)
            node.comm_size = Tensor.eval_expr(
                Tensor.eval_size(tensor.y_shape), symbol_map_value
            )
            node.comm_tag = tag
            node.comm_dst = dst_rank
            node.y_tensor_size = 0
            nodes_this_tensor[f"{HybridGraph.NodeType.Y_SEND}{tag}"] = node

        @classmethod
        def _insert_recv_node(
            cls, tensor, nodes_this_tensor, src_rank, tag, symbol_map_value
        ):
            assert tensor.op_type == Shadow.type_name
            node = Node()
            node.node_type = Node.NodeType.COMM_RECV_NODE
            node.name = tensor.id + "_Y_RECV"
            node.comm_size = Tensor.eval_expr(
                Tensor.eval_size(tensor.y_shape), symbol_map_value
            )
            node.comm_tag = tag
            node.comm_src = src_rank
            node.y_tensor_size = node.comm_size
            nodes_this_tensor[HybridGraph.NodeType.Y_RECV] = node

        @classmethod
        def apply_before_cross_bucket_comms(
            cls, tensor_graph, symbol_map_value, spatial_parallel_syms
        ):
            cls._sanity_check(tensor_graph, symbol_map_value, spatial_parallel_syms)
            tensor_map_nodes = dict()
            for tensor in tensor_graph.tensors:
                nodes_this_tensor = cls._tensor_to_nodes(
                    tensor, symbol_map_value, spatial_parallel_syms
                )
                for node in nodes_this_tensor.values():
                    if node.node_type == Node.NodeType.COLL_COMM_NODE:
                        parallel_dim = node._comm_meta_data[3]
                        assert parallel_dim in tensor_graph.comm_groups
                        node.comm_group = tensor_graph.comm_groups[parallel_dim][0]

                tensor_map_nodes[tensor] = nodes_this_tensor
            return tensor_map_nodes

        @classmethod
        def apply_after_cross_bucket_comms(cls, tensor_map_nodes, symbol_map_value):
            cls._connect_tensors_node(tensor_map_nodes)
            cls._comm_info_post_process(tensor_map_nodes, symbol_map_value)
            graph = HybridGraph(
                tensor_map_nodes.keys(), tensor_map_nodes, symbol_map_value
            )
            cls._clean_empty_comp(graph)
            cls.add_ctrl_deps(graph.tensor_map_nodes)
            return graph

        @classmethod
        def add_ctrl_deps(cls, tensor_map_nodes):
            for tensor in tensor_map_nodes.keys():
                if not "ctrl_deps" in tensor.extra_attr.keys():
                    continue
                for parent in tensor.extra_attr["ctrl_deps"]:
                    if parent.op_type == PlaceHolder.type_name:
                        continue
                    while len(tensor_map_nodes[parent]) == 0:
                        assert parent.op_type == Identical.type_name
                        parent = parent.x1
                    parent_node = cls._get_output_node(tensor_map_nodes[parent])
                    child_node = cls._get_x1_input_node(tensor_map_nodes[tensor])
                    child_node.ctrl_deps.append(parent_node.id)

        # @classmethod
        # def _clean_empty_comp(cls, graph):
        #     # soft clean
        #     for node in graph.get_nodes():
        #         if node.node_type == Node.NodeType.COMP_NODE:
        #             if node.num_ops == 0:
        #                 node.num_ops = 10
        #                 node.tensor_size = 10

    @classmethod
    def _get_comm_group_id_map_num_comm_group(
        cls, readable_comm_groups, readable_rank_map_number_rank
    ):
        comm_group_id_map_number_comm_group = dict()

        def _tuple_to_dict(tuple_):
            ret = dict()
            for key, value in tuple_:
                ret[key] = value
            return ret

        for readable_comm_group in readable_comm_groups.values():
            comm_group_id = readable_comm_group[0]
            readable_comm_group = readable_comm_group[1:]
            num_comm_group = list()
            for asked_readable_rank in readable_comm_group:
                asked_readable_rank_dict = _tuple_to_dict(asked_readable_rank)
                matched_rank = None
                for test_readable_rank in readable_rank_map_number_rank.keys():
                    if _tuple_to_dict(test_readable_rank) == asked_readable_rank_dict:
                        matched_rank = readable_rank_map_number_rank[test_readable_rank]
                        break
                assert not matched_rank is None
                num_comm_group.append(matched_rank)
            comm_group_id_map_number_comm_group[comm_group_id] = num_comm_group
        return comm_group_id_map_number_comm_group

    @classmethod
    def _readout_comm_group(cls, comm_group_file, comm_group):
        f = open(comm_group_file, "w")
        json.dump(comm_group, f)
        f.close()

    @classmethod
    def apply(
        cls,
        bundled_graph,
        symbol_map_value,
        comm_group_file,
        readable_rank_map_number_rank=None,
    ):
        for symbol in bundled_graph.symbol_map_value:
            assert bundled_graph.symbol_map_value[symbol] == symbol_map_value[symbol]
        for sym in copy.copy(bundled_graph.spatial_parallel_dims):
            if symbol_map_value[sym] == 1:
                bundled_graph.spatial_parallel_dims.remove(sym)
        if readable_rank_map_number_rank is None:
            readable_rank_map_number_rank = dict()
            for num_rank, asked_readable_rank in enumerate(bundled_graph.graphs.keys()):
                readable_rank_map_number_rank[asked_readable_rank] = num_rank
        assert hasattr(bundled_graph, "comm_groups")
        comm_group = cls._get_comm_group_id_map_num_comm_group(
            bundled_graph.comm_groups, readable_rank_map_number_rank
        )
        cls._readout_comm_group(comm_group_file, comm_group)

        buckets = dict()
        for asked_readable_rank in bundled_graph.graphs.keys():
            tensor_graph = bundled_graph.graphs[asked_readable_rank]
            tensor_map_nodes = cls._ConvertChakra.apply_before_cross_bucket_comms(
                tensor_graph, symbol_map_value, bundled_graph.spatial_parallel_dims
            )
            tensor_id_map_tensor = tensor_graph.get_tensor_id_map_tensor()
            buckets[asked_readable_rank] = [tensor_map_nodes, tensor_id_map_tensor]

        tag_cnt = random.randint(0, int(1e6))
        for link in bundled_graph.remote_parent_shadow_pairs:
            (remote_readable_rank, remote_id), (shadow_readable_rank, shadow_id) = link
            remote_tensor = buckets[remote_readable_rank][1][remote_id]
            remote_tensor_nodes = buckets[remote_readable_rank][0][remote_tensor]
            shadow_tensor = buckets[shadow_readable_rank][1][shadow_id]
            shadow_tensor_nodes = buckets[shadow_readable_rank][0][shadow_tensor]
            remote_num_rank = readable_rank_map_number_rank[remote_readable_rank]
            shadow_num_rank = readable_rank_map_number_rank[shadow_readable_rank]
            cls._ConvertChakra._insert_send_node(
                remote_tensor,
                remote_tensor_nodes,
                shadow_num_rank,
                tag_cnt,
                symbol_map_value,
            )
            cls._ConvertChakra._insert_recv_node(
                shadow_tensor,
                shadow_tensor_nodes,
                remote_num_rank,
                tag_cnt,
                symbol_map_value,
            )
            tag_cnt += 1

        for asked_readable_rank in bundled_graph.graphs.keys():
            hybrid_graph = cls._ConvertChakra.apply_after_cross_bucket_comms(
                buckets[asked_readable_rank][0], symbol_map_value
            )
            buckets[asked_readable_rank] = hybrid_graph
        return BundledHybridGraph(
            buckets,
            bundled_graph.remote_parent_shadow_pairs,
            bundled_graph.spatial_parallel_dims,
            bundled_graph.temporal_parallel_dims,
            symbol_map_value,
            readable_rank_map_number_rank,
        )
