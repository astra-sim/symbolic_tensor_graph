import copy
import typing
import sympy as sp
from ..ops import Add, PlaceHolder, Customized, Identical
from ..tensor import Tensor
from ..graph.graph import TensorGraph, BundledHybridGraph, HybridGraph
from ..graph.replicate_graph import ReplicateGraph
from ..graph.connect_graph import ConnectGraph


class GradUpdater:
    @classmethod
    def _default_revision_fn(cls, old_replicate):
        return str(int(old_replicate) + 1)

    @classmethod
    def _update_grad(cls, tensor, grad, new_revision_fn):
        print(f"{tensor.name}: {grad.y_shape} @ {grad.y_hidden}")
        updated_tensor = Tensor(create_empty=True)
        updated_tensor.name = tensor.name
        updated_tensor.require_grads = tensor.require_grads
        updated_tensor.x1 = tensor
        updated_tensor.x2 = grad
        updated_tensor.op_type = Add.type_name
        updated_tensor.op_attr = None
        updated_tensor.x1_shape = tensor.y_shape
        updated_tensor.x1_hidden = tensor.y_hidden
        updated_tensor.x2_shape = tensor.y_shape
        updated_tensor.x2_hidden = tensor.y_hidden
        updated_tensor.revision = new_revision_fn(tensor.revision)
        return updated_tensor

    @classmethod
    def apply(cls, graph, new_revision=None, inplace=False):
        if not inplace:
            graph = copy.deepcopy(graph)

        if new_revision is None:
            new_revision = cls._default_revision_fn
        elif isinstance(new_revision, str):
            new_revision = lambda _: new_revision
        elif isinstance(new_revision, typing.Callable):
            pass
        else:
            assert False

        tensor_id_map_tensor = graph.get_tensor_id_map_tensor()
        for tensor_id in tensor_id_map_tensor.keys():
            tensor = tensor_id_map_tensor[tensor_id]
            if tensor.require_grads:
                grad = tensor._grad
                assert grad in graph.out_tensors
                updated_tensor = cls._update_grad(tensor, grad, new_revision)
                graph.out_tensors.remove(grad)
                graph.tensors.append(updated_tensor)
                graph.out_tensors.append(updated_tensor)
        return graph


class FSDPWeightGradManager:
    @classmethod
    def fsdp_weight_distributor(cls, weights, name=None):
        fsdp = sp.symbols("fsdp")

        if name is None:
            name = ""
        reduce_expr = sp.parse_expr("1")
        total_weight_size = 0
        for weight in weights:
            total_weight_size += Tensor.eval_size(weight.y_shape)

        sharded_weight = Tensor(create_empty=True)
        sharded_weight.name = f"{name}_sharded_weight"
        sharded_weight.revision = weights[0].revision
        sharded_weight.require_grads = True
        sharded_weight.op_type = PlaceHolder.type_name
        sharded_weight.x1_shape = [total_weight_size / fsdp]
        sharded_weight.x1_hidden = [reduce_expr]

        assembled_weight = Tensor(create_empty=True)
        assembled_weight.name = f"{name}_assembled_weight"
        assembled_weight.revision = weights[0].revision
        assembled_weight.require_grads = False
        assembled_weight.op_type = Identical.type_name
        assembled_weight.x1 = sharded_weight
        assembled_weight.x1_shape = [total_weight_size]
        assembled_weight.x1_hidden = [reduce_expr]

        for weight in weights:
            assert weight.op_type == PlaceHolder.type_name
            assert weight.require_grads

            weight.op_type = Customized.type_name
            weight.op_attr = "0"
            weight.require_grads = False
            weight.x1 = assembled_weight
            weight.x2_shape = weight.x1_shape
            weight.x2_hidden = weight.x1_hidden
            weight.x1_shape = assembled_weight.x1_shape
            weight.x1_hidden = assembled_weight.x1_hidden
        return sharded_weight, assembled_weight

    @classmethod
    def fsdp_backward_weight_shadow(
        cls,
        tensors,
        sharded_weight,
        assembled_weight,
        weights,
        name=None,
        grad_filter=None,
    ):
        if grad_filter is None:

            def grad_filter(_t):
                return _t.name.split(".")[-1].startswith("d")

        if name is None:
            name = ""
        backward_weights = list()

        assembled_weight_backward = Tensor(create_empty=True)
        assembled_weight_backward.name = f"{name}_assembled_weight_backward"
        assembled_weight_backward.revision = sharded_weight.revision
        assembled_weight_backward.require_grads = False
        assembled_weight_backward.op_type = Identical.type_name
        assembled_weight_backward.x1 = sharded_weight
        assembled_weight_backward.x1_shape = assembled_weight.x1_shape
        assembled_weight_backward.x1_hidden = assembled_weight.x1_hidden

        for weight in weights:
            backward_weight = Tensor(create_empty=True)
            backward_weight.name = f"{name}_{weight.name}_backward"
            backward_weight.revision = weight.revision
            backward_weight.require_grads = False
            backward_weight.op_type = Customized.type_name
            backward_weight.op_attr = "0"
            backward_weight.x1 = assembled_weight_backward
            backward_weight.x1_shape = assembled_weight_backward.x1_shape
            backward_weight.x1_hidden = assembled_weight_backward.x1_hidden
            backward_weight.x2_shape = weight.y_shape
            backward_weight.x2_hidden = weight.y_hidden
            backward_weights.append(backward_weight)

        weight_map_backward_weight = dict()
        for weight, backward_weight in zip(weights, backward_weights):
            weight_map_backward_weight[weight] = backward_weight

        for tensor in filter(grad_filter, tensors):
            if tensor.x1 in weight_map_backward_weight:
                tensor.x1 = weight_map_backward_weight[tensor.x1]
            if tensor.x2 in weight_map_backward_weight:
                tensor.x2 = weight_map_backward_weight[tensor.x2]
        return backward_weights, assembled_weight_backward

    @classmethod
    def fsdp_grad_gatherer(cls, grads, assembled_weight, name=None):
        if name is None:
            name = ""
        reduce_expr = sp.parse_expr("1/(cp*dp)")
        assembled_grad = Tensor(create_empty=True)
        assembled_grad.name = f"{name}_assembled_grad"
        assembled_grad.revision = grads[0].revision
        assembled_grad.require_grads = False
        assembled_grad.op_type = Customized.type_name
        assembled_grad.op_attr = "0"
        assembled_grad.x1_shape = grads[0].y_shape
        assembled_grad.x1_hidden = grads[0].y_hidden
        assembled_grad.x2_shape = assembled_weight.y_shape
        assembled_grad.x2_hidden = grads[0].y_hidden
        assembled_grad.x1 = grads[0]
        assembled_grad.extra_attr["data_deps"] = list()
        for i, grad in enumerate(grads):
            if i == 0:
                continue
            assembled_grad.extra_attr["data_deps"].append(grad)

        sharded_grad = Tensor(create_empty=True)
        sharded_grad.name = f"{name}_sharded_grad"
        sharded_grad.revision = grads[0].revision
        sharded_grad.require_grads = False
        sharded_grad.op_type = Identical.type_name
        sharded_grad.x1 = assembled_grad

        sharded_weight = assembled_weight.x1
        sharded_grad.x1_shape = sharded_weight.y_shape
        sharded_grad.x1_hidden = sharded_weight.y_hidden
        sharded_grad.grad_of = sharded_weight
        sharded_weight._grad = sharded_grad

        return sharded_grad, assembled_grad

    @classmethod
    def apply(cls, graph, inplace=False):
        if not inplace:
            graph = copy.deepcopy(graph)
        tensor_id_map_tensor = graph.get_tensor_id_map_tensor()
        weights = list()
        grads = list()
        for tensor in tensor_id_map_tensor.values():
            if tensor.op_type == PlaceHolder.type_name and tensor.require_grads:
                weights.append(tensor)
                grads.append(tensor._grad)
                graph.in_tensors.remove(tensor)
                graph.out_tensors.remove(tensor._grad)
        sharded_weight, assembled_weight = cls.fsdp_weight_distributor(weights)
        sharded_grad, assembled_grad = cls.fsdp_grad_gatherer(grads, assembled_weight)
        graph.tensors.append(sharded_weight)
        graph.tensors.append(assembled_weight)
        graph.tensors.append(sharded_grad)
        graph.tensors.append(assembled_grad)
        graph.in_tensors.append(sharded_weight)
        graph.out_tensors.append(sharded_grad)
        backward_weights, assembled_weight_backward = cls.fsdp_backward_weight_shadow(
            graph.tensors, sharded_weight, assembled_weight, weights
        )
        graph.tensors.extend(backward_weights)
        graph.tensors.append(assembled_weight_backward)

        return graph


class MicroBatchReplicator:
    @classmethod
    def get_weights_grads_others(cls, graph):
        weights = list()
        grads = list()
        others = list()
        for tensor in graph.tensors:
            if tensor.op_type == PlaceHolder.type_name and tensor.require_grads:
                weights.append(tensor)
                grads.append(tensor._grad)
            else:
                others.append(tensor)
        for grad in grads:
            others.remove(grad)
        return weights, grads, others

    @classmethod
    def apply(cls, graph, symbol_map_value, inplace=False):
        raise NotImplementedError("Too slow, use the postprocess instead")
        batch, microbatch = sp.symbols("Batch MicroBatch")
        assert microbatch in symbol_map_value
        assert batch in symbol_map_value
        num_batches = symbol_map_value[batch] / symbol_map_value[microbatch]
        assert int(num_batches) == num_batches
        num_batches = int(num_batches)

        if not inplace:
            graph = copy.deepcopy(graph)
        weights, grads, others = cls.get_weights_grads_others(graph)

        microbatch_graphs = list()

        for i in range(num_batches):
            microbatch_graph = ReplicateGraph.apply(
                graph, f"mb{i}.%s", old_symbol_map_new_symbol={batch: microbatch}
            )
            microbatch_graphs.append(microbatch_graph)

        merged_graph = ConnectGraph.apply(microbatch_graphs, dict())

        new_weight_map_old_weight = dict()
        old_grad_map_new_grads = dict()
        for tensor in merged_graph.tensors:
            for weight in weights:
                if tensor.name[tensor.name.find(".") + 1 :] == weight.name:
                    new_weight_map_old_weight[tensor] = weight
                    break
            for grad in grads:
                if tensor.name[tensor.name.find(".") + 1 :] == grad.name:
                    if not grad in old_grad_map_new_grads:
                        old_grad_map_new_grads[grad] = list()
                    old_grad_map_new_grads[grad].append(tensor)
                    break

        for tensor in merged_graph.tensors:
            if tensor.x1 in new_weight_map_old_weight:
                tensor.x1 = new_weight_map_old_weight[tensor.x1]
            if tensor.x2 in new_weight_map_old_weight:
                tensor.x2 = new_weight_map_old_weight[tensor.x2]

        old_grad_map_merged_grad = dict()
        for old_grad in old_grad_map_new_grads:
            merged_grad = Tensor(create_empty=True)
            merged_grad.name = f"{old_grad.name}"
            merged_grad.revision = old_grad.revision
            merged_grad.require_grads = False
            merged_grad.op_type = Customized.type_name
            merged_grad.op_attr = str(Tensor.eval_size(old_grad.y_shape))
            merged_grad.x1 = old_grad_map_new_grads[old_grad][0]
            merged_grad.x1_shape = old_grad_map_new_grads[old_grad][0].y_shape
            merged_grad.x1_hidden = old_grad_map_new_grads[old_grad][0].y_hidden
            merged_grad.x2_shape = old_grad_map_new_grads[old_grad][0].y_shape
            merged_grad.x2_hidden = old_grad_map_new_grads[old_grad][0].y_hidden
            merged_grad.extra_attr["data_deps"] = list()
            for i, new_grad in enumerate(old_grad_map_new_grads[old_grad]):
                if i == 0:
                    continue
                merged_grad.extra_attr["data_deps"].append(new_grad)
            old_grad_map_merged_grad[old_grad] = merged_grad
            merged_graph.tensors.append(merged_grad)
            merged_graph.out_tensors.append(merged_grad)
            for new_grad in old_grad_map_new_grads[old_grad]:
                merged_graph.out_tensors.remove(new_grad)

        for old_grad in old_grad_map_new_grads:
            for new_grad in old_grad_map_new_grads[old_grad]:
                new_weight = new_grad.grad_of
                old_weight = new_weight_map_old_weight[new_weight]
                new_grad.grad_of = old_weight
                old_weight._grad = old_grad_map_merged_grad[old_grad]
                merged_graph.tensors.remove(new_weight)
                merged_graph.in_tensors.remove(new_weight)
        for old_weight in weights:
            merged_graph.in_tensors.append(old_weight)
            merged_graph.tensors.append(old_weight)

        merged_graph.sanity_check()
        return merged_graph


class MicroBatchReplicatorPostProcess:
    OFFSET = 1000000000

    @classmethod
    def find_weights_grads(cls, graph: HybridGraph):
        weights_map_grads = dict()
        for tensor in graph.tensors:
            if tensor.op_type == PlaceHolder.type_name and tensor.require_grads:
                weights_map_grads[tensor] = tensor._grad
        return weights_map_grads

    @classmethod
    def replicate_micro_batches(cls, graph: HybridGraph, num_micro_batches):
        if num_micro_batches == 1:
            return graph
        # this is not accurate, but works.
        for tensor in graph.tensor_map_nodes.keys():
            nodes_this_tensor = graph.tensor_map_nodes[tensor]
            old_keys = list(nodes_this_tensor.keys())
            for mb in range(num_micro_batches):
                if mb == 0:
                    continue
                for key in old_keys:
                    old_node = nodes_this_tensor[key]
                    new_key = f"mb{mb}_{key}"
                    new_node = copy.deepcopy(old_node)
                    new_node.name = f"mb{mb}.{old_node.name}"
                    new_node.id = old_node.id + cls.OFFSET * mb
                    data_deps = old_node.data_deps
                    new_node.data_deps = list()
                    for data_dep in data_deps:
                        new_node.data_deps.append(data_dep + cls.OFFSET * mb)
                    ctrl_deps = old_node.ctrl_deps
                    new_node.ctrl_deps = list()
                    for ctrl_dep in ctrl_deps:
                        new_node.ctrl_deps.append(ctrl_dep + cls.OFFSET * mb)
                    nodes_this_tensor[new_key] = new_node
            graph.tensor_map_nodes[tensor] = nodes_this_tensor

    @classmethod
    def apply(cls, bundled_graph: BundledHybridGraph, num_micro_batches, inplace=True):
        assert inplace
        print("Replicating micro batches")
        for readable_rank in bundled_graph.graphs.keys():
            # print(f"Rank {readable_rank}")
            hybrid_graph = bundled_graph.graphs[readable_rank]
            cls.replicate_micro_batches(hybrid_graph, num_micro_batches)
        print("Replicate micro batches done")
        return bundled_graph
