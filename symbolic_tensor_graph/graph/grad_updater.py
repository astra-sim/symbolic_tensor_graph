import copy
from ..ops import Add
from ..tensor import Tensor


class GradUpdater:
    @classmethod
    def _default_revision_fn(cls, old_replicate):
        return str(int(old_replicate) + 1)

    @classmethod
    def _update_grad(cls, tensor, grad, new_revision_fn):
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
        elif isinstance(new_revision, callable):
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
