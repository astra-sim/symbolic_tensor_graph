import sympy as sp
import copy
from .op_base import OPBase


class Slice(OPBase):
    type_name = "SLICE"

    @classmethod
    def _sanity_check(cls, tensor):
        from ..tensor import Tensor

        assert isinstance(tensor.op_attr, str)
        assert ":" in tensor.op_attr
        terms = tensor.op_attr.split(":")
        assert len(terms) == 2
        try:
            axis = int(terms[0])
            if axis < 0:
                axis += len(tensor.x1_shape)
            assert axis <= len(tensor.x1_shape)
        except ValueError:
            assert False

    @classmethod
    def _eval_impl(cls, tensor):
        from ..tensor import Tensor

        terms = tensor.op_attr.split(":")
        axis = int(terms[0])
        slice_size = sp.parse_expr(terms[1])
        new_shape = copy.deepcopy(tensor.x1_shape)
        new_shape[axis] = slice_size

        num_ops = Tensor.eval_size(new_shape)

        return new_shape, tensor.x1_hidden, num_ops

    @classmethod
    def _shardable_options_impl(cls, tensor):
        x1_shape = tensor.x1_shape
        return list(range(0, len(x1_shape)))
