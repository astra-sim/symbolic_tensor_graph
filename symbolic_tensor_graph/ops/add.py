from .op_base import OPBase


class Add(OPBase):
    type_name = "A"

    @classmethod
    def _sanity_check(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr
        assert op_attr is None

        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden

        assert x1_shape == x2_shape
        assert abs(Tensor.eval_size(x1_hidden) - Tensor.eval_size(x2_hidden)) < 1e-9

    @classmethod
    def _eval_impl(cls, tensor):
        from ..tensor import Tensor

        x1_shape = tensor.x1_shape
        x1_hidden = tensor.x1_hidden

        y_shape = x1_shape
        y_hidden = x1_hidden
        num_ops = Tensor.eval_size(x1_shape)

        return y_shape, y_hidden, num_ops

    @classmethod
    def _shardable_options_impl(cls, tensor):
        x1_shape = tensor.x1_shape
        return list(range(0, len(x1_shape)))
