from .op_base import OPBase


class Element2(OPBase):
    type_name = "E2"

    @classmethod
    def _sanity_check(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr
        amp = float(op_attr)
        assert amp >= 0

        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden

        assert x1_shape == x2_shape
        assert abs(Tensor.eval_size(x1_hidden) - Tensor.eval_size(x2_hidden)) < 1e-9

    @classmethod
    def _eval_impl(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr

        x1_shape = tensor.x1_shape
        x1_hidden = tensor.x1_hidden

        amp = float(op_attr)
        y_shape = x1_shape
        y_hidden = x1_hidden
        num_ops = Tensor.eval_size(x1_shape)
        num_ops *= amp

        return y_shape, y_hidden, num_ops

    @classmethod
    def _shardable_options_impl(cls, tensor):
        x1_shape = tensor.x1_shape
        return list(range(0, len(x1_shape)))
