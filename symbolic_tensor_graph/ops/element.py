from .op_base import OPBase


class Element(OPBase):
    type_name = "E"

    @classmethod
    def _sanity_check(cls, tensor):
        op_attr = tensor.op_attr
        x2_shape = tensor.x2_shape
        x2_hidden = tensor.x2_hidden
        assert x2_shape is None
        assert x2_hidden is None
        amp = float(op_attr)
        assert amp >= 0

    @classmethod
    def _eval_impl(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr

        x1_shape = tensor.x1_shape
        x1_hidden = tensor.x1_hidden

        amp = float(op_attr)
        num_ops = Tensor.eval_size(x1_shape)
        num_ops *= amp
        return x1_shape, x1_hidden, num_ops

    @classmethod
    def _shardable_options_impl(cls, tensor):
        x1_shape = tensor.x1_shape
        return list(range(0, len(x1_shape)))
