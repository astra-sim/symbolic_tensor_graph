from .op_base import OPBase


class Identical(OPBase):
    type_name = "I"

    @classmethod
    def _eval_impl(cls, tensor):
        op_attr = tensor.op_attr
        assert op_attr is None

        x1_shape = tensor.x1_shape
        x1_hidden = tensor.x1_hidden
        x2_shape = tensor.x2_shape
        x2_hidden = tensor.x2_hidden

        assert not x1_shape is None
        assert not x1_hidden is None
        assert x2_shape is None
        assert x2_hidden is None

        return x1_shape, x1_hidden, 0
