from .op_base import OPBase


class PlaceHolder(OPBase):
    type_name = "T"

    @classmethod
    def _eval_impl(cls, tensor):
        op_attr = tensor.op_attr
        assert op_attr is None

        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden
        assert x1_shape is None
        assert x1_hidden is None
        assert x2_shape is None
        assert x2_hidden is None

        return tensor.shape, tensor.hidden, 0
