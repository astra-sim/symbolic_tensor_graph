from .op_base import OPBase


class Element(OPBase):
    type_name = "E"

    @classmethod
    def _eval_impl(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr

        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden
        assert x2_shape is None
        assert x2_hidden is None

        amp = float(op_attr)
        num_ops = Tensor.eval_size(x1_shape)
        num_ops *= amp
        return x1_shape, x1_hidden, num_ops
