from ..tensor import Tensor
from .op_base import OPBase


class Add(OPBase):
    type_name = "A"

    @classmethod
    def _eval_impl(cls, tensor):
        op_attr = tensor.op_attr
        assert op_attr is None

        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden

        assert Tensor.eval_size(x1_shape) == Tensor.eval_size(x2_shape)
        assert Tensor.eval_size(x1_hidden) == Tensor.eval_size(x2_hidden)

        y_shape = x1_shape
        y_hidden = x1_hidden
        num_ops = Tensor.eval_size(x1_shape)

        return y_shape, y_hidden, num_ops
