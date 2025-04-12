import sympy as sp
from .op_base import OPBase
import json


class Customized(OPBase):
    type_name = "CUSTOM"

    @classmethod
    def _sanity_check(cls, tensor):
        assert tensor.op_attr is not None
        assert tensor.x1 is not None
        assert tensor.x2 is None
        assert tensor.x2_shape is not None
        assert tensor.x2_hidden is not None
        _ = sp.parse_expr(tensor.op_attr)

    @classmethod
    def _eval_impl(cls, tensor):
        y_shape = tensor.x2_shape
        y_hidden = tensor.x2_hidden
        num_ops = sp.parse_expr(tensor.op_attr)
        return y_shape, y_hidden, num_ops

    @classmethod
    def _shardable_options_impl(cls, tensor):
        raise NotImplementedError("Customized op is not shardable")
