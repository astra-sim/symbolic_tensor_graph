from .op_base import OPBase


class Reshape(OPBase):
    type_name = "R"

    @classmethod
    def _sanity_check(cls, tensor):
        from ..tensor import Tensor

        op_attr = tensor.op_attr
        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden
        assert op_attr is None

        assert Tensor.eval_size(x1_shape) == Tensor.eval_size(x2_shape)
        assert Tensor.eval_size(x1_hidden) == Tensor.eval_size(x2_hidden)

    @classmethod
    def _eval_impl(cls, tensor):
        from ..tensor import Tensor

        x2_shape = tensor.x2_shape
        x2_hidden = tensor.x2_hidden

        return x2_shape, x2_hidden, Tensor.eval_size(x2_shape)

    @classmethod
    def _shardable_options_impl(cls, tensor):
        syms = set()
        for dims in tensor.x1_shape:
            for sym in dims.free_symbols:
                syms.add(sym)
        return list(syms)
