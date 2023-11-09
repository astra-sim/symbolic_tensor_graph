from .op_base import OPBase


class Shadow(OPBase):
    type_name = "S"

    @classmethod
    def _sanity_check(cls, tensor):
        assert tensor.x1_shape is not None
        assert tensor.x1_hidden is not None
        assert tensor.x2_shape is None
        assert tensor.x2_hidden is None

    @classmethod
    def _eval_impl(cls, tensor):
        return tensor.x1_shape, tensor.x1_hidden, 0

    @classmethod
    def _shardable_options_impl(cls, tensor):
        raise NotImplementedError()
