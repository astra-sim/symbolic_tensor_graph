from .op_base import OPBase


class Customize(OPBase):
    type_name = "CUSTOM"
    raise NotImplementedError()

    @classmethod
    def _sanity_check(cls, tensor):

        op_attr = tensor.op_attr

    @classmethod
    def _eval_impl(cls, tensor):
        op_attr = tensor.op_attr

        return y_shape, y_hidden, num_ops

    @classmethod
    def _shardable_options_impl(cls, tensor):
        op_attr = tensor.op_attr
        terms = op_attr.split("->")[0].split(",")
        charset = set()
        for char in terms[0]:
            charset.add(char)
        for char in terms[1]:
            charset.add(char)
        return list(charset)
