from ..tensor import Tensor

OPTIMIZE = True


class OPBase:
    _cache = dict()
    type_name = ""

    @classmethod
    def tokenrize(cls, tensor):
        op_attr = tensor.op_attr
        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden

        x1_shape_str = (
            Tensor.stringfy_shape(x1_shape) if x1_shape is not None else "None"
        )
        x2_shape_str = (
            Tensor.stringfy_shape(x2_shape) if x2_shape is not None else "None"
        )
        x1_hidden_str = (
            Tensor.stringfy_shape(x1_hidden) if x1_hidden is not None else "None"
        )
        x2_hidden_str = (
            Tensor.stringfy_shape(x2_hidden) if x2_hidden is not None else "None"
        )
        token = (
            cls.type_name
            + x1_shape_str
            + x2_shape_str
            + x1_hidden_str
            + x2_hidden_str
            + op_attr
        )
        return token

    @classmethod
    def eval(cls, tensor):
        token = cls.tokenrize(tensor)
        if token in cls._cache:
            return cls._cache[token]
        direct_output_shape, direct_output_hidden, num_ops = cls._eval_impl(tensor)
        cls._cache[token] = direct_output_shape, direct_output_hidden, num_ops
        return direct_output_shape, direct_output_hidden, num_ops

    @classmethod
    def _eval_impl(cls, tensor):
        raise NotImplementedError()
