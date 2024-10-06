from .op_base import OPBase


class Einsum(OPBase):
    type_name = "M"

    @classmethod
    def _sanity_check(cls, tensor):
        op_attr = tensor.op_attr
        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape
        x1_hidden = tensor.x1_hidden
        x2_hidden = tensor.x2_hidden

        assert len(x1_hidden)==1 and abs(float(x1_hidden[0])-1)<1e-9
        assert len(x2_hidden)==1 and abs(float(x2_hidden[0])-1)<1e-9

        terms = op_attr.split("->")
        assert len(terms) == 2
        terms = terms[0].split(",")
        assert len(terms) == 2
        x1_einsum, x2_einsum = terms[0], terms[1]
        assert len(x1_einsum) == len(x1_shape)
        assert len(x2_einsum) == len(x2_shape)

    @classmethod
    def _eval_impl(cls, tensor):
        op_attr = tensor.op_attr

        x1_shape = tensor.x1_shape
        x2_shape = tensor.x2_shape

        terms = op_attr.split("->")
        y_einsum = terms[1]
        terms = terms[0].split(",")
        x1_einsum, x2_einsum = terms[0], terms[1]

        einsum_letter_map_dim_symbol = dict()
        for char, symbol in zip(x1_einsum, x1_shape):
            if not char in einsum_letter_map_dim_symbol:
                einsum_letter_map_dim_symbol[char] = symbol
            else:
                assert einsum_letter_map_dim_symbol[char] == symbol
        for char, symbol in zip(x2_einsum, x2_shape):
            if not char in einsum_letter_map_dim_symbol:
                einsum_letter_map_dim_symbol[char] = symbol
            else:
                # if not einsum_letter_map_dim_symbol[char] == symbol:
                    # hook = 1
                assert einsum_letter_map_dim_symbol[char] == symbol
        y_shape = list()
        for char in y_einsum:
            assert char in einsum_letter_map_dim_symbol
            y_shape.append(einsum_letter_map_dim_symbol[char])
        reduced_dims = list()
        for char in x1_einsum:
            if not char in y_einsum:
                assert char in x2_einsum
                reduced_dims.append(char)
        y_hidden = list()
        for char in reduced_dims:
            y_hidden.append(einsum_letter_map_dim_symbol[char])

        num_ops = 1
        for dim in y_shape:
            num_ops *= dim
        for dim in y_hidden:
            num_ops *= dim
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
