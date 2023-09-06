import sympy as sp
import numpy as np
import pandas as pd
import graphviz

_Tensor_IO_parse_fn = dict()  # attr_name: parse_fn
_Tensor_IO_stringfy_fn = dict()


class _Tensor_IO:
    @staticmethod
    def register_parse_method(attr_name, parse_fn_):
        _Tensor_IO_parse_fn[attr_name] = parse_fn_

    @staticmethod
    def register_stringfy_method(attr_name, parse_fn_):
        _Tensor_IO_stringfy_fn[attr_name] = parse_fn_

    @staticmethod
    def parse_shape(shape):
        print("21", shape)
        shape = str(shape)
        ret = list()
        terms = shape.strip().split(",")
        for term in terms:
            ret.append(_Tensor_IO.parse_expr(term))
        return ret

    @staticmethod
    def stringfy_shape(shape):
        print("31", shape)
        ret = ""
        for term in shape:
            print(term)
            ret += str(term) + ","
        ret = ret[:-1]
        if len(ret) == 0:
            ret = "1"
        return ret

    @staticmethod
    def parse_expr(expr):
        return sp.parse_expr(expr.strip())

    @staticmethod
    def stringfy_expr(expr):
        return str(expr)

    @staticmethod
    def parse_str(term):
        return term.strip()

    @staticmethod
    def stringfy_str(term):
        return term.strip()

    @staticmethod
    def parse_bool(term):
        if isinstance(term, str):
            term = term.strip()
            return term == "Y" or term == "True" or term == "T"
        elif isinstance(term, int):
            return term > 0
        else:
            assert False

    @staticmethod
    def stringfy_bool(term):
        if term:
            return "Y"
        else:
            return "N"

    @staticmethod
    def parse_list(terms, term_parse_fn):
        terms = terms.strip().split(",")
        ret = list()
        for term in terms:
            ret.append(term_parse_fn(term))
        return ret

    @staticmethod
    def stringfy_list(terms, term_stringfy_fn):
        ret = ""
        for term in terms:
            ret += term_stringfy_fn(term) + ","
        return ret[:-1]

    @staticmethod
    def parse_fp(term):
        return float(term)

    @staticmethod
    def stringfy_fp(term):
        return str(term)

    register_parse_method("id_", parse_str)
    register_stringfy_method("id_", stringfy_str)
    register_parse_method("require_grads", parse_bool)
    register_stringfy_method("require_grads", stringfy_bool)
    register_parse_method("shape", parse_shape)
    register_stringfy_method("shape", stringfy_shape)
    register_parse_method("hidden", parse_shape)
    register_stringfy_method("hidden", stringfy_shape)
    register_parse_method("x1", parse_str)
    register_stringfy_method("x1", stringfy_str)
    register_parse_method("x2", parse_str)
    register_stringfy_method("x2", stringfy_str)
    register_parse_method("op_type", parse_str)
    register_stringfy_method("op_type", stringfy_str)
    register_parse_method("op_attr", parse_str)
    register_stringfy_method("op_attr", stringfy_str)
    register_parse_method("x1_shape", parse_shape)
    register_stringfy_method("x1_shape", stringfy_shape)
    register_parse_method("x1_hidden", parse_shape)
    register_stringfy_method("x1_hidden", stringfy_shape)
    register_parse_method("x2_shape", parse_shape)
    register_stringfy_method("x2_shape", stringfy_shape)
    register_parse_method("x2_hidden", parse_shape)
    register_stringfy_method("x2_hidden", stringfy_shape)
    register_parse_method("direct_output_shape", parse_shape)
    register_stringfy_method("direct_output_shape", stringfy_shape)
    register_parse_method("direct_output_hidden", parse_shape)
    register_stringfy_method("direct_output_hidden", stringfy_shape)
    register_parse_method("post_communications", parse_str)
    register_stringfy_method("post_communications", stringfy_str)
    register_parse_method("ops", parse_expr)
    register_stringfy_method("ops", stringfy_expr)
    register_parse_method("gradient_of", parse_str)
    register_stringfy_method("gradient_of", stringfy_str)
    register_parse_method("num_iter", parse_fp)
    register_stringfy_method("num_iter", stringfy_fp)
    register_parse_method("offloading", parse_fp)
    register_stringfy_method("offloading", stringfy_fp)

    @staticmethod
    def parse(key, value):
        if value is None or value == "":
            return None
        print(key)
        assert key in _Tensor_IO_parse_fn.keys()
        parse_fn = _Tensor_IO_parse_fn[key]
        return parse_fn(value)

    @staticmethod
    def stringfy(key, value):
        if value is None:
            return ""
        assert key in _Tensor_IO_stringfy_fn.keys()
        stringfy_fn = _Tensor_IO_stringfy_fn[key]
        return stringfy_fn(value)


class Tensor:
    def __init__(
        self,
        id_=None,
        require_grads=None,
        shape=None,
        hidden=None,
        x1=None,
        x2=None,
        op_type=None,
        op_attr=None,
        gradient_of=None,
        num_iter=None,
        offloading=None,
        **kwargs
    ):
        self.id_ = id_
        self.require_grads = require_grads
        self.shape = shape
        self.hidden = hidden
        self.x1 = x1
        self.x2 = x2
        self.op_type = op_type
        self.op_attr = op_attr
        self.gradient_of = gradient_of
        self.num_iter = num_iter
        self.offloading = offloading
        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

    @staticmethod
    def get_size(shape):
        size = 1
        for dimension in shape:
            size *= dimension
        return size

    @staticmethod
    def parse_record(terms, keys=None):
        if keys is None:
            keys = [
                "id_",
                "require_grads",
                "shape",
                "hidden",
                "x1",
                "x2",
                "op_type",
                "op_attr",
            ]
        tensor_attr = dict()
        for key, term in zip(keys, terms):
            tensor_attr[key] = _Tensor_IO.parse(key, term)
        tensor = Tensor(**tensor_attr)
        return tensor

    def to_record(tensor, keys=None):
        if keys is None:
            keys = list()
            for key in tensor.__dict__.keys():
                keys.append(key)
        terms = list()
        for key in keys:
            terms.append(_Tensor_IO.stringfy(key, tensor.__getattribute__(key)))
        return terms


if __name__ == "__main__":
    tensors = Tensor.parse_records(
        "sharding_spreadsheets/dp/graphs/multiHeadAttentionBwd.csv"
    )
    Tensor.visualize(tensors, "vis")
    Tensor.to_records(tensors, "resave.csv")

    tensors = Tensor.parse_records("resave.csv")
    Tensor.visualize(tensors, "revis")
    Tensor.to_records(tensors, "resave2.csv")

    tensor_dict = Tensor.get_tensor_dict(tensors)
    hook = 1
