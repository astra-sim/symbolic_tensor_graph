from typing import Any
import sympy as sp


_Tensor_IO_load_fn = dict()
_Tensor_IO_save_fn = dict()


class _Tensor_IO:
    @staticmethod
    def register_load_save_method(attr_name, load_fn, save_fn):
        _Tensor_IO.register_load_method(attr_name, load_fn)
        _Tensor_IO.register_save_method(attr_name, save_fn)

    @staticmethod
    def register_load_method(attr_name, load_fn):
        _Tensor_IO_load_fn[attr_name] = load_fn

    @staticmethod
    def register_save_method(attr_name, save_fn):
        _Tensor_IO_save_fn[attr_name] = save_fn

    @staticmethod
    def load_list(list_str, term_load_fn):
        list_str = list_str.strip().replace("[", "").replace("]", "")
        terms = list_str.split(",")
        loaded_terms = list()
        for term in terms:
            loaded_terms.append(term_load_fn(term))
        return loaded_terms

    @staticmethod
    def save_list(terms, term_save_fn):
        list_str = "["
        for term in terms:
            list_str += term_save_fn(term) + ","
        if len(terms) > 0:
            list_str = list_str[:-1] + "]"
        else:
            list_str += "]"
        return list_str

    @staticmethod
    def load_expr(expr):
        return sp.parse_expr(expr.strip())

    @staticmethod
    def save_expr(expr):
        return str(expr)

    @staticmethod
    def load_str(term):
        return term.strip()

    @staticmethod
    def save_str(term):
        return term.strip()

    @staticmethod
    def load_bool(term):
        term = term.strip()
        if term == "Y" or term == "True" or term == "T":
            return True
        else:
            try:
                term_num = int(term)
                return term_num > 0
            except ValueError:
                pass
        return False

    @staticmethod
    def save_bool(term):
        return "Y" if term else "N"

    @staticmethod
    def load_fp(term):
        return float(term)

    @staticmethod
    def save_fp(term):
        return str(term)

    @staticmethod
    def load_int(term):
        return int(term)

    @staticmethod
    def save_int(term):
        return str(term)

    @staticmethod
    def load_shape(term):
        return _Tensor_IO.load_list(term, _Tensor_IO.load_expr)

    @staticmethod
    def save_shape(term):
        return _Tensor_IO.save_list(term, _Tensor_IO.save_expr)

    register_load_save_method("tensor_id", load_str, save_str)
    register_load_save_method("require_grads", load_bool, save_bool)
    register_load_save_method("shape", load_shape, save_shape)
    register_load_save_method("hidden", load_shape, save_shape)
    register_load_save_method("x1", load_str, save_str)
    register_load_save_method("x2", load_str, save_str)
    register_load_save_method("op_type", load_str, save_str)
    register_load_save_method("op_attr", load_str, save_str)
    register_load_save_method("direct_output_shape", load_str, save_str)
    register_load_save_method("post_communications", load_str, save_str)
    register_load_save_method("ops", load_expr, save_expr)
    register_load_save_method("gradient_of", load_str, save_str)
    register_load_save_method("num_iter", load_int, save_int)
    register_load_save_method("offloading", load_fp, save_fp)

    @staticmethod
    def load(key, value):
        return _Tensor_IO_load_fn[key](value)

    @staticmethod
    def save(key, value):
        return _Tensor_IO_save_fn[key](value)


class Tensor(dict):
    def __init__(self, *args, **kwargs):
        super(Tensor, self).__init__(*args, **kwargs)

    def check_signature(self, keys=None):
        if keys is None:
            keys = [
                "tensor_id",
                "tensor_name",
                "require_grads",
                "shape",
                "hidden",
                "x1_id",
                "x2_id",
                "op_type",
                "op_attr",
            ]
        for key in keys:
            assert key in self.keys()

    @staticmethod
    def load(terms, keys):
        tensor = Tensor()
        for key, term in zip(keys, terms):
            tensor[key] = term

    def save(self, keys):
        for key in keys:
            assert key in self.keys()
        terms = list()
        for key in keys:
            terms.append(_Tensor_IO.save(key, self.get[key]))
        return terms

    def __getattribute__(self, __name: str) -> Any:
        if __name in self.keys:
            return self.__getitem__[__name]
        return super().__getattribute__(__name)


def get_tensor_size(shape):
    size = 1
    for dim in shape:
        size *= dim
    return size
    