import sympy as sp
import numpy as np
import pandas as pd
import graphviz
import copy
from .ops.op_handler import OPHandler

OPTIMIZE = True


class Tensor:
    _parsed_expr_cache = dict()
    _eval_expr_cache = list()
    _stringfy_expr_cache = dict()
    CSV_HEADER = [
        "id",
        "require_grads",
        "x1",
        "x2",
        "op_type",
        "op_attr",
        "x1_shape",
        "x1_hidden",
        "x2_shape",
        "x2_hidden",
        "grad_of",
    ]

    def __init__(self, create_empty=False):
        if not create_empty:
            assert False  # not allow create empty tensor, need to parse from file,
            # here we impl something like private constructor
        self.name = None
        self.require_grads = None
        self.x1 = None
        self.x2 = None
        self.op_type = None
        self.op_attr = None
        self.x1_shape = None
        self.x1_hidden = None
        self.x2_shape = None
        self.x2_hiden = None
        self.grad_of = None
        self._grad = None

        self.revision = None

        self._op_token = None
        self._op_results = None

    @staticmethod
    def parse_shape(shape):
        if shape is None:
            return None
        shape = str(shape)
        ret = list()
        terms = shape.strip().split(",")
        for term in terms:
            ret.append(Tensor.parse_expr(term))
        return ret

    @staticmethod
    def stringfy_shape(shape):
        ret = ""
        for term in shape:
            ret += Tensor.stringfy_expr(term) + ","
        ret = ret[:-1]
        if len(ret) == 0:
            ret = "1"
        return ret

    @staticmethod
    def parse_expr(expr):
        if not OPTIMIZE:
            return sp.parse_expr(expr)
        expr = expr.strip()
        if not expr in Tensor._parsed_expr_cache:
            Tensor._parsed_expr_cache[expr] = sp.parse_expr(expr)
        return copy.deepcopy(Tensor._parsed_expr_cache[expr])

    @staticmethod
    def stringfy_expr(expr):
        if not OPTIMIZE:
            return str(expr)
        if not expr in Tensor._stringfy_expr_cache:
            Tensor._stringfy_expr_cache[expr] = str(expr)
        return Tensor._stringfy_expr_cache[expr]

    @staticmethod
    def eval_expr(expr, target_symbol_value_dict):
        if not OPTIMIZE:
            return expr.evalf(subs=target_symbol_value_dict)
        target_eval_expr_cache = None
        for (
            target_symbol_value_dict_,
            target_eval_expr_cache_,
        ) in Tensor._eval_expr_cache:
            if target_symbol_value_dict == target_symbol_value_dict_:
                target_eval_expr_cache = target_eval_expr_cache_
        if target_eval_expr_cache is None:
            target_eval_expr_cache = dict()
            Tensor._eval_expr_cache.append(
                (copy.deepcopy(target_symbol_value_dict), target_eval_expr_cache)
            )

        if not expr in target_eval_expr_cache:
            target_eval_expr_cache[expr] = expr.evalf(subs=target_symbol_value_dict)
        return target_eval_expr_cache[expr]

    @staticmethod
    def eval_size(shape):
        size = 1
        for dim in shape:
            size *= dim
        return size

    @staticmethod
    def parse_id(id_):
        terms = id_.split("@")
        if len(terms) == 1:
            tensor_name = terms[0]
            tensor_revision = 0
        elif len(terms) == 2:
            tensor_name = terms[0]
            tensor_revision = int(terms[1])
        else:
            assert False
        return tensor_name, tensor_revision

    @staticmethod
    def stringfy_id(tensor_name, tensor_revision):
        return f"{tensor_name}@{tensor_revision}"

    @property
    def id(self):
        return Tensor.stringfy_id(self.name, self.revision)

    @property
    def y_shape(self):
        token = OPHandler.tokenrize(self)
        if not token == self._op_token:
            self._op_token = token
            self._op_results = OPHandler.eval(self)
        return self._op_results[0]

    @property
    def y_hidden(self):
        token = OPHandler.tokenrize(self)
        if not token == self._op_token:
            self._op_token = token
            self._op_results = OPHandler.eval(self)
        return self._op_results[1]

    @property
    def ops(self):
        token = OPHandler.tokenrize(self)
        if not token == self._op_token:
            self._op_token = token
            self._op_results = OPHandler.eval(self)
        return self._op_results[2]

    @staticmethod
    def _parse_record(terms):
        assert len(terms) == len(Tensor.CSV_HEADER)
        tensor = Tensor(create_empty=True)
        tensor_name, tensor_revision = Tensor.parse_id(terms[0])
        tensor.name = tensor_name
        tensor.revision = tensor_revision

        tensor.require_grads = terms[1].strip() == "Y"

        if not terms[2] is None:
            x1_name, x1_revision = Tensor.parse_id(terms[2])
            tensor.x1 = Tensor.stringfy_id(x1_name, x1_revision)
        else:
            tensor.x1 = None

        if not terms[3] is None:
            x2_name, x2_revision = Tensor.parse_id(terms[3])
            tensor.x2 = Tensor.stringfy_id(x2_name, x2_revision)
        else:
            tensor.x2 = None

        tensor.op_type = terms[4]
        tensor.op_attr = terms[5]
        tensor.x1_shape = Tensor.parse_shape(terms[6])
        tensor.x1_hidden = Tensor.parse_shape(terms[7])
        tensor.x2_shape = Tensor.parse_shape(terms[8])
        tensor.x2_hidden = Tensor.parse_shape(terms[9])

        if not terms[10] is None:
            grad_of_name, grad_of_revision = Tensor.parse_id(terms[10])
            tensor.grad_of = Tensor.stringfy_id(grad_of_name, grad_of_revision)
        else:
            tensor.grad_of = None

        return tensor

    def _to_record(tensor):
        terms = list()
        terms.append(tensor.id)
        terms.append("Y" if tensor.require_grads else "N")
        terms.append(tensor.x1.id if not tensor.x1 is None else "")
        terms.append(tensor.x2.id if not tensor.x2 is None else "")
        terms.append(tensor.op_type)
        terms.append(tensor.op_attr)
        terms.append(
            Tensor.stringfy_shape(tensor.x1_shape)
            if not tensor.x1_shape is None
            else ""
        )
        terms.append(
            Tensor.stringfy_shape(tensor.x1_hidden)
            if not tensor.x1_hidden is None
            else ""
        )
        terms.append(
            Tensor.stringfy_shape(tensor.x2_shape)
            if not tensor.x2_shape is None
            else ""
        )
        terms.append(
            Tensor.stringfy_shape(tensor.x2_hidden)
            if not tensor.x2_hidden is None
            else ""
        )
        terms.append(tensor.grad_of.id if not tensor.grad_of is None else "")
        return terms

    @staticmethod
    def parse_records(csv_filename):
        df = pd.read_csv(csv_filename, encoding="utf-8")
        df = df.replace({np.nan: None})
        assert list(df.columns) == Tensor.CSV_HEADER
        tensors = list()
        for i in range(df.shape[0]):
            data = np.array(df[i : i + 1]).reshape(-1)
            tensors.append(Tensor._parse_record(data))

        tensor_id_map_tensor = dict()
        for tensor in tensors:
            key = tensor.id
            assert not key in tensor_id_map_tensor
            tensor_id_map_tensor[key] = tensor

        for tensor in tensors:
            if tensor.x1 is not None:
                tensor.x1 = tensor_id_map_tensor[tensor.x1]
            if tensor.x2 is not None:
                tensor.x2 = tensor_id_map_tensor[tensor.x2]
            if tensor.grad_of is not None:
                assert tensor.grad_of in tensor_id_map_tensor
                tensor.grad_of = tensor_id_map_tensor[tensor.grad_of]
                tensor.grad_of._grad = tensor
        return tensors

    @staticmethod
    def to_records(tensors, csv_filename):
        data = list()
        for tensor in tensors:
            data.append(tensor._to_record())
        df = pd.DataFrame(data)
        assert len(df[0]) == len(df[0].unique())
        df.to_csv(csv_filename, encoding="utf-8", header=Tensor.CSV_HEADER, index=None)

    @staticmethod
    def visualize(tensors, filename, format="pdf"):
        f = graphviz.Digraph()
        for tensor in tensors:
            f.node(name=tensor.id, lable=tensor.id, id=tensor.id, shape="box")
            if tensor.x1 is not None:
                f.edge(tensor.x1.id, tensor.id)
            if tensor.x2 is not None:
                f.edge(tensor.x2.id, tensor.id)
        f.render(filename, format=format, cleanup=True)

    # def __eq__(one, another):
    #     return one._to_record() == another._to_record()

    def __repr__(self):
        x1_id = self.x1.id if not self.x1 is None else "None"
        x2_id = self.x2.id if not self.x2 is None else "None"
        grad_id = self._grad.id if not self._grad is None else "None"
        return f"{self.id}: {x1_id}[{self.x1_shape}@{self.x1_hidden}] {x2_id}[{self.x2_shape}@{self.x2_hidden}] grad={grad_id}"

    def __str__(self):
        return self.__repr__()
