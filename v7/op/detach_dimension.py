from tensor import Tensor
from op.op import OP
import sympy as sp

class DetachDimension(OP):
    def forward(self, tensor: Tensor, detach_dimension: sp.Expr, tensor_name: str):
        shape = list()
        for dimension in tensor.shape:
            if dimension == detach_dimension:
                name = dimension.name
                rep_id = 1
                if '%' in name:
                    terms = name.split('%')
                    assert len(terms) == 2
                    name = terms[0]
                    rep_id = int(terms[1])+1
                dimension = sp.Symbol(f"{dimension.name}%{rep_id}")
            shape.append(dimension)
        ret = Tensor(tensor_name, shape)
        self.context["input"] = tensor
        return ret

    def backward(self, grad):
        pass
