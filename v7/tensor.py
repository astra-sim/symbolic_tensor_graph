import sympy as sp

class Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = list()
        for dimension in shape:
            assert isinstance(dimension, sp.Expr)
            self.shape.append(dimension)
        
        