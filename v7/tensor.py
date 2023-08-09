import sympy as sp

class Tensor:
    def __init__(self, name, shape, require_grad=False):
        self.name = name
        self.shape = list()
        self.sharding = dict()
        for dimension in shape:
            assert isinstance(dimension, sp.Expr)
            self.shape.append(dimension)
            self.sharding[dimension] = 1
        self.context = list()
        self.gradient = None
        self.gradient_of = None
        self.require_grad = require_grad
        self._visited = False
        
    def create_gradient(self):
        gradient = Tensor("d_"+self.name, self.shape)
        self.gradient = gradient
        gradient.gradient_of = self
        return

    def parents(self):
        ret = list()
        for ctx in self.context:
            for parent in ctx.inputs:
                if not parent in ret:
                    ret.append(parent)
        return ret
    
    def backward(self):
        if self._visited:
            return
        self._visited = True
        assert self.gradient is not None
        for ctx in self.context:
            ctx.backward(self.gradient)
        for parent in self.parents():
            parent.backward()
        return

    ## TODO: please only use in for fwd graphs, then build bwd, finally gather all gradients
    @staticmethod
    def get_all_tensors(y, tensors=None):
        if tensors is None:
            tensors = list()
        tensors.append(y)
        for parent in y.parents():
            if not parent in tensors:
                Tensor.get_all_tensors(parent, tensors)
        return tensors

    def apply_sharding(self, shape_dimension, sharding_dimension):
        assert shape_dimension in self.shape
        self.sharding[shape_dimension] = sharding_dimension
        