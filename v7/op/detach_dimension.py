from tensor import Tensor
from op.op import OP
import sympy as sp

class DetachDimension(OP):
    cnt = 0
    
    @staticmethod
    def get_tensor_name():
        DetachDimension.cnt += 1
        return f"detach_{DetachDimension.cnt}"
    
    def forward(self, tensor: Tensor, detach_dimension: sp.Expr, detach_dimension_new: sp.Expr = None, ret: Tensor = None):
        self.inputs.append(tensor)
        shape = list()
        for dimension in tensor.shape:
            if dimension == detach_dimension:
                if detach_dimension_new is None:
                    detach_dimension_new = sp.Symbol(detach_dimension.name+"_copy")
                dimension = detach_dimension_new
            shape.append(dimension)
        if ret is None:
            ret = Tensor(DetachDimension.get_tensor_name(), shape)
        else:
            assert ret.shape == shape
        self.context["input"] = tensor
        self.context["old_dimension"] = detach_dimension
        self.context["new_dimension"] = detach_dimension_new
        return ret

    def backward(self, grad_y):
        # grad_x and grad_y has same shape, but different dimension name, change it back
        x = self.context["input"]
        detach_dimension_old = self.context["old_dimension"]
        detach_dimension_new = self.context["new_dimension"]
        assert detach_dimension_new in grad_y.shape
        if x.gradient is None:
            x.create_gradient()
        grad_x = x.grad
        DetachDimension.apply(tensor=grad_y, detach_dimension=detach_dimension_new, detach_dimension_new=detach_dimension_old, ret=grad_x)
        return grad_x
    
    def output_sharding(self):
        detach_dimension_old = self.context["old_dimension"]
        detach_dimension_new = self.context["new_dimension"]
        
        new_sharding = super(DetachDimension, self).output_sharding()

        assert detach_dimension_old in new_sharding
        detached_sharding = new_sharding[detach_dimension_old]
        del new_sharding[detach_dimension_old]
        
        new_sharding[detach_dimension_new] = detached_sharding
        return new_sharding
