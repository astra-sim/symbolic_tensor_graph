from tensor import Tensor
from op.op import OP
import sympy as sp

class ElementWise(OP):
    cnt = 0
    
    @staticmethod
    def get_tensor_name():
        ElementWise.cnt += 1
        return f"element_wise_{ElementWise.cnt}"
    
    def forward(self, x: Tensor, ret: Tensor = None):
        self.inputs.append(x)
        if ret is None:
            ret = Tensor(ElementWise.get_tensor_name(), x.shape)
        else:
            assert ret.shape == x.shape
        self.context["input"] = x
        return ret
    
    def backward(self, grad_y):
        x = self.context["input"]
        if x.gradient is None:
            x.create_gradient()
        grad_x = x.grad
        ElementWise.apply(x=grad_y, ret=grad_x)
        return grad_x

    def get_ops(self):
        x = self.context["input"]
        sharding_map = self.output_sharding()
        
        ops = 1
        for shape in x.shape:
            for symbol in shape.free_symbols:
                ops = ops * symbol / sharding_map[symbol]
        return ops