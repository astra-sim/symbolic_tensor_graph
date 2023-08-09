from tensor import Tensor
from op.op import OP
import sympy as sp

class Reshape(OP):
    cnt = 0
    
    @staticmethod
    def get_tensor_name():
        Reshape.cnt += 1
        return f"reshape_{Reshape.cnt}"
    
    def forward(self, x: Tensor, new_shape, ret: Tensor = None):
        self.inputs.append(x)
        old_total_size = 1
        for dimension in x.shape:
            old_total_size *= dimension
        new_total_size = 1
        for dimension in new_shape:
            new_total_size *= dimension
        assert new_total_size == old_total_size
        
        if ret is None:
            ret = Tensor(Reshape.get_tensor_name(), new_shape)
        else:
            assert ret.shape == new_shape
        self.context["input"] = x
        return ret
    
    def backward(self, grad_y):
        x = self.context["input"]
        if x.gradient is None:
            x.create_gradient()
        grad_x = x.grad
        Reshape.apply(grad_y, x.shape, ret=grad_x)
        return grad_x


    def get_ops(self):
        x = self.context["input"]
        sharding_map = self.output_sharding()
        
        ops = 1
        for shape in x.shape:
            for symbol in shape.free_symbols:
                ops = ops * symbol / sharding_map[symbol]
        return ops
    