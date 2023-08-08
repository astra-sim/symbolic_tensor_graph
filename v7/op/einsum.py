from tensor import Tensor
from op.op import OP
import sympy as sp

class EinSum(OP):
    cnt = 0
    
    @staticmethod
    def get_tensor_name():
        EinSum.cnt += 1
        return f"einsum_{EinSum.cnt}"
    
    def forward(self, x1: Tensor, x2: Tensor, einsum_str: str, ret: Tensor = None):
        self.inputs.append(x1)
        self.inputs.append(x2)
        terms = einsum_str.split("->")
        assert len(terms) == 2
        inputs_str, output_str = terms[0], terms[1]
        terms = inputs_str.split(",")
        assert len(terms) == 2
        input1_str, input2_str = terms[0], terms[1]
    
        assert len(input1_str) == len(x1.shape)
        assert len(input2_str) == len(x2.shape)
        char_symbol_table = dict()
        for char_, symbol in zip(input1_str, x1.shape):
            if not char_ in char_symbol_table:
                assert not symbol in char_symbol_table.values()
            char_symbol_table[char_] = symbol
        for char_, symbol in zip(input2_str, x2.shape):
            if not char_ in char_symbol_table:
                assert not symbol in char_symbol_table.values()
            char_symbol_table[char_] = symbol
        new_shape = list()
        for char_ in output_str:
            assert char_ in char_symbol_table
            new_shape.append(char_symbol_table[char_])
        reduced_shape = list()
        for symbol in char_symbol_table.values():
            if not symbol in new_shape:
                reduced_shape.append(symbol)
        if ret is None:
            ret = Tensor(EinSum.get_tensor_name(), new_shape)
        else:
            assert ret.shape == new_shape
        self.context["x1"] = x1
        self.context["x2"] = x2
        self.context["einsum_str"] = einsum_str
        self.context["y_shape"] = new_shape
        self.context["reduced_shape"] = reduced_shape
        return ret
        
    def backward(self, grad_y):
        x1 = self.context["x1"]
        x2 = self.context["x2"]
        fwd_einsum_str = self.context["einsum_str"]
        
        terms = fwd_einsum_str.split("->")
        assert len(terms) == 2
        inputs_str, output_str = terms[0], terms[1]
        terms = inputs_str.split(",")
        assert len(terms) == 2
        input1_str, input2_str = terms[0], terms[1]
        
        if x1.gradient is None:
            x1.create_gradient()
        if x2.gradient is None:
            x2.create_gradient()
        grad_x1 = x1.grad
        grad_x2 = x2.grad
        
        x1_einsum_str = f"{output_str},{input2_str}->{input1_str}"
        x2_einsum_str = f"{output_str},{input1_str}->{input2_str}"
        EinSum.apply(grad_y, x2, x1_einsum_str, grad_x1)
        EinSum.apply(grad_y, x1, x2_einsum_str, grad_x2)
        return grad_x1, grad_x2

    def output_sharding(self):
        x1 = self.context["x1"]
        x2 = self.context["x2"]
        y_shape = self.context["y_shape"]
        reduced_shape = self.context["reduced_shape"]
        
        shape_sharding_table = dict()
        for shape_symbol, sharding_symbol in zip(x1.shape, x1.sharding):
            if shape_symbol in shape_sharding_table:
                assert shape_sharding_table[shape_symbol] == sharding_symbol
            else:
                shape_sharding_table[shape_symbol] = sharding_symbol
        for shape_symbol, sharding_symbol in zip(x2.shape, x2.sharding):
            if shape_symbol in shape_sharding_table:
                assert shape_sharding_table[shape_symbol] == sharding_symbol
            else:
                shape_sharding_table[shape_symbol] = sharding_symbol
        sharding = list()
        for shape in y_shape:
            assert shape in shape_sharding_table.keys()
            sharding.append(shape_sharding_table[shape])
        assert len(reduced_shape) <= 1
        for reduced in reduced_shape:
            assert reduced in shape_sharding_table.keys()
            sharding.append(reduced)
        return sharding
        
        