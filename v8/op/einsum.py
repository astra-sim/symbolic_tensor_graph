from op.op import OP
from tensor import Tensor

class EinSum(OP):
    def __init__(self, einsum_str):
        terms = einsum_str.split("->")
        assert len(terms) == 2
        xs_str, y_str = terms[0], terms[1]
        terms = xs_str.split(",")
        assert len(terms) == 2
        x1_str, x2_str = terms[0], terms[1]
        
        self.y_str = y_str
        self.x1_str = x1_str
        self.x2_str = x2_str
    
    def _parse_char_symbol_table(self, x1_shape, x2_shape):
        assert len(self.x1_str) == len(x1_shape)
        assert len(self.x2_str) == len(x2_shape)
        char_symbol_table = dict()
        for char_, symbol in zip(self.x1_str, x1_shape):
            if not char_ in char_symbol_table:
                assert not symbol in char_symbol_table.value()
            char_symbol_table[char_] = symbol
        for char_, symbol in zip(self.x2_str, x2_shape):
            if not char_ in char_symbol_table:
                assert not symbol in char_symbol_table.value()
            char_symbol_table[char_] = symbol
        return char_symbol_table

    def _get_output_tensor(self):
        y_shape = list()
        y_reduced = list()
        ops = 1
        
        for char_ in self.y_str:
            y_shape.append(self.char_symbol_table[char_])
            ops = ops * self.char_symbol_table[char_]
        
        for char_ in self.char_symbol_table.keys():
            if not char_ in self.y_str:
                y_reduced.append(self.char_symbol_table[char_])
                ops = ops * self.char_symbol_table[char_]
                
        return Tensor(y_shape, y_reduced, ops=ops)
    
    def _x1_einsum_op(self):
        return EinSum(f"{self.y_str},{self.x2_str}->{self.x1_str}")
    
    def _x2_einsum_op(self):
        return EinSum(f"{self.y_str},{self.x1_str}->{self.x2_str}")
    
    def forward(self, x1, x2):
        self.char_symbol_table = self._parse_char_symbol_table(x1.shape, x2.shape)
        y = self._get_output_tensor()
        y.parents.append(x1)
        y.parents.append(x2)
        return y
    
    def backward(self, fwd_outputs, fwd_inputs):
        y = fwd_outputs[0]
        x1, x2 = fwd_inputs
        
        
        