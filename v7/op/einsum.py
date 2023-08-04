from tensor import Tensor

def einsum(name: str, einsum_str: str, tensor_1: Tensor, tensor_2: Tensor):
    terms = einsum_str.split("->")
    assert len(terms) == 2
    inputs_str, output_str = terms[0], terms[1]
    terms = inputs_str.split(",")
    assert len(terms) == 2
    input1_str, input2_str = terms[0], terms[1]
    
    assert len(input1_str) == len(tensor_1.shape)
    assert len(input2_str) == len(tensor_2.shape)
    char_symbol_table = dict()
    for char_, symbol in zip(input1_str, tensor_1.shape):
        if not char_ in char_symbol_table:
            assert not symbol in char_symbol_table.values()
        char_symbol_table[char_] = symbol
    for char_, symbol in zip(input2_str, tensor_2.shape):
        if not char_ in char_symbol_table:
            assert not symbol in char_symbol_table.values()
        char_symbol_table[char_] = symbol
    new_shape = list()
    for char_ in output_str:
        assert char_ in char_symbol_table
        new_shape.append(char_symbol_table[char_])
    ret = Tensor(name, new_shape)
    return ret
