from . import Tensor

def matmul(tensor_a, tensor_b, label=""):
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert tensor_a.dimensions_symbol[-1] == tensor_b.dimensions_symbol[-2]
    
    einsum = ""
    einsum_end = ""
    i = 0
    shrinked_dimension = chars[len(tensor_a.dimensions_symbol)-1]
    for symbol in tensor_a.dimensions_symbol:
        einsum += chars[i]
        if i != len(tensor_a.dimensions_symbol)-1:
            einsum_end += chars[i]
        i += 1
    einsum += ","
    for j, symbol in enumerate(tensor_b.dimensions_symbol):
        if j == len(tensor_b.dimensions_symbol)-2:
            einsum += shrinked_dimension
        else:
            einsum += chars[i]
            einsum_end += chars[i]
        i += 1
    einsum += "->" + einsum_end
    return Tensor.einsum(einsum, tensor_a, tensor_b, label)

def einsum(einsum_str, tensor_a, tensor_b, label=""):
    return Tensor.einsum(einsum_str, tensor_a, tensor_b, label)
    