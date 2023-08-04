from tensor import Tensor

def element_wise(name: str, tensor: Tensor):
    ret = Tensor(name, tensor.shape)
    return ret
    