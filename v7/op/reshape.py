from tensor import Tensor

def reshape(name: str, tensor: Tensor, new_shape):
    old_total_size = 1
    for dimension in tensor.shape:
        old_total_size *= dimension
    new_total_size = 1
    for dimension in new_shape:
        new_total_size *= dimension
    assert new_total_size == old_total_size
    
    ret = Tensor(name, new_shape)
    return ret
