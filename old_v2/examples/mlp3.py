import sys
sys.path.append("../tensor_network")

from ops import matmul, place_holder
from tensor import Tensor

def mlp3_generator(num_layers):
    tensors = list()
    last_d = Tensor.get_new_dimension_symbol("d")
    x = place_holder(("batchsize", last_d), "x")
    tensors.append(x)
    for i in range(num_layers):
        new_d = Tensor.get_new_dimension_symbol("d")
        weight = place_holder(
            (last_d, new_d), f"w{i}", require_grads=True)
        tensors.append(weight)
        x = matmul(x, weight, label=f"x{i}")
        tensors.append(x)
        x = Tensor.elementWise(x, label=f"x{i}act")
        tensors.append(x)
        last_d = new_d
        new_d = None
    return tensors


if __name__ == '__main__':
    mlp3 = mlp3_generator(5)
    for tensor in mlp3:
        print(tensor)
        