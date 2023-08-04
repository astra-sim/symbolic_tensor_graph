from . import Tensor

def place_holder(dimensions_symbol, label, require_grads=False):
    ret = Tensor(dimensions_symbol, label, require_grads)
    ret.op = "place_holder"
    return ret
