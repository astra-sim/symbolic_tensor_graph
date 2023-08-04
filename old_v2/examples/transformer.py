import sys
sys.path.append("../tensor_network")

from ops import matmul, place_holder, einsum
from tensor import Tensor

def multihead_attention(x, label=""):
    if not label=="":
        label += "_"
    assert len(x.dimensions_symbol) == 4    # B S H N
    batch, seq, head, din = x.dimensions_symbol
    dqk = Tensor.get_new_dimension_symbol(f"{label}dqk")
    dv = Tensor.get_new_dimension_symbol(f"{label}dv")
    tensors = list()
    
    wq = place_holder((head, dfeat, dqk), f"{label}wq", require_grads=True)
    wk = place_holder((head, dfeat, dqk), f"{label}wk", require_grads=True)
    wv = place_holder((head, dfeat, dv), f"{label}wv", require_grads=True)
    q = einsum("bshd,hdn->bshn", x, wq, f"{label}q")
    k = einsum("bshd,hdn->bshn", x, wk, f"{label}k")
    v = einsum("bshd,hdn->bshn", x, wv, f"{label}v")
    qk = einsum("bqhn,bkhn->bqkh", q, k, f"{label}qk")
    softmax_qk = Tensor.elementWise(qk, f"{label}sqk")
    qkv = einsum("bskh,bkhn->bshn", softmax_qk, v, f"{label}qkv")
    tensors.extend([wq, wk, wv, q, k, v, qk, softmax_qk, qkv])
    return qkv, tensors
    
def transformer_ffn(x, label=""):
    if not label=="":
        label += "_"
    assert len(x.dimensions_symbol) == 3    # B S N
    batch, seq, din = x.dimensions_symbol
    d1 = Tensor.get_new_dimension_symbol(f"{label}dfn1")
    d2 = Tensor.get_new_dimension_symbol(f"{label}dfn1")
    tensors = list()
    
    w1 = place_holder((din, d1), f"{label}w1", require_grads=True)
    w2 = place_holder((d1, d2), f"{label}w2", require_grads=True)
    x1 = einsum("bsn,nm->bsm", x, w1)
    x2 = einsum("bsn,nm->bsm", x1, w2)
    tensors.append([w1, w2, x1, x2])
    return x2, tensors

def transformer_stack(x):
    
    