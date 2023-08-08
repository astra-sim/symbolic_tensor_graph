from tensor import Tensor
from op.detach_dimension import DetachDimension
from op.einsum import EinSum
from op.element_wise import ElementWise
from op.reshape import Reshape
import sympy as sp


def multi_head_attention(x, prefix):
    B, H, S, DModel, DModel1, W = sp.symbols("B H S DModel DModel1 W")
    assert x.shape == [B, S, H, DModel]
    wq = Tensor(f"{prefix}_wq", (H, DModel, DModel1), require_grad=True)
    wk = Tensor(f"{prefix}_wk", (H, DModel, DModel1), require_grad=True)
    wv = Tensor(f"{prefix}_wv", (H, DModel, DModel1), require_grad=True)
    q = EinSum.apply(wq, x, "hde,bshd->bshe")
    k = DetachDimension.apply(EinSum.apply(wk, x, "hde,bshd->bshe"), S, W)
    v = DetachDimension.apply(EinSum.apply(wv, x, "hde,bshd->bshe"), S, W)
    qk = EinSum.apply(q, k, "bshd,bwhd->bswh")
    qkv = DetachDimension.apply(EinSum.apply(qk, v, "bswh,bwhd->bshd"), DModel1, DModel)
    res = ElementWise.apply(qkv)
    ElementWise.apply(x, ret=res)
    norm = ElementWise.apply(res)
    return norm


def feed_forward_network(x, prefix):
    B, H, S, DModel, DFF = sp.symbols("B H S DModel DFF")
    assert x.shape == [B, S, H, DModel]
    x0 = Reshape.apply(x, (B, S, H*DModel))
    w1 = Tensor(f"{prefix}_w1", (H*DModel, H*DFF), require_grad=True)
    w2 = Tensor(f"{prefix}_w2", (H*DFF, H*DModel), require_grad=True)
    x1 = EinSum.apply(w1, x0, "de,bsd->bse")
    x2 = EinSum.apply(w2, x1, "de,bsd->bse")
    x2_reshape = Reshape.apply(x2, (B, S, H, DModel))
    res = ElementWise.apply(x2_reshape)
    ElementWise.apply(x, ret=res)
    norm = ElementWise.apply(res)
    return norm


def transformer(x0, num_stack):
    B, H, S, DIn, DModel, DOut = sp.symbols("B H S DIn DModel DOut")
    # x0 = Tensor("input", (B, S, H*DIn))
    assert x0.shape == [B, S, H*DIn]
    w1 = Tensor("inEmbW", (H*DIn, H*DModel), require_grad=True)
    x = EinSum.apply(x0, w1, "bsd,de->bse")
    x = Reshape.apply(x, (B, S, H, DModel))
    for i in range(num_stack):
        x = multi_head_attention(x, f"stack{i}")
        x = feed_forward_network(x, f"stack{i}")
    x = Reshape.apply(x, (B, S, H*DModel))
    w2 = Tensor("outEmbW", (H*DModel, H*DOut), require_grad=True)
    y = EinSum.apply(x, w2, "bsd,de->bse")
    return y


if __name__ == '__main__':
    B, H, S, DIn = sp.symbols("B H S DIn")
    x = Tensor("inEmbX", (B, S, H*DIn))
    y = transformer(x, 2)
    hook = 0
    y.create_gradient()
    y.backward()
    hook = 1
    