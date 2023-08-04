from tensor import Tensor, ShapedTensor, Symbol, Shape


def build_symbol_table():
    table = dict()
    table["B"] = Symbol("B")
    table["S"] = Symbol("S")
    table["H"] = Symbol("H")
    table["DIn"] = Symbol("DIn")
    table["DModel"] = Symbol("DModel")
    table["DFF"] = Symbol("DFF")
    table["DOut"] = Symbol("DOut")
    return table


def multi_head_attention(input_tensor, symbol_table, prefix=""):
    symbol_B = symbol_table["B"]
    symbol_S = symbol_table["S"]
    symbol_H = symbol_table["H"]
    symbol_D = symbol_table["DModel"]
    
    tensors = list()
    
    x = input_tensor
    
    wq = Tensor(prefix+"WQ", (symbol_H, symbol_D, symbol_D), require_grads=True)
    q = Tensor(prefix+"Q", (symbol_B, symbol_S, symbol_H, symbol_D))
    q.add_parent_product("bshd,hde->bshe", 
                         x.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)),
                         wq.as_shape((symbol_H, symbol_D, symbol_D)))
    tensors.append(wq)
    tensors.append(q)
    
    wk = Tensor(prefix+"WK", (symbol_H, symbol_D, symbol_D), require_grads=True)
    k = Tensor(prefix+"K", (symbol_B, symbol_S, symbol_H, symbol_D))
    k.add_parent_product("bshd,hde->bshe", 
                         x.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)),
                         wk.as_shape((symbol_H, symbol_D, symbol_D)))
    tensors.append(wk)
    tensors.append(k)
    
    wv = Tensor(prefix+"WV", (symbol_H, symbol_D, symbol_D), require_grads=True)
    v = Tensor(prefix+"V", (symbol_B, symbol_S, symbol_H, symbol_D))
    v.add_parent_product("bshd,hde->bshe", 
                         x.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)),
                         wv.as_shape((symbol_H, symbol_D, symbol_D)))
    tensors.append(wv)
    tensors.append(v)
    
    qk = Tensor(prefix+"QK", (symbol_B, symbol_S, symbol_S, symbol_H))
    qk.add_parent_product("bshd,bwhd->bswh",
                          q.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)),
                          k.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)))
    tensors.append(qk)
    
    qkv = Tensor(prefix+"QKV", (symbol_B, symbol_S, symbol_H, symbol_D))
    qkv.add_parent_product("bswh,bwhd->bshd",
                           qk.as_shape((symbol_B, symbol_S, symbol_S, symbol_H)),
                           v.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)))
    tensors.append(qkv)
    
    res = Tensor(prefix+"attRes", (symbol_B, symbol_S, symbol_H, symbol_D))
    res.add_parent_add(qkv.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)))
    res.add_parent_add(x.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)))
    tensors.append(res)
    
    norm = Tensor(prefix+"attNorm", (symbol_B, symbol_S, symbol_H, symbol_D))
    norm.add_parent_add(res.as_shape((symbol_B, symbol_S, symbol_H, symbol_D)))
    tensors.append(norm)
    
    return tensors, norm
    

def feed_forward_network(input_tensor, symbol_table, prefix=""):
    symbol_B = symbol_table["B"]
    symbol_S = symbol_table["S"]
    symbol_H = symbol_table["H"]
    symbol_DModel = symbol_table["DModel"]
    symbol_DFF = symbol_table["DFF"]
    
    tensors = list()
    
    x0 = input_tensor
    
    w1 = Tensor(prefix+"W1", (symbol_H, symbol_DModel, symbol_H, symbol_DFF), require_grads=True)
    x1 = Tensor(prefix+"X1", (symbol_B, symbol_S, symbol_H, symbol_DFF))
    x1.add_parent_product("bsd,de->bse",
                          x0.as_shape((symbol_B, symbol_S, (symbol_H, symbol_DModel))),
                          w1.as_shape(((symbol_H, symbol_DModel), (symbol_H, symbol_DFF))))
    tensors.append(w1)
    tensors.append(x1)
    
    w2 = Tensor(prefix+"W2", (symbol_H, symbol_DFF, symbol_H, symbol_DModel), require_grads=True)
    x2 = Tensor(prefix+"X2", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    x2.add_parent_product("bsd,de->bse",
                          x1.as_shape((symbol_B, symbol_S, (symbol_H, symbol_DFF))),
                          w2.as_shape(((symbol_H, symbol_DFF), (symbol_H, symbol_DModel))))
    tensors.append(w2)
    tensors.append(x2)
    
    res = Tensor(prefix+"ffnRes", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    res.add_parent_add(x2.as_shape((symbol_B, symbol_S, symbol_H, symbol_DModel)))
    res.add_parent_add(x0.as_shape((symbol_B, symbol_S, symbol_H, symbol_DModel)))
    tensors.append(res)
    
    norm = Tensor(prefix+"ffnNorm", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    norm.add_parent_add(res.as_shape((symbol_B, symbol_S, symbol_H, symbol_DModel)))
    tensors.append(norm)
    
    return tensors, norm


def transformer(num_stack, symbol_table):
    symbol_B = symbol_table["B"]
    symbol_S = symbol_table["S"]
    symbol_H = symbol_table["H"]
    symbol_DModel = symbol_table["DModel"]
    symbol_DIn = symbol_table["DIn"]
    symbol_DOut = symbol_table["DOut"]
    
    tensors = list()
    in_embed_x = Tensor("inEmbedX", (symbol_B, symbol_S, symbol_H, symbol_DIn))
    in_embed_w = Tensor("inEmbedW", (symbol_H, symbol_DIn, symbol_H, symbol_DModel), require_grads=True)
    in_embed_y = Tensor("inEmbedY", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    in_embed_y.add_parent_product("bsd,de->bse",
                                  in_embed_x.as_shape((symbol_B, symbol_S, (symbol_H, symbol_DIn))),
                                  in_embed_w.as_shape(((symbol_H, symbol_DIn), (symbol_H, symbol_DModel))))
    tensors.append(in_embed_x)
    tensors.append(in_embed_w)
    tensors.append(in_embed_y)
    
    input = in_embed_y
    for stack in range(num_stack):
        att_tensors, input = multi_head_attention(input, symbol_table, f"stack{stack}")
        ffn_tensors, input = feed_forward_network(input, symbol_table, f"stack{stack}")
        tensors.extend(att_tensors)
        tensors.extend(ffn_tensors)
    
    out_embed_x = input
    out_embed_w = Tensor("outEmbedW", (symbol_H, symbol_DModel, symbol_H, symbol_DOut), require_grads=True)
    out_embed_y = Tensor("outEmbedY", (symbol_B, symbol_S, symbol_H, symbol_DOut))
    out_embed_y.add_parent_product("bsd,de->bse",
                                   out_embed_x.as_shape((symbol_B, symbol_S, (symbol_H, symbol_DModel))),
                                   out_embed_w.as_shape(((symbol_H, symbol_DModel), (symbol_H, symbol_DOut))))
    tensors.append(out_embed_w)
    tensors.append(out_embed_y)
    
    return tensors, (in_embed_x, out_embed_y)


if __name__ == '__main__':
    symbol_table = build_symbol_table()
    fwd, (x, y) = transformer(2, symbol_table)
    Tensor.visualize_graph(fwd, "fwd")
    hook = 1
    bwd = Tensor.build_bwd_graph(fwd, y)
    Tensor.visualize_graph(bwd, "bwd")
    hook = 2
    