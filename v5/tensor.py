import copy, graphviz

from dimension import Dimension

class Tensor:
    def __init__(self, name, dimensions, require_grads=False):
        self.name = name
        self.dimensions = list()
        for dimension in dimensions:
            assert isinstance(dimension, Dimension)
            self.dimensions.append(dimension.copy())

        self.parent_nomials = list()
        self.require_grads = require_grads
        self.gradient = None
        self.gradient_of = None
        
        self._visited = False
        
    def create_gradient(self):
        assert self.gradient is None
        gradient = Tensor("d_"+self.name, self.dimensions)
        self.gradient = gradient
        gradient.gradient_of = self
        return gradient
    
    def transfer_gradient(self, require_grads_tensors):
        assert self.gradient is not None
        for nomial in self.parent_nomials:
            if len(nomial) == 1:
                parent = nomial[0]
                if (parent.gradient is None) and (parent in require_grads_tensors):
                    parent.create_gradient()
                    require_grads_tensors.remove(parent)
                    print(len(require_grads_tensors))
                if parent.gradient is not None:
                    parent.gradient.add_parent_add(self.gradient)
            elif len(nomial) == 2:
                p1, p2 = nomial
                if (p1.gradient is None) and (p1 in require_grads_tensors):
                    p1.create_gradient()
                    require_grads_tensors.remove(p1)
                    print(len(require_grads_tensors))
                if p1.gradient is not None:
                    p1.gradient.add_parent_product(self.gradient, p2)
                    
                if (p2.gradient is None) and (p2 in require_grads_tensors):
                    p2.create_gradient()
                    require_grads_tensors.remove(p2)
                    print(len(require_grads_tensors))
                if p2.gradient is not None:
                    p2.gradient.add_parent_product(self.gradient, p1)
            else:
                raise NotImplementedError("For now only product of two are supported")

        for nomial in self.parent_nomials:
            for parent in nomial:
                if (not parent._visited) and (parent.gradient is not None):
                    parent._visited = True
                    parent.transfer_gradient(require_grads_tensors)
        return
    
    def add_parent_product(self, p1, p2):
        self.parent_nomials.append((p1, p2))
        
    def add_parent_add(self, p1):
        self.parent_nomials.append((p1,))
        
    @staticmethod
    def clean_bwd_graph(tensors):
        filtered_graph = list()
        for tensor in tensors:
            if tensor.gradient_of is None:
                tensor.gradient = None
                filtered_graph.append(tensor)
        return filtered_graph
    
    @staticmethod
    def build_backward_comp_graph(tensors, output_tensor):
        # make sure current input tensors are only fwd graph, no gradient tensors
        tensors = Tensor.clean_bwd_graph(tensors)
        assert isinstance(output_tensor, Tensor)
        for tensor in tensors:
            tensor._visited = False
        
        require_grads_tensors = Tensor.get_require_grads_tensors(tensors)
        assert output_tensor in require_grads_tensors
        output_tensor.create_gradient()
        require_grads_tensors.remove(output_tensor)
        output_tensor.transfer_gradient(require_grads_tensors)
        output_tensor.gradient.add_parent_add(output_tensor)
        
        gradient_tensors = list()
        for tensor in tensors:
            if tensor.gradient is not None:
                gradient_tensors.append(tensor.gradient)
                
        tensors.extend(gradient_tensors)
        
        return tensors
    
    @staticmethod
    def get_leaf_tensors(tensors):
        leaf_tensors = list()
        for tensor in tensors:
            if len(tensor.parents) == 0:
                leaf_tensors.append(tensor)
        return leaf_tensors
    
    @staticmethod
    def get_require_grads_tensors(tensors, include_intermediate=True):
        def _if_require_grad_rcsv(tensor):
            for nomial in tensor.parent_nomials:
                for parent in nomial:
                    if _if_require_grad_rcsv(parent):
                        return True
            return tensor.require_grads
        
        require_grads_tensors = list()
        if not include_intermediate:
            for tensor in tensors:
                if tensor.require_grads:
                    require_grads_tensors.append(tensor)
        else:
            for tensor in tensors:
                if _if_require_grad_rcsv(tensor):
                    require_grads_tensors.append(tensor)
        return require_grads_tensors
    
    @staticmethod
    def visualize(tensors, filename, format="pdf"):
        def _get_color(tensor):
            if tensor.require_grads:
                color = "red"
            elif len(tensor.parent_nomials)==0:
                color = "orange"
            elif tensor.gradient_of is not None:
                if tensor.gradient_of.require_grads:
                    color = "cyan"
                else:
                    color = "darkturquoise"
            else:
                color = "azure"
            return color
        
        f = graphviz.Digraph()
        for tensor in tensors:
            f.node(name=tensor.name,
                   label=str(tensor),
                   id=tensor.name,
                   shape="box",
                   style="filled",
                   fillcolor=_get_color(tensor))
            for nomial in tensor.parent_nomials:
                for parent in nomial:
                    f.edge(parent.name, tensor.name)
        f.render(filename, format=format, cleanup=True)
    
    def __str__(self):
        parent_str = ""
        for nomial in self.parent_nomials:
            for term in nomial:
                parent_str += term.name + "*"
            parent_str = parent_str[:-1] + "+"
        parent_str = parent_str[:-1]
        return f"(name={self.name}, dimensions={self.dimensions}, parent={parent_str})"


def multi_head_attention_comp_graph(input_tensor, dimension_table, prefix=""):
    assert input_tensor is not None
    assert isinstance(input_tensor, Tensor)
    
    symbol_B = dimension_table["B"]
    symbol_S = dimension_table["S"]
    symbol_H = dimension_table["H"]
    symbol_D = dimension_table["DModel"]
    
    tensors = list()
    
    x = input_tensor
    
    wq = Tensor(prefix+"WQ", (symbol_H, symbol_D, symbol_D), require_grads=True)
    q = Tensor(prefix+"Q", (symbol_B, symbol_S, symbol_H, symbol_D))
    q.add_parent_product(wq, x)
    tensors.append(wq)
    tensors.append(q)
    
    wk = Tensor(prefix+"WK", (symbol_H, symbol_D, symbol_D), require_grads=True)
    k = Tensor(prefix+"K", (symbol_B, symbol_S, symbol_H, symbol_D))
    k.add_parent_product(wk, x)
    tensors.append(wk)
    tensors.append(k)
    
    wv = Tensor(prefix+"WV", (symbol_H, symbol_D, symbol_D), require_grads=True)
    v = Tensor(prefix+"V", (symbol_B, symbol_S, symbol_H, symbol_D))
    v.add_parent_product(wv, x)
    tensors.append(wv)
    tensors.append(v)
    
    qk = Tensor(prefix+"QK", (symbol_B, symbol_S, symbol_S, symbol_H))
    qk.add_parent_product(q, k)
    qkv = Tensor(prefix+"QKV", (symbol_B, symbol_S, symbol_H, symbol_D))
    qkv.add_parent_product(qk, v)
    tensors.append(qk)
    tensors.append(qkv)
    
    res = Tensor(prefix+"attRes", (symbol_B, symbol_S, symbol_H, symbol_D))
    res.add_parent_add(qkv)
    res.add_parent_add(x)
    tensors.append(res)
    
    norm = Tensor(prefix+"attNorm", (symbol_B, symbol_S, symbol_H, symbol_D))
    norm.add_parent_add(res)
    tensors.append(norm)
    
    output_tensor = norm
    return tensors, output_tensor
    

def transformer_feed_forward_comp_graph(input_tensor, dimension_table, prefix=""):
    assert input_tensor is not None
    assert isinstance(input_tensor, Tensor)
    
    symbol_B = dimension_table["B"]
    symbol_S = dimension_table["S"]
    symbol_H = dimension_table["H"]
    symbol_DModel = dimension_table["DModel"]
    symbol_DFF = dimension_table["DFF"]
    
    tensors = list()
    
    x0 = input_tensor
    
    w1 = Tensor(prefix+"W1", (symbol_H, symbol_DModel, symbol_H, symbol_DFF), require_grads=True)
    x1 = Tensor(prefix+"X1", (symbol_B, symbol_S, symbol_H, symbol_DFF))
    x1.add_parent_product(w1, x0)
    tensors.append(w1)
    tensors.append(x1)
    
    w2 = Tensor(prefix+"W2", (symbol_H, symbol_DFF, symbol_H, symbol_DModel), require_grads=True)
    x2 = Tensor(prefix+"X2", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    x2.add_parent_product(w2, x1)
    tensors.append(w2)
    tensors.append(x2)
    
    res = Tensor(prefix+"ffnRes", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    res.add_parent_add(x2)
    res.add_parent_add(x0)
    tensors.append(res)
    
    norm = Tensor(prefix+"ffnNorm", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    norm.add_parent_add(res)
    tensors.append(norm)
    
    return tensors, norm


def transformer_comp_graph(num_stack, dimension_table):
    tensors = list()
    
    symbol_B = dimension_table["B"]
    symbol_S = dimension_table["S"]
    symbol_H = dimension_table["H"]
    symbol_DModel = dimension_table["DModel"]
    symbol_DInput = dimension_table["DInput"]
    symbol_DOutput = dimension_table["DOutput"]
    
    x_in_emb = Tensor("inputEmbedX", (symbol_B, symbol_S, symbol_H, symbol_DInput))
    w_in_emb = Tensor("inputEmbedW", (symbol_H, symbol_DInput, symbol_H, symbol_DModel), require_grads=True)
    y_in_emb = Tensor("inputEmbedY", (symbol_B, symbol_S, symbol_H, symbol_DModel))
    y_in_emb.add_parent_product(w_in_emb, x_in_emb)
    tensors.append(x_in_emb)
    tensors.append(w_in_emb)
    tensors.append(y_in_emb)
    
    input_tensor = y_in_emb
    for stack in range(num_stack):
        prefix = f"stack{stack}"
    
        stack_mha_tensors, input_tensor = multi_head_attention_comp_graph(input_tensor, dimension_table, prefix)
        tensors.extend(stack_mha_tensors)
        
        stack_ffn_tensors, input_tensor = transformer_feed_forward_comp_graph(input_tensor, dimension_table, prefix)
        tensors.extend(stack_ffn_tensors)
        
    x_out_emb = input_tensor
    w_out_emb = Tensor("outputEmbedW", (symbol_H, symbol_DModel, symbol_H, symbol_DOutput), require_grads=True)
    y_out_emb = Tensor("outputEmbedY", (symbol_B, symbol_S, symbol_H, symbol_DOutput))
    y_out_emb.add_parent_product(w_out_emb, x_out_emb)
    tensors.append(w_out_emb)
    tensors.append(y_out_emb)
    
    return tensors, y_out_emb


if __name__ == '__main__':
    dimension_table = {
        "B": Dimension("B"),
        "S": Dimension("S"),
        "H": Dimension("H"),
        "DInput": Dimension("DVocab"),
        "DModel": Dimension("DModel"),
        "DFF": Dimension("DFF"),
        "DOutput": Dimension("DOutput")
    }
    
    fwd_graph, y = transformer_comp_graph(2, dimension_table)
    Tensor.visualize(fwd_graph, "fwd")
    
    bwd_graph = Tensor.build_backward_comp_graph(fwd_graph, y)
    Tensor.visualize(bwd_graph, "bwd")
    