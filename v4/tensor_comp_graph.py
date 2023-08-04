import copy, pydot, graphviz, sys


class Tensor:
    def __init__(self, name, require_grads=False):
        self.name = name
        self.require_grads = require_grads
        self.parents = list()
        self.gradient = None
        self.gradient_of = None
        self._visited = False
        
    def create_gradient(self):
        assert self.gradient is None
        gradient = Tensor("d_"+self.name)
        self.gradient = gradient
        gradient.gradient_of = self
        return gradient
           
    def transfer_gradient(self, require_grads_tensors):
        assert self.gradient is not None
        for nomial in self.parents:
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
        
        for nomial in self.parents:
            for parent in nomial:
                if (not parent._visited) and (parent.gradient is not None):
                    parent._visited = True
                    parent.transfer_gradient(require_grads_tensors)
        return

    def add_parent_product(self, p1, p2):
        self.parents.append((p1, p2))
        
    def add_parent_add(self, p1):
        self.parents.append((p1,))
        
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
    def build_gradient_update_comp_graph(tensors):
        for tensor in tensors:
            if tensor.require_grads:
                assert tensor.gradient is not None
                tensor.add_parent_add(tensor.gradient)
        return tensors
    
    @staticmethod
    def visualize_comp_graph(tensors, filename):
        f = graphviz.Digraph()
        for tensor in tensors:
            if tensor.require_grads:
                f.node(name=f"{tensor.name}",
                       label=f"{tensor.name}",
                        id=tensor.name,
                        shape="box",
                        style="filled",
                        fillcolor="red")
            elif len(tensor.parents)==0:
                f.node(name=f"{tensor.name}",
                       label=f"{tensor.name}",
                        id=tensor.name,
                        shape="box",
                        style="filled",
                        fillcolor="orange")
            elif tensor.gradient_of is not None:
                if tensor.gradient_of.require_grads:
                    f.node(name=f"{tensor.name}",
                           label=f"{tensor.name}",
                            id=tensor.name,
                            shape="box",
                            style="filled",
                            fillcolor="cyan")
                else:
                    f.node(name=f"{tensor.name}",
                       label=f"{tensor.name}",
                        id=tensor.name,
                        shape="record",
                        style="filled",
                        fillcolor="darkturquoise")
            else:
                f.node(name=f"{tensor.name}",
                       label=f"{tensor.name}",
                        id=tensor.name,
                        shape="box",
                        style="filled",
                        fillcolor="azure")
            for nomial in tensor.parents:
                for parent in nomial:
                    f.edge(parent.name, tensor.name)
        f.render(filename, format="pdf", cleanup=True)
        
    @staticmethod
    def get_leaf_tensors(tensors):
        leaf_tensors = list()
        for tensor in tensors:
            if len(tensor.parents) == 0:
                leaf_tensors.append(tensor)
        return leaf_tensors
    
    @staticmethod
    def get_require_grads_tensors(tensors, include_intermidiat=True):
        def _if_require_grad_rcsv(tensor):
            for nomial in tensor.parents:
                for parent in nomial:
                    if _if_require_grad_rcsv(parent):
                        return True
            return tensor.require_grads
        
        
        require_grads_tensors = list()
        if not include_intermidiat:
            for tensor in tensors:
                if tensor.require_grads:
                    require_grads_tensors.append(tensor)
        else:
            for tensor in tensors:
                if _if_require_grad_rcsv(tensor):
                    require_grads_tensors.append(tensor)
        return require_grads_tensors
        

def multi_head_attention_comp_graph(input_tensor, prefix=""):
    if not prefix == "":
        prefix += "_"
    tensors = list()
    
    x = input_tensor
    
    wq = Tensor(prefix+"WQ", True)
    q = Tensor(prefix+"Q")
    q.add_parent_product(wq, x)
    tensors.append(wq)
    tensors.append(q)
    
    wk = Tensor(prefix+"WK", True)
    k = Tensor(prefix+"K")
    k.add_parent_product(wk, x)
    tensors.append(wk)
    tensors.append(k)
    
    wv = Tensor(prefix+"WV", True)
    v = Tensor(prefix+"V")
    v.add_parent_product(wv, x)
    tensors.append(wv)
    tensors.append(v)
    
    qk = Tensor(prefix+"QK")
    qk.add_parent_product(q, k)
    qkv = Tensor(prefix+"QKV")
    qkv.add_parent_product(qk, v)
    res = Tensor(prefix+"attRes")
    res.add_parent_add(qkv)
    norm = Tensor(prefix+"attNorm")
    norm.add_parent_add(res)
    tensors.append(qk)
    tensors.append(qkv)
    tensors.append(res)
    tensors.append(norm)
    
    output_tensor = norm
    return tensors, (input_tensor, output_tensor)


def transformer_feed_forward_comp_graph(input_tensor, prefix=""):
    if not prefix == "":
        prefix += "_"
    tensors = list()
    x0 = input_tensor
    
    w1 = Tensor(prefix+"W1", True)
    x1 = Tensor(prefix+"X1")
    x1.add_parent_product(w1, x0)
    tensors.append(w1)
    tensors.append(x1)
    
    w2 = Tensor(prefix+"W2", True)
    x2 = Tensor(prefix+"X2")
    x2.add_parent_product(w2, x1)
    tensors.append(w2)
    tensors.append(x2)
    
    res = Tensor(prefix+"ffnRes")
    res.add_parent_add(x2)
    norm = Tensor(prefix+"ffnNorm")
    norm.add_parent_add(res)
    tensors.append(res)
    tensors.append(norm)
    
    output_tensor = norm
    return tensors, (input_tensor, output_tensor)


def transformer_comp_graph(num_stack):
    tensors = list()
    x_in_embed = Tensor("inputEmbedX")
    w_in_embed = Tensor("inputEmbedW", True)
    y_in_embed = Tensor("inputEmbedY")
    y_in_embed.add_parent_product(w_in_embed, x_in_embed)
    tensors.append(x_in_embed)
    tensors.append(w_in_embed)
    tensors.append(y_in_embed)
    
    input_tensor = y_in_embed
    for stack in range(num_stack):
        prefix = f"stack{stack}"
        
        stack_mha_tensors, (input_tensor, output_tensor) = multi_head_attention_comp_graph(input_tensor, prefix)
        tensors.extend(stack_mha_tensors)
        input_tensor = output_tensor
        
        stack_ffn_tensors, (input_tensor, output_tensor) = transformer_feed_forward_comp_graph(input_tensor, prefix)
        tensors.extend(stack_ffn_tensors)
        input_tensor = output_tensor
        
    x_out_embed = input_tensor
    w_out_embed = Tensor("outputEmbedW", True)
    y_out_embed = Tensor("outputEmbedY")
    y_out_embed.add_parent_product(w_out_embed, x_out_embed)
    tensors.append(w_out_embed)
    tensors.append(y_out_embed)
    return tensors, (x_in_embed, y_out_embed)


if __name__ == '__main__':
    fwd_comp_graph, (x, y) = transformer_comp_graph(1)
    Tensor.visualize_comp_graph(fwd_comp_graph, "fwd")
    
    require_grads_tensors = Tensor.get_require_grads_tensors(fwd_comp_graph)
    
    bwd_comp_graph = Tensor.build_backward_comp_graph(fwd_comp_graph, y)
    Tensor.visualize_comp_graph(bwd_comp_graph, "bwd")
    
    gradient_updated_comp_graph = Tensor.build_gradient_update_comp_graph(bwd_comp_graph)
    Tensor.visualize_comp_graph(gradient_updated_comp_graph, "gradient_update")
    
    