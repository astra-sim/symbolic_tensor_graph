from tensor_comp_graph import Tensor
import copy
import graphviz


class Symbol:
    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        
    def __str__(self):
        return self.name
        
    
class FractionExpressions:
    def __init__(self, numerators, denumerators):
        self.numerators = numerators
        self.denumerators = denumerators
        
    def simplify(self):
        simplified_numerators = list()
        simplified_denumerators = list()
        
        # simplify each numerator
        for numerator in self.numerators:
            if isinstance(numerator, Symbol):
                simplified_numerators.append(numerator)
            elif isinstance(numerator, Expression):
                simplified_sub_expression = numerator.simplify()
                simplified_numerators.extend(simplified_sub_expression.numerators)
                simplified_denumerators.extend(simplified_sub_expression.denumerators)
            else:
                # should be either symbol or expression, otherwise invalid
                assert False
                
        # simplify each denumerator
        for denumerator in self.denumerators:
            if isinstance(denumerator, Symbol):
                simplified_denumerators.append(numerator)
            elif isinstance(denumerator, Expression):
                simplified_sub_expression = numerator.simplify()
                simplified_numerators.extend(simplified_sub_expression.denumerators)
                simplified_denumerators.extend(simplified_sub_expression.numerators)
            else:
                # should be either symbol or expression, otherwise invalid
                assert False
        
        # cancle terms in either numerator and denumerator
        numerators_to_be_removed = list()
        for term in simplified_numerators:
            if term in simplified_denumerators:
                numerators_to_be_removed.append(term)
                simplified_denumerators.remove(term)
        for term in numerators_to_be_removed:
            simplified_numerators.remove(term)
        
        return FractionExpressions(simplified_numerators, simplified_denumerators)
    
    def eval_value(self, symbol_value_table):
        value = 1
        simplified = self.simplify()
        for term in simplified.numerators:
            assert isinstance(term, Symbol)
            assert term in symbol_value_table
            value *= symbol_value_table[term]
        
        for term in simplified.denumerators:
            assert isinstance(term, Symbol)
            assert term in symbol_value_table
            value /= symbol_value_table[term]
        
        return value
    
    def literally_equal(one, another):
        # this only check whether these two expression literally the same
        another_numerators = copy.deepcopy(another.numerators)
        for numerator in one.numerators:
            if not numerator in another_numerators:
                return False
            else:
                another_numerators.remove(numerator)
        if not another_numerators.empty():
            return False
        
        another_denumerators = copy.deepcopy(another.denumerators)
        for denumerator in one.denumerators:
            if not denumerator in another_denumerators:
                return False
            else:
                another_denumerators.remove(denumerator)
        if not another_denumerators.empty():
            return False
        return True
    
    def simplified_equal(one, another):
        return one.simplify().literally_equal(another.simplify())
    
    def __eq__(one, another):
        return one.literally_equal(another)
    
    def __str__(self):
        ret = "{"
        for _ in self.numerators:
            ret += _.__str__() + ", "
        ret += "},{"
        for _ in self.denumerators:
            ret += _.__str__() + ", "
        ret += "}"
        return ret
        


class Shape:
    def __init__(self, shape, hidden=()):
        self.shape = list()
        self.hidden = list()
        for expr in shape:
            self.shape.append(copy.copy(expr))
        for symbol in hidden:
            self.hidden.append(symbol)
        
        
    @staticmethod
    def einsum(einsum, shape1, shape2):
        assert len(shape1.hidden) == 0
        assert len(shape2.hidden) == 0
        inputs_expr, output_expr = einsum.split("->")
        input1_expr, input2_expr = inputs_expr.split(",")
        print(input1_expr, shape1)
        print(input2_expr, shape2)
        assert len(input1_expr) == len(shape1.shape)
        assert len(input2_expr) == len(shape2.shape)
        char_expression_map = dict()
        for char, expression in zip(input1_expr, shape1.shape):
            if not char in char_expression_map:
                char_expression_map[char] = expression
        for char, expression in zip(input2_expr, shape2.shape):
            if not char in char_expression_map:
                char_expression_map[char] = expression
        for char in output_expr:
            # char in output should comes from at least one input
            assert char in char_expression_map
        
        unique, shared, reduced = list(), list(), list()
        for char in char_expression_map.keys():
            char_appears_in_input = 0
            if char in input1_expr:
                char_appears_in_input += 1
            if char in input2_expr:
                char_appears_in_input += 1
                
            if char in output_expr:
                if char_appears_in_input == 2:
                    shared.append(char_expression_map[char])
                elif char_appears_in_input == 1:
                    unique.append(char_expression_map[char])
                else:
                    # char in output comes from neither inputs
                    assert False
            else:
                # reduced dimension, assume no auto broadcast
                assert char_appears_in_input == 2
                reduced.append(char_expression_map[char])
        
        new_shape, new_hidden = list(), list()
        new_shape.extend(shared)
        new_shape.extend(unique)
        for expr in reduced:
            new_hidden.extend(expr.simplify().denumerators)
        return Shape(new_shape, new_hidden)
    
    def __eq__(one, another):
        if (one is None) or (another is None):
            return False
        if not one.shape == another.shape:
            return False
        if not one.hidden == another.hidden:
            return False
        return True
    
    def __str__(self):
        ret = "shape={"
        for shape in self.shape:
            if shape is None:
                ret += "None, "
            else:
                ret += str(shape) + ", "
        ret += "}, hidden={"
        for hidden in self.hidden:
            ret += str(hidden) + ", "
        ret += "}"
        return ret


class ShapedTensor(Tensor):
    def __init__(self, name, default_shape=None, require_grads=False):
        super(ShapedTensor, self).__init__(name, require_grads)
        self.shapes = list()
        self.shapes.append(default_shape)
        self.parent_shapes = list()
    
    def create_gradient(self):
        assert self.gradient is None
        gradient = ShapedTensor("d_"+self.name)
        gradient.shapes = self.shapes
        gradient.parent_shapes = self.parent_shapes
        self.gradient = gradient
        gradient.gradient_of = self
        return gradient
    
    def transfer_gradient(self, require_grads_tensors):
        assert self.gradient is not None
        for i, nomial in enumerate(self.parents):
            assert len(self.parent_shapes[i]) == len(nomial)
            if len(nomial) == 1:
                parent = nomial[0]
                parent_shape = self.parent_shapes[i][0]
                if (parent.gradient is None) and (parent in require_grads_tensors):
                    parent.create_gradient()
                    require_grads_tensors.remove(parent)
                    print(len(require_grads_tensors))
                if parent.gradient is not None:
                    parent.gradient.add_parent_add(parent, parent_shape)
            elif len(nomial) == 2:
                p1, p2 = nomial
                p1_shape, p2_shape = self.parent_shapes[i]
                result_shape = self.gradient.shapes[i+1]
                if (p1.gradient is None) and (p1 in require_grads_tensors):
                    p1.create_gradient()
                    require_grads_tensors.remove(p1)
                    print(len(require_grads_tensors))
                if p1.gradient is not None:
                    p1.gradient._add_parent_product(self.gradient, p2, result_shape, p2_shape, p1_shape)
                
                if (p2.gradient is None) and (p2 in require_grads_tensors):
                    p2.create_gradient()
                    require_grads_tensors.remove(p2)
                    print(len(require_grads_tensors))
                if p2.gradient is not None:
                    p2.gradient._add_parent_product(self.gradient, p1, result_shape, p1_shape, p2_shape)
            else:
                raise NotImplementedError("For now only product of two are supported")
        for nomial in self.parents:
            for parent in nomial:
                if (not parent._visited) and (parent.gradient is not None):
                    parent._visited = True
                    parent.transfer_gradient(require_grads_tensors)
        return 

    def add_parent_product(self, einsum, p1, p2, p1_shape, p2_shape):
        result_shape = Shape.einsum(einsum, p1_shape, p2_shape)
        self._add_parent_product(p1, p2, p1_shape, p2_shape, result_shape)
    
    def _add_parent_product(self, p1, p2, p1_shape, p2_shape, result_shape):
        assert p1_shape in p1.shapes
        assert p2_shape in p2.shapes
        super(ShapedTensor, self).add_parent_product(p1, p2)
        self.shapes.append(result_shape)
        self.parent_shapes.append((p1_shape, p2_shape))
        
    def add_parent_add(self, p1, p1_shape):
        assert p1_shape in p1.shapes
        super(ShapedTensor, self).add_parent_add(p1)
        self.shapes.append(p1_shape)
        self.parent_shapes.append((p1_shape,))
        
    @staticmethod
    def clean_bwd_graph(tensors):
        tensors = Tensor.clean_bwd_graph(tensors)
        return tensors
        
    @staticmethod
    def build_backward_comp_graph(tensors, output_tensor):
        assert len(output_tensor.shapes) == len(output_tensor.parents) + 1
        output_tensor_shape = output_tensor.shapes[-1]
        # make sure current input tensors are only fwd graph, no gradient tensors
        tensors = ShapedTensor.clean_bwd_graph(tensors)
        assert isinstance(output_tensor, ShapedTensor)
        for tensor in tensors:
            tensor._visited = False
            
        require_grads_tensors = ShapedTensor.get_require_grads_tensors(tensors)
        output_tensor.create_gradient()
        require_grads_tensors.remove(output_tensor)
        output_tensor.transfer_gradient(require_grads_tensors)
        output_tensor.gradient.add_parent_add(output_tensor, output_tensor_shape)
        
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
                gradient_shape = tensor.gradient.shapes[0]
                if gradient_shape is None:
                    gradient_shape = tensor.gradient.shapes[-1]
                tensor.add_parent_add(tensor.gradient, gradient_shape)
        return tensors
    
    @staticmethod
    def visualize_comp_graph(tensors, filename):
        f = graphviz.Digraph()
        for tensor in tensors:
            label = str(tensor)
            if tensor.require_grads:
                fillcolor = "red"
            elif len(tensor.parents)==0:
                fillcolor="orange"
            elif tensor.gradient_of is not None:
                if tensor.gradient_of.require_grads:
                    fillcolor = "cyan"
                else:
                    fillcolor = "darkturquoise"
            else:
                fillcolor = "azure"
            f.node(name=f"{tensor.name}",
                   label=label,
                   id=tensor.name,
                   shape="box",
                   style="filled",
                   fillcolor=fillcolor)
            for nomial in tensor.parents:
                for parent in nomial:
                    f.edge(parent.name, tensor.name)
        f.render(filename, format="pdf", cleanup=True)
            
        
    @staticmethod
    def get_leaf_tensors(tensors):
        return Tensor.get_leaf_tensors(tensors)
        
    @staticmethod
    def get_require_grads_tensors(tensors, include_intermidiat=True):
        return Tensor.get_require_grads_tensors(tensors, include_intermidiat)
    
    def __str__(self):
        ret = f"name={self.name}, shapes=["
        for shape in self.shapes:
            ret += str(shape) + ", "
        ret += "]"
        return ret


def multi_head_attention_comp_graph(input_tensor, shape_symbol_table, prefix="", input_shape=None):
    if input_shape is None:
        input_shape = input_tensor.shapes[0]
    assert input_shape is not None
    assert input_shape in input_tensor.shapes
    
    symbol_B = shape_symbol_table["B"]
    symbol_S = shape_symbol_table["S"]
    symbol_H = shape_symbol_table["H"]
    symbol_D = shape_symbol_table["DModel"]
    
    expr_B = FractionExpressions((symbol_B,), ())
    expr_S = FractionExpressions((symbol_S,), ())
    expr_H = FractionExpressions((symbol_H,), ())
    expr_D = FractionExpressions((symbol_D,), ())
    expr_Dmodel = FractionExpressions((symbol_D, symbol_H), ())
    if not prefix == "":
        prefix += "_"
    tensors = list()
    
    x = input_tensor
    x_shape = Shape((expr_B, expr_S, expr_H, expr_D), ())
    assert x.shapes[0] is None
    x.shapes[0] = x_shape
    
    wq_shape = Shape((expr_H, expr_D, expr_D))
    wq = ShapedTensor(prefix+"WQ", default_shape=wq_shape, require_grads=True)
    q = ShapedTensor(prefix+"Q")
    q.add_parent_product("bshd,hde->bshe", x, wq, x_shape, wq_shape)
    tensors.append(wq)
    tensors.append(q)
    
    wk_shape = Shape((expr_H, expr_D, expr_D))
    wk = ShapedTensor(prefix+"WK", default_shape=wk_shape, require_grads=True)
    k = ShapedTensor(prefix+"K")
    k.add_parent_product("bshd,hde->bshe", x, wk, x_shape, wk_shape)
    tensors.append(wk)
    tensors.append(k)
    
    wv_shape = Shape((expr_H, expr_D, expr_D))
    wv = ShapedTensor(prefix+"WV", default_shape=wv_shape, require_grads=True)
    v = ShapedTensor(prefix+"V")
    v.add_parent_product("bshd,hde->bshe", x, wv, x_shape, wv_shape)
    tensors.append(wv)
    tensors.append(v)
    
    q_shape, k_shape = q.shapes[1], k.shapes[1]
    qk = ShapedTensor(prefix+"QK")
    qk.add_parent_product("bshd,bwhd->bswh", q, k, q_shape, k_shape)
    tensors.append(qk)
    
    qk_shape, v_shape = qk.shapes[1], v.shapes[1]
    qkv_shape = Shape((expr_B, expr_S, expr_Dmodel), ())
    qkv = ShapedTensor(prefix+"QKV", default_shape = qkv_shape)
    qkv.add_parent_product("bswh,bwhd->bshd", qk, v, qk_shape, v_shape)
    tensors.append(qkv)
    
    qkv_shape = qkv.shapes[0]
    res = ShapedTensor(prefix+"attRes")
    res.add_parent_add(qkv, qkv_shape)
    res.add_parent_add(x, x.shapes[1])
    tensors.append(res)
    
    res_shape = res.shapes[2]
    norm = ShapedTensor(prefix+"attNorm")
    norm.add_parent_add(res, res_shape)
    tensors.append(norm)
    
    output_tensor = norm
    output_shape = norm.shapes[1]
    
    return tensors, (input_tensor, output_tensor, input_shape, output_shape)


def transformer_feed_forward_comp_graph(input_tensor, shape_symbol_table, prefix="", input_shape=None):
    if input_shape is None:
        input_shape = input_tensor.shapes[0]
    assert input_shape is not None
    assert input_shape in input_tensor.shapes
    
    symbol_B = shape_symbol_table["B"]
    symbol_S = shape_symbol_table["S"]
    symbol_H = shape_symbol_table["H"]
    symbol_Dmodel = shape_symbol_table["DModel"]
    symbol_Dff = shape_symbol_table["DFF"]
    
    expr_B = FractionExpressions((symbol_B,), ())
    expr_S = FractionExpressions((symbol_S,), ())
    expr_Dmodel = FractionExpressions((symbol_H, symbol_Dmodel), ())
    expr_Dff = FractionExpressions((symbol_H, symbol_Dff), ())
    if not prefix == "":
        prefix += "_"
    tensors = list()
    
    x0 = input_tensor
    x0_shape = Shape((expr_B, expr_S, expr_Dmodel), ())
    assert x0.shapes[0] is None
    x0.shapes[0] = x0_shape
    
    w1_shape = Shape((expr_Dmodel, expr_Dff), ())
    w1 = ShapedTensor(prefix+"W1", default_shape=w1_shape, require_grads=True)
    x1 = ShapedTensor(prefix+"X1")
    x1.add_parent_product("bsd,de->bse", x0, w1, x0_shape, w1_shape)
    tensors.append(w1)
    tensors.append(x1)
    
    w2_shape, x1_shape = Shape((expr_Dff, expr_Dmodel), ()), x1.shapes[1]
    w2 = ShapedTensor(prefix+"W2", default_shape=w2_shape, require_grads=True)
    x2 = ShapedTensor(prefix+"X2")
    x2.add_parent_product("bsd,de->bse", x1, w2, x1_shape, w2_shape)
    tensors.append(w2)
    tensors.append(x2)
    
    x2_shape = x2.shapes[1]
    res = ShapedTensor(prefix+"ffnRes")
    res.add_parent_add(x2, x2_shape)
    tensors.append(res)
    
    res_shape = res.shapes[1]
    norm = ShapedTensor(prefix+"ffnNorm")
    norm.add_parent_add(res, res_shape)
    tensors.append(norm)
    
    output_tensor = norm
    output_shape = norm.shapes[1]
    
    return tensors, (input_tensor, output_tensor, input_shape, output_shape)


def transformer_comp_graph(num_stack, shape_symbol_table):
    tensors = list()
    # TODO: finish it according to tensor version, also test, update visualize with shape information.
    
    symbol_B = shape_symbol_table['B']
    symbol_S = shape_symbol_table['S']
    symbol_H = shape_symbol_table['H']
    symbol_Dvocab = shape_symbol_table['DVocab']
    symbol_Dmodel = shape_symbol_table['DModel']
    symbol_Doutput = shape_symbol_table['DOutput']
    
    expr_B = FractionExpressions((symbol_B,), ())
    expr_S = FractionExpressions((symbol_S,), ())
    expr_Dvocab = FractionExpressions((symbol_Dvocab, symbol_H), ())
    expr_Dmodel = FractionExpressions((symbol_Dmodel, symbol_H), ())
    expr_Doutput = FractionExpressions((symbol_Doutput, symbol_H), ())
    expr_H = FractionExpressions((symbol_H,), ())
    expr_Dmodel_only = FractionExpressions((symbol_Dmodel,), ())
    
    x_in_embed_shape = Shape((expr_B, expr_S, expr_Dvocab), ())
    w_in_embed_shape = Shape((expr_Dvocab, expr_Dmodel), ())
    x_in_embed = ShapedTensor("inputEmbedX", default_shape=x_in_embed_shape, require_grads=False)
    w_in_embed = ShapedTensor("inputEmbedW", default_shape=w_in_embed_shape, require_grads=True)
    y_in_embed = ShapedTensor("inputEmbedY")
    y_in_embed.add_parent_product("bsd,de->bse", x_in_embed, w_in_embed, x_in_embed_shape, w_in_embed_shape)
    tensors.append(x_in_embed)
    tensors.append(w_in_embed)
    tensors.append(y_in_embed)
    
    input_tensor = y_in_embed
    input_tensor_shape = y_in_embed.shapes[1]
    for stack in range(num_stack):
        prefix = f"stack{stack}"
        
        stack_mha_tensors, (input_tensor, output_tensor, input_tensor_shape, output_tensor_shape) = \
            multi_head_attention_comp_graph(input_tensor, shape_symbol_table, prefix=prefix, input_shape=input_tensor_shape)
        tensors.extend(stack_mha_tensors)
        input_tensor = output_tensor
        input_tensor_shape = output_tensor_shape
        
        
        stack_ffn_tensors, (input_tensor, output_tensor, input_tensor_shape, output_tensor_shape) = \
            transformer_feed_forward_comp_graph(input_tensor, shape_symbol_table, prefix=prefix, input_shape=input_tensor_shape)
        tensors.extend(stack_ffn_tensors)
        input_tensor = output_tensor
        input_tensor_shape = output_tensor_shape
        
    x_out_embed_shape = input_tensor_shape
    w_out_embed_shape = Shape((expr_Dmodel, expr_Doutput), ())
    x_out_embed = input_tensor
    w_out_embed = ShapedTensor("outputEmbedW", default_shape=w_out_embed_shape, require_grads=True)
    y_out_embed = ShapedTensor("outputEmbedY")
    y_out_embed.add_parent_product("bsd,de->bse", x_out_embed, w_out_embed, x_out_embed_shape, w_out_embed_shape)
    tensors.append(w_out_embed)
    tensors.append(y_out_embed)
    return tensors, (x_in_embed, y_out_embed, x_in_embed_shape, y_out_embed.shapes[1])


if __name__ == '__main__':
    shape_symbol_table = {
        "B": Symbol("B"),
        "S": Symbol("S"),
        "H": Symbol("H"),
        "DVocab": Symbol("DVocab"),
        "DModel": Symbol("DModel"),
        "DFF": Symbol("DFF"),
        "DOutput": Symbol("DOutput")
    }
    fwd_comp_graph, (x, y, x_shape, y_shape) = transformer_comp_graph(1, shape_symbol_table)
    ShapedTensor.visualize_comp_graph(fwd_comp_graph, "shaped_fwd")
    
    require_grads_tensors = ShapedTensor.get_require_grads_tensors(fwd_comp_graph)
    
    bwd_comp_graph = ShapedTensor.build_backward_comp_graph(fwd_comp_graph, y)
    ShapedTensor.visualize_comp_graph(bwd_comp_graph, "shaped_bwd")
    
    gradient_updated_comp_graph = ShapedTensor.build_gradient_update_comp_graph(bwd_comp_graph)
    ShapedTensor.visualize_comp_graph(gradient_updated_comp_graph, "shaped_gradient_update")
