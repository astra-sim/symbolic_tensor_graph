from symbol import Symbol
from shape import Shape
import graphviz

class Tensor:
    def __init__(self, name, dimensions, require_grads=False):
        self.name = name
        self.dimensions = list()
        for dimension in dimensions:
            assert isinstance(dimension, Symbol)
            self.dimensions.append(dimension)
            
        self.require_grads=require_grads
        self.gradient = None
        self.gradient_of = None
        
        self.shaped_tensor_in = list()
        self.shaped_tensor_out = list()
        
        self._require_grads_itermediate=None
        self._visited=False
        
    def create_gradient(self):
        assert self.gradient is None
        gradient = Tensor("d_"+self.name, self.dimensions)
        self.gradient = gradient
        gradient.gradient_of = self
        return gradient
    
    def as_shape(self, shape, copy=True):
        if not isinstance(shape, Shape):
            shape = Shape(shape, copy=False)
        if copy:
            shape = shape.copy()
        ret = ShapedTensor(self, shape)
        self.add_shaped_tensor_out(ret)
        return ret
    
    def add_shaped_tensor_in(self, shaped_tensor_in):
        assert isinstance(shaped_tensor_in, ShapedTensor)
        assert shaped_tensor_in.shape.has_same_dimensions(Shape(self.dimensions))
        self.shaped_tensor_in.append(shaped_tensor_in)
        
    def add_shaped_tensor_out(self, shaped_tensor_out):
        assert isinstance(shaped_tensor_out, ShapedTensor)
        assert shaped_tensor_out.shape.has_same_dimensions(Shape(self.dimensions))
        self.shaped_tensor_out.append(shaped_tensor_out)
        
    def _add_parent_product(self, child, parent1, parent2):
        assert isinstance(child, ShapedTensor)
        assert isinstance(parent1, ShapedTensor)
        assert isinstance(parent2, ShapedTensor)
        self.add_shaped_tensor_in(child)
        child.parents.append(parent1)
        child.parents.append(parent2)
        
    def add_parent_product(child, einsum_str, parent1, parent2):
        assert isinstance(einsum_str, str)
        assert isinstance(parent1, ShapedTensor)
        assert isinstance(parent2, ShapedTensor)
        
        terms = einsum_str.split("->")
        assert len(terms) == 2
        inputs, outputs = terms[0], terms[1]
        del terms
            
        inputs = inputs.split(",")
        assert len(inputs) == 2
            
        char_set = list()
        char_grouped_dimension_map = dict()
            
        for i, char_ in enumerate(inputs[0]):
            char_grouped_dimension_map[char_] = parent1.shape.grouped_dimensions[i]
            if not char_ in char_set:
                char_set.append(char_)
                
        for i, char_ in enumerate(inputs[1]):
            char_grouped_dimension_map[char_] = parent2.shape.grouped_dimensions[i]
            if not char_ in char_set:
                char_set.append(char_)
                    
        for char_ in outputs:
            assert char_ in char_set
                
        unique, shared, reduced = list(), list(), list()
        for char_ in char_set:
            input_appear_cnt = 0
            if char_ in inputs[0]:
                input_appear_cnt += 1
            if char_ in inputs[1]:
                input_appear_cnt += 1
                    
            if char_ in outputs:
                if input_appear_cnt == 1:
                    unique.append(char_)
                elif input_appear_cnt == 2:
                    shared.append(char_)
                else:
                    assert False
            else:
                reduced.append(char_)

        output_shape = list()
        for char_ in outputs:
            output_shape.append(char_grouped_dimension_map[char_])
        output_shape = Shape(output_shape)
        output_tensor = ShapedTensor(child, output_shape)
        child.add_shaped_tensor_in(output_tensor)
        output_tensor.parents.append(parent1)
        output_tensor.parents.append(parent2)

    def add_parent_add(child, parent):
        assert isinstance(parent, ShapedTensor)
        output_shape = parent.shape
        output_tensor = ShapedTensor(child, output_shape)
        child.add_shaped_tensor_in(output_tensor)
        output_tensor.parents.append(parent)

    def transfer_gradient(self):
        if self._visited:
            return
        assert self.gradient is not None
        for shaped_tensor_in in self.shaped_tensor_in:
            if len(shaped_tensor_in.parents) == 1:
                parent = shaped_tensor_in.parents[0]
                parent_tensor = parent.tensor
                if parent_tensor.gradient is None:
                    parent_tensor.create_gradient()
                child_gradient_shaped_tensor = self.gradient.as_shape(shaped_tensor_in.shape, copy=False)
                parent_tensor.gradient.add_parent_add(child_gradient_shaped_tensor)
            elif len(shaped_tensor_in.parents) == 2:
                p1_shaped_tensor = shaped_tensor_in.parents[0]
                p2_shaped_tensor = shaped_tensor_in.parents[1]
                p1_tensor = p1_shaped_tensor.tensor
                p2_tensor = p2_shaped_tensor.tensor
                child_gradient_shaped_tensor = self.gradient.as_shape(shaped_tensor_in.shape, copy=False)
                if p1_tensor._require_grads_itermediate:
                    if p1_tensor.gradient is None:
                        p1_tensor.create_gradient()
                    p1_gradient_shaped_tensor = p1_tensor.gradient.as_shape(p1_shaped_tensor.shape, copy=False)
                    p1_tensor.gradient._add_parent_product(p1_gradient_shaped_tensor, p2_shaped_tensor, child_gradient_shaped_tensor)
                if p2_tensor._require_grads_itermediate:
                    if p2_tensor.gradient is None:
                        p2_tensor.create_gradient()
                    p2_gradient_shaped_tensor = p2_tensor.gradient.as_shape(p2_shaped_tensor.shape, copy=False)
                    p2_tensor.gradient._add_parent_product(p2_gradient_shaped_tensor, p1_shaped_tensor, child_gradient_shaped_tensor)
            else:
                assert False
        self._visited = True
        for shaped_tensor_in in self.shaped_tensor_in:
            for parent_shaped_tensor in shaped_tensor_in.parents:
                parent_tensor = parent_shaped_tensor.tensor
                if parent_tensor._require_grads_itermediate:
                    parent_tensor.transfer_gradient()
                
    @staticmethod
    def update_require_grads_itermediate(tensors):
        def _grads_rcsv(tensor):
            assert isinstance(tensor, Tensor)
            for shaped_in in tensor.shaped_tensor_in:
                for parent in shaped_in.parents:
                    if _grads_rcsv(parent.tensor):
                        parent.tensor._require_grads_itermediate = True
                        return True
            if tensor.require_grads:
                tensor._require_grads_itermediate = True
                return True
            return False
            
        for tensor in tensors:
            assert isinstance(tensor, Tensor)
            if not tensor.require_grads:
                tensor._require_grads_itermediate = None
            else:
                tensor._require_grads_itermediate = True
        
        for tensor in tensors:
            _grads_rcsv(tensor)
    
    @staticmethod
    def build_bwd_graph(tensors, output_tensor):
        for tensor in tensors:
            tensor._visited = False
        grad_tensors = list()
        Tensor.update_require_grads_itermediate(tensors)
        output_tensor.create_gradient()
        output_tensor.transfer_gradient()
        for tensor in tensors:
            if tensor.gradient is not None:
                grad_tensors.append(tensor.gradient)
        tensors.extend(grad_tensors)
        return tensors
    
    @staticmethod
    def clear_bwd_graph(tensors):
        grad_tensors = list()
        for tensor in tensors:
            if tensor.gradient is not None:
                tensor.gradient = None
            if tensor.gradient_of is not None:
                grad_tensors.append(tensor)
            shaped_tensor_in_to_remove = list()
            for shaped_tensor_in in tensor.shaped_tensor_in:
                for shaped_parent in shaped_tensor_in.parents:
                    if shaped_parent.tensor.gradient_of is not None:
                        shaped_tensor_in_to_remove.append(shaped_tensor_in)
                        break
            for shaped_tensor_in_to_remove_ in shaped_tensor_in_to_remove:
                tensor.shaped_tensor_in.remove(shaped_tensor_in_to_remove_)
        for tensor in grad_tensors:
            tensors.remove(tensor)
        return tensors
    
    @staticmethod
    def visualize_graph(tensors, filename, format="pdf"):
        def _tensor_color(tensor):
            if isinstance(tensor, ShapedTensor):
                tensor = tensor.tensor
            if tensor.gradient_of is not None:
                if tensor.gradient_of.require_grads:
                    return "cadetblue1"
                else:
                    return "darkolivegreen1"
            if tensor.require_grads:
                return "chocolate1"
            return "azure2"
        f = graphviz.Digraph()
        for tensor in tensors:
            sub_graph = graphviz.Digraph(name=tensor.name, graph_attr={'color': 'blue'})
            color = _tensor_color(tensor)
            sub_graph.node(name=tensor.name, 
                           label=tensor.name, 
                           id=tensor.name,
                           shape="box",
                           color=color)
            for i, shaped_tensor_in in enumerate(tensor.shaped_tensor_in):
                name = f"{tensor.name}_in_{i}"
                shaped_tensor_in._name = name
                sub_graph.node(name=name,
                               label=name,
                               id=name,
                               shape="invhouse",
                               color=color)
                sub_graph.edge(name, tensor.name, style="dotted")
            for i, shaped_tensor_out in enumerate(tensor.shaped_tensor_out):
                name = f"{tensor.name}_out_{i}"
                shaped_tensor_out._name = name
                sub_graph.node(name=name,
                               label=name,
                               id=name,
                               shape="house",
                               color=color)
                sub_graph.edge(tensor.name, name, style="dotted")
            f.subgraph(sub_graph)
        for tensor in tensors:
            for shaped_tensor_in in tensor.shaped_tensor_in:
                for parent in shaped_tensor_in.parents:
                    f.edge(parent._name, shaped_tensor_in._name)
        f.render(filename, format=format, cleanup=True)


class ShapedTensor:
    def __init__(self, tensor, shape):
        assert (isinstance(tensor, Tensor)) or (tensor is None)
        assert isinstance(shape, Shape)
        assert Shape(tensor.dimensions).has_same_dimensions(shape)
        
        self.tensor = tensor
        self.shape = shape
        
        self.parents = list()
        self._name = None
        
    def __del__(self):
        if self in self.tensor.shaped_tensor:
            self.tensor.shaped_tensor.remove(self)
