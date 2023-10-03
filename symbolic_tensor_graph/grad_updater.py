import copy
from .tensor import Tensor

class GradUpdater:
    def __init__(self, fwd_graph, bwd_graph):
        self.fwd_graph = fwd_graph
        self.bwd_graph = bwd_graph
        
    def get_require_grads_tensors(self):
        ret = list()
        for tensor in self.fwd_graph:
            if tensor.require_grads:
                ret.append(tensor)
        return ret
    
    def get_grad_of_tensor(self, tensor):
        id_ = tensor.id
        if id_.rfind('_') != -1:
            grad_id = id_[:id_.rfind('_')] + '_d_' + id_[id_.rfind('_')+1:]
        else:
            grad_id = 'd_' + id_
        # print(grad_id)
        for tensor in self.bwd_graph:
            if tensor.id == grad_id:
                return tensor
        assert False
        
    def update_tensor(self, x, dx):
        y = Tensor(create_empty=True)
        y.id = x.id + "_updated"
        assert x.require_grads
        y.require_grads = x.require_grads
        assert x.shape == dx.shape
        y.shape = copy.deepcopy(x.shape)
        assert x.hidden == dx.hidden
        y.hidden = copy.deepcopy(x.hidden)
        y.x1 = x.id
        y.x2 = dx.id
        y.op_attr = 'A'
        y.op_attr = ''
        y.x1_shape = copy.deepcopy(x.shape)
        y.x1_hidden = copy.deepcopy(x.hidden)
        y.x2_shape = copy.deepcopy(dx.shape)
        y.x2_hidden = copy.deepcopy(dx.hidden)
        y.direct_output_shape = copy.deepcopy(x.shape)
        y.direct_output_hidden = copy.deepcopy(x.hidden)
        y.post_communications = ''
        ops = 1
        for d in x.shape:
            ops = ops * d
        y.ops = ops
        return y
    
    def update_tensors(self):
        require_grads_tensor = self.get_require_grads_tensors()
        updated_tensors = list()
        for x in require_grads_tensor:
            dx = self.get_grad_of_tensor(x)
            y = self.update_tensor(x, dx)
            updated_tensors.append(y)
        return updated_tensors
