class Tensor:
    def __init__(self, shape, hidden=(), require_grads=False, ops=False):
        self.shape = shape
        self.hidden = list(hidden)
        
        self.parents = list()
        self.ops = ops
        self.require_grads_ = require_grads
        
        self.gradient = None
        self.gradient_of = None
    
    @property
    def require_grads(self):
        if self.require_grads_:
            return True
        for parent in self.parents:
            if parent.require_grads:
                return True
        return False
        
