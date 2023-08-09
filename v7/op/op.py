from tensor import Tensor

class OP:
    def __init__(self):
        self.context = dict()
        self.inputs = list()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError()
    
    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = cls()
        ret = ctx.forward(*args, **kwargs)
        assert isinstance(ret, Tensor)
        ret.context.append(ctx)
        return ret
    
    def output_sharding(self):
        new_sharding = dict()
        for input in self.inputs:
            for shape in input.shape:
                sharding = input.sharding[shape]
                if not shape in new_sharding:
                    new_sharding[shape] = sharding
                else:
                    assert new_sharding[shape] == sharding
        return

    def get_ops(self):
        raise NotImplementedError()