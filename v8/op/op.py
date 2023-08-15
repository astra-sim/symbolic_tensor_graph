class OP:
    def __init__(self):
        pass
    
    @classmethod
    def apply(cls, inputs):
        fwd_outputs = cls.forward(*inputs)
        cls.backward(fwd_outputs, inputs)
        return fwd_outputs
    
    def forward(self, *inputs):
        raise NotImplementedError()
    
    def backward(self, fwd_outputs, fwd_inputs):
        raise NotImplementedError()

        