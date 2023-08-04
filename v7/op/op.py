class OP:
    def __init__(self):
        self.context = dict()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
    def backward(self, *args, **kwargs):
        raise NotImplementedError()
