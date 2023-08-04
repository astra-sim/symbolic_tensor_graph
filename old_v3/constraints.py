from .tensor import Tensor

class Constraints:
    op_list = ["+", "-", "*", "/", "//", "(", ")"]
    def __init__(self, left, right, relationship="=="):
        super(Constraints, self).__init__()
        if self.relationship == "==":
            if len(Constraints.get_symbols(right))>1:
                _ = left
                left = right
                right = _
            assert len(Constraints.get_symbols(right))<2
        
        self.left = left
        self.right = right
        self.relationship = relationship


    @staticmethod
    def is_symbol(term):
        if term in Constraints.op_list:
            return False
        return True

    @staticmethod
    def get_symbols(expr):
        ret = set()
        for term in expr:
            if Constraints.is_symbol(term):
                ret.add(term)
        return ret
    


