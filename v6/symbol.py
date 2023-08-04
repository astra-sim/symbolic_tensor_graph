class Symbol:    
    def __init__(self, name):
        self.name = name
        
    def copy(self):
        return Symbol(self.name)
    
    def same_symbol(one, another):
        assert isinstance(one, Symbol)
        assert isinstance(another, Symbol)
        return one.name == another.name

    def __eq__(one, another):
        return one.same_symbol(another)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
