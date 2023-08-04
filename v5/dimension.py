class Dimension:
    registered_dimensions = list()
    
    def __init__(self, name, new_dimension=False):
        if new_dimension:
            assert name not in Dimension.registered_dimensions
        self.name = name
        if name in Dimension.registered_dimensions:
            assert not new_dimension
        else:
            Dimension.registered_dimensions.append(name)
            
    def copy(self):
        return Dimension(name=self.name, new_dimension=False)
    
    def same_dimension(one, another):
        assert isinstance(one, Dimension)
        assert isinstance(another, Dimension)
        assert one.name == another.name
        
    def __eq__(one, another):
        return one.same_dimension(another)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
