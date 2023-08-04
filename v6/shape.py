from symbol import Symbol


class Shape:
    def __init__(self, grouped_dimensions, copy=False):
        self.grouped_dimensions = list()
        for group in grouped_dimensions:
            copy_group = list()
            if isinstance(group, Symbol):
                group = (group.copy(),)
            elif isinstance(group, tuple):
                for symbol in group:
                    copy_group.append(symbol.copy())
                group = tuple(copy_group)
            else:
                assert False
            self.grouped_dimensions.append(group)
     
    def flatten_dimensions(self):
        ret = list()
        for group in self.grouped_dimensions:
            for symbol in group:
                ret.append(symbol)
        return ret
        
    def has_same_dimensions(one, another):                
        if not isinstance(one, Shape):
            return False
        if not isinstance(another, Shape):
            return False
        
        one_dimensions_flatten = one.flatten_dimensions()
        another_dimensions_flatten = another.flatten_dimensions()
        
        for symbol in one_dimensions_flatten:
            if not symbol in another_dimensions_flatten:
                return False
            another_dimensions_flatten.remove(symbol)
        return len(another_dimensions_flatten) == 0
    
    def has_same_shape(one, another):
        if not isinstance(one, Shape):
            return False
        if not isinstance(another, Shape):
            return False
        
        if not len(one.grouped_dimensions) == len(another.grouped_dimensions):
            return False
        for one_group, another_group in zip(one.grouped_dimensions, another.grouped_dimensions):
            if not len(one_group) == len(another_group):
                return False
            for one_symbol, another_symbol in zip(one_group, another_group):
                if not one_symbol == another_symbol:
                    return False
                
        return True

    def copy(self):
        return Shape(self.grouped_dimensions, copy=True)
        