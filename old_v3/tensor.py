import random

used_dimension_symbols = set()


def get_new_dimension(prefix="sym"):
    prefix += "_"

    postfix = 0
    symbol = f"{prefix}{postfix}"
    while symbol in used_dimension_symbols:
        postfix += 1
        symbol = f"{prefix}{postfix}"
    return symbol
        

class Tensor:
    def __init__(self, dimensions_symbols):
        super(Tensor, self).__init__()
        self.dimensions_symbols = tuple(dimensions_symbols)
        self.parent = list()
        self.dimensions_constraints = list()
        for symbol in dimensions_symbols:
            used_dimension_symbols.append(symbol)
        