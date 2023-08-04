class Tensor:
    def __init__(self, shape_symbol):
        super(Tensor, self).__init__()        
        self.shape_symbol = tuple(shape_symbol)
        
    def get_shape_symbol(self):
        return self.shape_symbol
    
    def get_shape(self, symbol_size_table):
        ret = list()
        for symbol in self.shape_symbol:
            value = symbol
            if symbol in symbol_size_table:
                value = symbol_size_table[symbol]
            ret.append(value)
        return tuple(ret)
    
    def to_dict(self, symbol_size_table=None):
        ret = {
            "id": id(self),
            "shape_symbol": list(self.shape_symbol)
        }
        if symbol_size_table is not None:
            shape = self.get_shape(symbol_size_table)
            ret["shape"] = shape
        return ret

    @staticmethod
    def from_dict(data):
        id_ = data["id"]
        shape_symbols = list()
        for shape_symbol in ret["shape_symbol"]:
            shape_symbols.append(shape_symbol)
        return id_, Tensor(shape_symbols)
        