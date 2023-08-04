import random

class Tensor:
    used_dimensions_symbol = set()
    
    def __init__(self, dimensions_symbol, label="", require_grads=False):
        super(Tensor, self).__init__()        
        self.dimensions_symbol = tuple(dimensions_symbol)
        self.dimensions = tuple(dimensions_symbol)
        self.label = label
        
        # demonstrate the dimension changes
        self.parent = list()
        self.identical_dimensions = list()
        self.common_dimensions = list()
        self.merged_dimensions = list()
        self.op = ""

        self.dimension_constraints = dict()
        
        self.require_grads = require_grads
        for symbol in dimensions_symbol:
            if not symbol in Tensor.used_dimensions_symbol:
                Tensor.used_dimensions_symbol.add(symbol)
        
    def apply_dimension(self, symbol, value):
        dimensions = list(self.dimensions)
        for i, symbol_ in enumerate(self.dimensions_symbol):
            if symbol_ == symbol:
                dimensions[i] = value
        self.dimensions = tuple(dimensions)
        
    def apply_dimensions(self, symbol_value_dict):
        for symbol in symbol_value_dict.keys():
            value = symbol_value_dict[symbol]
            self.apply_dimension(symbol, value)
            
    def reset_dimension(self, symbol):
        self.apply_dimension(symbol, symbol)
    
    def reset_dimensions(self, symbols):
        for symbol in symbols:
            self.reset_dimension(symbol)
        
    @staticmethod
    def einsum(einsum, tensor_a, tensor_b, label=""):
        def _parse_einsum(einsum, input_tensors):
            # parse einsum expression
            terms = einsum.split("->")
            assert len(terms) == 2
            inputs, outputs = terms[0], terms[1]
            del terms
            
            inputs = inputs.split(",")
            assert len(input_tensors) == len(inputs)
        
            # find out how many chars appears in einsum expression
            symbol_set = list()
            for input_ in inputs:
                for char_ in input_:
                    if not char_ in symbol_set:
                        symbol_set.append(char_)
            for char_ in outputs:
                assert char_ in symbol_set
            
            # find corresponding symbol for each char, and category them
            seperated, common, merged = list(), list(), list()
            char_symbol_table = dict()
            for char_ in symbol_set:
                char_appear_count = 0
                symbol = None
                for input_, tensor in zip(inputs, input_tensors):
                    for char__, symbol_ in zip(input_, tensor.dimensions_symbol):
                        if char__ == char_:
                            char_appear_count += 1
                            if symbol == None:
                                symbol = symbol_
                            else:
                                assert symbol == symbol_
                if not char_ in outputs:
                    merged.append(symbol)
                elif char_appear_count > 1:
                    common.append(symbol)
                else:
                    seperated.append(symbol)
                char_symbol_table[char_] = symbol
            output_tensor_dimensions = list()
            for char_ in outputs:
                output_tensor_dimensions.append(char_symbol_table[char_])
            return seperated, common, merged, output_tensor_dimensions
        seperated, common, merged, output = _parse_einsum(einsum, (tensor_a, tensor_b))
        
        output_tensor = Tensor(output, label)
        output_tensor.parent.append(tensor_a)
        output_tensor.parent.append(tensor_b)
        output_tensor.identical_dimensions.extend(seperated)
        output_tensor.common_dimensions.extend(common)
        output_tensor.merged_dimensions.extend(merged)
        output_tensor.op = "einsum"
        return output_tensor

    @staticmethod
    def elementWise(tensor, label=""):
        output_tensor = Tensor(tensor.dimensions_symbol, label)
        output_tensor.parent.append(tensor)
        output_tensor.identical_dimensions.extend(list(tensor.dimensions_symbol))
        output_tensor.op = "element_wise"
        return output_tensor
    
    @staticmethod
    def reshape(tensor, new_dimension_symbols, label=""):
        new_dimension_symbols_reg = list()
        
        for dimension in new_dimension_symbols:

    def __str__(self):
        ret = f"label={self.label}, dimensions_symbol={self.dimensions_symbol}, identical={self.identical_dimensions}, common={self.common_dimensions}, merged={self.merged_dimensions}, parents=["
        for parent in self.parent:
            ret += f"{parent.label}@{parent.dimensions_symbol}, "
        ret += "]"
        return ret

    @staticmethod
    def get_new_dimension_symbol(prefix=""):
        symbol = prefix + "0"
        while symbol in Tensor.used_dimensions_symbol:
            num = random.randint(0, 1e5)
            symbol = f"{prefix}{num}"
        return symbol

        