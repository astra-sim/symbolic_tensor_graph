class OP:
    def __init__(self, input_tensors, output_tensor):
        super(OP, self).__init__()
        self.input_tensors = input_tensors
        self.output_tensor = output_tensor
    
    def to_dict(self, dict_=None):
        if dict_ is None:
            dict_ = dict()
        input_tensors = list()
        for input_tensor in self.input_tensors:
            input_tensors.append(id(input_tensor))
        dict_["input_tensors"] = input_tensors
        dict_["output_tensor"] = id(output_tensor)
        return dict_
    
    @staticmethod
    def from_dict(data, tensors, op=None):
        input_tensors = list()
        for tensor_id in data['input_tensors']:
            tensor_id = int(tensor_id)
            assert tensor_id in tensors
            input_tensors.append(tensors[tensor_id])
        tensor_id = data['output_tensor']
        assert tensor_id in tensors
        output_tensor = tensors[tensor_id]
        if op is None:
            op = OP(None, None)
        op.input_tensors = input_tensors
        op.output_tensor = output_tensor
        return op

    
class Einsum(OP):
    def __init__(self, einsum, *input_tensors):
        seperated, common, merged = Einsum._parse_einsum(einsum, input_tensors)
        self.seperated = seperated
        self.common = common
        self.merged = merged
        
        output_tensor = Einsum._get_output_tensor(seperated, common, merged)
        super(Einsum, self).__init__(input_tensors, output_tensor)
        
    @staticmethod
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
                if not char_ in symbol_set
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
                for char__, symbol_ in zip(input_, tensor):
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
        return seperated, common, merged

    @staticmethod
    def _get_output_tensor(seperated, common, merged):
        output_dimensions = list()
        for dimension in seperated:
            output_dimensions.append(dimension)
        for dimension in common:
            output_dimensions.append(dimension)
        return Tensor(new_tensor_symbol)

    def to_dict(self, dict_=None):
        dict_ = super(Einsum, self).to_dict(dict_)
        dict_['type'] = "Einsum"
        

class ElementWise(OP):
    def __init__(self, input_tensor):
        output_tensor = Tensor(input_tensor.get_shape_symbol())
        super(ElementWise, self).__init__((input_tensor,), output_tensor)
        
