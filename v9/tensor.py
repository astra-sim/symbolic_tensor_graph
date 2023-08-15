import sympy as sp
import numpy as np
import pandas as pd
import graphviz

class Tensor:
    def __init__(self, create_empty=False):
        if not create_empty:
            assert False        # not allow create empty tensor, need to parse from file, 
                                # here we impl something like private constructor
        self.id = None
        self.require_grads = None
        self.shape = None
        self.hidden = None
        self.x1 = None
        self.x2 = None
        self.op_type = None
        self.op_attr = None
        self.x1_shape = None
        self.x1_hidden = None
        self.x2_shape = None
        self.x2_hidden = None
        self.direct_output_shape = None
        self.direct_output_hidden = None
        self.post_communications = None
        self.ops = None
        
    @staticmethod
    def parse_shape(shape):
        shape = str(shape)
        ret = list()
        terms = shape.strip().split(",")
        for term in terms:
            ret.append(Tensor.parse_expr(term))
        return ret
    
    @staticmethod
    def stringfy_shape(shape):
        ret = ""
        for term in shape:
            ret += str(term) + ","
        ret = ret[:-1]
        if len(ret) == 0:
            ret = '1'
        return ret
    
    @staticmethod
    def parse_expr(expr):
        print(expr.strip())
        return sp.parse_expr(expr.strip())
    
    @staticmethod
    def parse_record(terms):
        assert len(terms) == 16
        tensor = Tensor(create_empty=True)
        tensor.id = terms[0]
        tensor.require_grads = (terms[1].strip()=='Y')
        tensor.shape = Tensor.parse_shape(terms[2])
        tensor.hidden = Tensor.parse_shape(terms[3])
        tensor.x1 = terms[4]
        tensor.x2 = terms[5]
        tensor.op_type = terms[6]
        tensor.op_attr = terms[7] if not terms[8] is None else ""
        if tensor.x1 is not None:
            tensor.x1_shape = Tensor.parse_shape(terms[8])
            tensor.x1_hidden = Tensor.parse_shape(terms[9])
        if tensor.x2 is not None:
            tensor.x2_shape = Tensor.parse_shape(terms[10])
            tensor.x2_hidden = Tensor.parse_shape(terms[11])
        tensor.direct_output_shape = Tensor.parse_shape(terms[12])
        tensor.direct_output_hidden = Tensor.parse_shape(terms[13])
        tensor.post_communications = terms[14]
        tensor.ops = Tensor.parse_expr(terms[15]) if not terms[15] is None else 0
        return tensor
    
    def to_record(tensor):
        terms = list()
        terms.append(tensor.id)
        terms.append('Y' if tensor.require_grads else 'N')
        terms.append(Tensor.stringfy_shape(tensor.shape))
        terms.append(Tensor.stringfy_shape(tensor.hidden))
        terms.append(tensor.x1 if not tensor.x1 is None else "")
        terms.append(tensor.x2 if not tensor.x2 is None else "")
        terms.append(tensor.op_type)
        terms.append(tensor.op_attr)
        if tensor.x1 is not None:
            terms.append(Tensor.stringfy_shape(tensor.x1_shape))
            terms.append(Tensor.stringfy_shape(tensor.x1_hidden))
        else:
            terms.append("")
            terms.append("")
        if tensor.x2 is not None:
            terms.append(Tensor.stringfy_shape(tensor.x2_shape))
            terms.append(Tensor.stringfy_shape(tensor.x2_hidden))
        else:
            terms.append("")
            terms.append("")
        terms.append(Tensor.stringfy_shape(tensor.direct_output_shape))
        terms.append(Tensor.stringfy_shape(tensor.direct_output_hidden))
        terms.append(tensor.post_communications if not tensor.post_communications is None else "")
        terms.append(tensor.ops if not tensor.ops==0 else "")
        return terms

    @staticmethod
    def parse_records(csv_filename):
        df = pd.read_csv(csv_filename, encoding='utf-8', header=None)
        df = df.replace({np.nan:None})
        tensors = list()
        for i in range(df.shape[0]):
            data = np.array(df[i:i+1]).reshape(-1)
            tensors.append(Tensor.parse_record(data))
        return tensors
    
    @staticmethod
    def to_records(tensors, csv_filename):
        data = list()
        for tensor in tensors:
            data.append(tensor.to_record())
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, encoding='utf-8', header=None, index=None)
        
    @staticmethod
    def visualize(tensors, filename, format='pdf'):
        f = graphviz.Digraph()
        for tensor in tensors:
            f.node(name=tensor.id, 
                   lable=tensor.id, 
                   id=tensor.id, 
                   shape="box")
            if tensor.x1 is not None:
                f.edge(tensor.x1, tensor.id)
            if tensor.x2 is not None:
                f.edge(tensor.x2, tensor.id)
        f.render(filename, format=format, cleanup=True)
