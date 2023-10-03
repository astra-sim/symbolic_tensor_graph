import pandas as pd
import numpy as np

class OffloadStrategy:
    def __init__(self, create_empty=False):
        if not create_empty:
            assert False
        self.offload_strategy = dict()
    
    def get_offload(self, tensor, restrict=True):
        if not tensor.id in self.offload_strategy:
            if restrict:
                assert False
            self.offload_strategy[tensor.id] = 0.0
        return self.offload_strategy[tensor.id]
    
    def set_offload(self, tensor, offload=1.0):
        self.offload_strategy[tensor.id] = offload
        
    def set_all_weight_offload(self, tensors, offload=1.0):
        for tensor in tensors:
            if tensor.require_grads:
                self.set_offload(tensor, offload)
                
    def set_all_intermediate_offload(self, tensors, offload=1.0):
        for tensor in tensors:
            if (not tensor.x1 is None) or (not tensor.x2 is None):
                self.set_offload(tensor, offload)
    
    def set_all_leaf_offload(self, tensors, offload=1.0):
        for tensor in tensors:
            if (tensor.x1 is None) and (tensor.x2 is None):
                self.set_offload(tensor, offload)
    
    @staticmethod
    def parse_records(csv_filename):
        offload_strategy = OffloadStrategy(True)
        df = pd.read_csv(csv_filename, encoding='utf-8', header=None)
        df = df.replace({np.nan:None})
        for i in range(df.shape[0]):
            data = np.array(df[i:i+1]).reshape(-1)
            offload_strategy.offload_strategy[data[0]] = data[1]
        return offload_strategy
            
    def to_records(self, csv_filename):
        data = list()
        for tensor_id, offload in self.offload_strategy.items():
            data.append([tensor_id, offload])
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, encoding='utf-8', header=None, index=None)
        return
    
    @staticmethod
    def create_blank(tensors):
        offload_strategy = OffloadStrategy(True)
        for tensor in tensors:
            offload_strategy.set_offload(tensor, 0)
        return offload_strategy
