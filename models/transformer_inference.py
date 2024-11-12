from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.connect_graph import ConnectGraph

raise NotImplementedError()
def transformer(prefilling_transformer, decoding_transformer, num_decoding):
    
    transformer_steps = list()
    
    transformer_steps.append(ReplicateGraph.apply(
        prefilling_transformer, f"prefilling_%s"
    ))
    
    links = dict()
    
    for i in range(num_decoding):
        this_step = ReplicateGraph.apply(
            decoding_transformer, f"decoding_{i}_%s", old_symbol_map_new_symbol={"Seq": f"Seq+{i}"})
        transformer_steps.append(this_step)
        
        
        
