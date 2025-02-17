import sympy as sp
from ..tensor import Tensor
from .replicate_graph import ReplicateGraph
from .connect_graph import ConnectGraph
from ..ops import Add


def naive_pipeline_emb_separate_n_layer_each_stage(graph, temporal_parallel_dims, symbol_map_value, num_stacks, layer_each_stage=1):
    tensors = graph.tensors
    tensor_map = dict()
    assert len(temporal_parallel_dims) == 1
    parallel_dim = temporal_parallel_dims[0]
    pp_size = symbol_map_value[parallel_dim]
    for tensor in tensors:
        for num_stack in range(num_stacks):
            if f"stack_{num_stack}_" in tensor.id:
                tensor_map[tensor.id] = {parallel_dim: ((num_stack+1)//layer_each_stage) % pp_size}
                break
        if "in_emb" in tensor.id:
            tensor_map[tensor.id] = {parallel_dim: 0}
        elif "out_emb" in tensor.id:
            tensor_map[tensor.id] = {parallel_dim: ((num_stacks+2)//layer_each_stage) % pp_size}
    return graph, tensor_map


def naive_pipeline_emb_separate_evenly(graph, temporal_parallel_dims, symbol_map_value, num_stacks):
    assert len(temporal_parallel_dims) == 1
    parallel_dim = temporal_parallel_dims[0]
    pp_size = symbol_map_value[parallel_dim]
    layer_each_stage = (num_stacks+pp_size-1) // pp_size
    return naive_pipeline_emb_separate_n_layer_each_stage(graph, temporal_parallel_dims, symbol_map_value, num_stacks, layer_each_stage)


def naive_pipeline_n_layer_each_stage(graph, temporal_parallel_dims, symbol_map_value, num_stacks, layer_each_stage=1):
    tensors = graph.tensors
    tensor_map = dict()
    assert len(temporal_parallel_dims) == 1
    parallel_dim = temporal_parallel_dims[0]
    pp_size = symbol_map_value[parallel_dim]
    for tensor in tensors:
        for num_stack in range(num_stacks):
            if f"stack_{num_stack}_" in tensor.id:
                tensor_map[tensor.id] = {parallel_dim: (num_stack//layer_each_stage) % pp_size}
                break
        if "in_emb" in tensor.id:
            tensor_map[tensor.id] = {parallel_dim: 0}
        elif "out_emb" in tensor.id:
            tensor_map[tensor.id] = {parallel_dim: 0}
    return graph, tensor_map

def naive_pipeline_evenly(graph, temporal_parallel_dims, symbol_map_value, num_stacks):
    assert len(temporal_parallel_dims) == 1
    parallel_dim = temporal_parallel_dims[0]
    pp_size = symbol_map_value[parallel_dim]
    layer_each_stage = (num_stacks+pp_size-1) // pp_size
    return naive_pipeline_n_layer_each_stage(graph, temporal_parallel_dims, symbol_map_value, num_stacks, layer_each_stage)

def gpipe_pipeline_prepare(graph, symbol_map_value):
    micro_batch_sym = sp.symbols("MicroBatch")
    batch_sym = sp.symbols("Batch")
    assert micro_batch_sym in symbol_map_value
    assert batch_sym in symbol_map_value
    # micro_batches = (symbol_map_value[batch]+symbol_map_value[micro_batch]-1) // symbol_map_value[micro_batch]
    micro_batches = symbol_map_value[micro_batch_sym]
    # TODO: here it modify batch value in the symbol_map_value, however, idealy should not because it might be used multiple times.
    # symbol_map_value[batch_sym] = symbol_map_value[batch_sym] // micro_batches
    micro_batch_graphs = list()
    for i in range(micro_batches):
        micro_batch_graph = ReplicateGraph.apply(graph, f"mb{i}_%s")
        micro_batch_graphs.append(micro_batch_graph)
    no_microbatch_tensors = {"in_emb", "out_emb", "mha_wq", "mha_wk", "mha_wv", "mha_wo", "ffn_w1", "ffn_w2", "ffn_wo"}
    for i, graph in enumerate(micro_batch_graphs):
        for tensor in graph.tensors:
            no_microbatch = False
            for no_microbatch_layer in no_microbatch_tensors:
                if no_microbatch_layer in tensor.id:
                    no_microbatch = True
            if not no_microbatch:
                continue
            tensor.name = tensor.name.replace(f"mb{i}_", "")
    merged_graph = ConnectGraph.apply(micro_batch_graphs, dict())
    merged_graph = ReplicateGraph.apply(merged_graph, "%s", old_symbol_map_new_symbol={batch_sym: batch_sym/micro_batch_sym})
    
    tensor_id_map_tensor = merged_graph.get_tensor_id_map_tensor()
    microbatch_grads = dict()
    for tensor_id in tensor_id_map_tensor:
        tensor = tensor_id_map_tensor[tensor_id]
        if tensor.grad_of is not None and tensor.grad_of.require_grads:
            grad_of_id = tensor.grad_of.id
            if grad_of_id.startswith("mb"):
                foo = grad_of_id[grad_of_id.find("mb")+len("mb"):]
                foo = foo[foo.find("_")+1:]
                grad_of_id = grad_of_id[:grad_of_id.find("mb")] + foo
                assert grad_of_id in tensor_id_map_tensor
            tensor.grad_of._grad = None
            tensor.grad_of = None
            if not grad_of_id in microbatch_grads:
                microbatch_grads[grad_of_id] = list()
            microbatch_grads[grad_of_id].append(tensor)
    for tensor_id in microbatch_grads:
        from_ = microbatch_grads[tensor_id][0]
        for i, sub_grads in enumerate(microbatch_grads[tensor_id]):
            if i == 0:
                continue
            new_tensor = Tensor(create_empty=True)
            new_tensor.name = f"{sub_grads.name}_add_mb{i}"
            new_tensor.revision = sub_grads.revision
            new_tensor.op_type = Add.type_name
            new_tensor.x1 = from_
            new_tensor.x2 = sub_grads
            new_tensor.x1_shape = from_.y_shape
            new_tensor.x2_shape = sub_grads.y_shape
            new_tensor.x1_hidden = from_.y_hidden
            new_tensor.x2_hidden = sub_grads.y_hidden
            new_tensor.op_attr = None
            new_tensor.grad_of = None
            new_tensor._grad = None
            merged_graph.tensors.append(new_tensor)
            from_ = new_tensor
        if len(microbatch_grads[tensor_id]) > 1:
            merged_graph.out_tensors.append(from_)
        from_.grad_of = tensor_id_map_tensor[tensor_id]
        tensor_id_map_tensor[tensor_id]._grad = from_
    return merged_graph


def gpipe_n_layer_each_stage(graph, temporal_parallel_dims, symbol_map_value, num_stacks, layer_each_stage=1):
    graph = gpipe_pipeline_prepare(graph, symbol_map_value)
    tensors = graph.tensors
    tensor_map = dict()
    assert len(temporal_parallel_dims) == 1
    parallel_dim = temporal_parallel_dims[0]
    pp_size = symbol_map_value[parallel_dim]
    for tensor in tensors:
        for num_stack in range(num_stacks):
            if f"stack_{num_stack}_" in tensor.id:
                tensor_map[tensor.id] = {parallel_dim: (num_stack//layer_each_stage) % pp_size}
                break
        if "in_emb" in tensor.id:
            tensor_map[tensor.id] = {parallel_dim: 0}
        elif "out_emb" in tensor.id:
            tensor_map[tensor.id] = {parallel_dim: 0}
    return graph, tensor_map


def gpipe_evenly(graph, temporal_parallel_dims, symbol_map_value, num_stacks):
    assert len(temporal_parallel_dims) == 1
    parallel_dim = temporal_parallel_dims[0]
    pp_size = symbol_map_value[parallel_dim]
    layer_each_stage = (num_stacks+pp_size-1) // pp_size
    return gpipe_n_layer_each_stage(graph, temporal_parallel_dims, symbol_map_value, num_stacks, layer_each_stage)
