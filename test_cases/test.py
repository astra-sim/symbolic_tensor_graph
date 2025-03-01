import sympy as sp
from symbolic_tensor_graph.graph.graph import TensorGraph
from symbolic_tensor_graph.graph.coll_comm_matcher import CommunicationMatcher
from symbolic_tensor_graph.graph.grad_updater import GradUpdater
from symbolic_tensor_graph.graph.convert_chakra import (
    ConvertChakra,
    BundledConvertChakra,
)
from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph
from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
from symbolic_tensor_graph.graph.graph_distributer import GraphDistributer
from symbolic_tensor_graph.chakra.node import Node
from symbolic_tensor_graph.chakra.backends.json_backend import JsonBackend
from models.transformer import (
    transformer as transformer_fn,
    transformer_stack as transformer_stack_fn,
    # transformer_stacks as transformer_fn,
)


def test1():
    paths = [
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv",
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv",
        "./sharding_spreadsheets/module/divya/reshape.csv",
        "./sharding_spreadsheets/module/divya/linear.csv",
    ]

    tp, dp = sp.symbols("tp dp")
    parallel_dims = [tp, dp]

    for path in paths:
        print(path)
        graph = TensorGraph.load_tensor_graph(path)
        for tensor in graph.tensors:
            print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
            if tensor.x1 is not None:
                from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
                to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
                print(
                    f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
                )
            if tensor.x2 is not None:
                from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
                to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
                print(
                    f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
                )
        updated_graph = GradUpdater.apply(graph)
        for tensor in updated_graph.tensors:
            print(tensor)


def test2():
    dp, tp = sp.symbols("dp tp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 32,
        tp: 32,
    }

    parallel_dims = [dp, tp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, 2)
    transformer = transformer_fn(stack, 2)
    transformer_updated_grad = GradUpdater.apply(transformer)
    for tensor in transformer_updated_grad.tensors:
        print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
        if tensor.x1 is not None:
            from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
            to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
            print(
                f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )
        if tensor.x2 is not None:
            from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
            to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
            print(
                f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )

    chakra_nodes = ConvertChakra.apply(
        transformer_updated_grad, symbol_map_value, parallel_dims
    )

    nodes = chakra_nodes.get_nodes()
    Node.readout_nodes(nodes, "test.0.eg")
    Node.readout_nodes(nodes, "test.json", backend=JsonBackend)


def test3():
    dp, tp = sp.symbols("dp tp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 1024,
        tp: 1,
    }

    parallel_dims = [dp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, 2)
    transformer = transformer_fn(stack, 2)
    transformer_updated_grad = GradUpdater.apply(transformer)
    for tensor in transformer_updated_grad.tensors:
        print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
        if tensor.x1 is not None:
            from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
            to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
            print(
                f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )
        if tensor.x2 is not None:
            from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
            to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
            print(
                f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )

    chakra_nodes = ConvertChakra.apply(
        transformer_updated_grad, symbol_map_value, parallel_dims
    )

    nodes = chakra_nodes.get_nodes()
    Node.readout_nodes(nodes, "test.0.eg")
    Node.readout_nodes(nodes, "test.json", backend=JsonBackend)


def test4():
    dp, tp = sp.symbols("dp tp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 32,
        tp: 32,
    }

    parallel_dims = [dp, tp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, 2)
    transformer = transformer_fn(stack, 2)
    transformer_updated_grad = GradUpdater.apply(transformer)
    transformer_updated_grad_c2 = ReplicateGraph.apply(
        transformer_updated_grad, new_revision=lambda old_revision: f"c2&{old_revision}"
    )
    transformer_updated_grad_2 = ConnectGraph.apply(
        [transformer_updated_grad, transformer_updated_grad_c2], {}
    )
    for tensor in transformer_updated_grad_2.tensors:
        print(f"{tensor.id} {tensor.y_shape}, {tensor.y_hidden}")
        if tensor.x1 is not None:
            from_shape, from_hidden = tensor.x1.y_shape, tensor.x1.y_hidden
            to_shape, to_hidden = tensor.x1_shape, tensor.x1_hidden
            print(
                f"{tensor.id} x1 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )
        if tensor.x2 is not None:
            from_shape, from_hidden = tensor.x2.y_shape, tensor.x2.y_hidden
            to_shape, to_hidden = tensor.x2_shape, tensor.x2_hidden
            print(
                f"{tensor.id} x2 {from_shape}@{from_hidden} => {to_shape}@{to_hidden} == {CommunicationMatcher.match_comms(from_shape, from_hidden, to_shape, to_hidden, parallel_dims)}"
            )

    chakra_nodes = ConvertChakra.apply(
        transformer_updated_grad_2, symbol_map_value, parallel_dims
    )

    nodes = chakra_nodes.get_nodes()
    Node.readout_nodes(nodes, "test.0.eg")
    Node.readout_nodes(nodes, "test.json", backend=JsonBackend)


def test5():
    dp, tp, pp = sp.symbols("dp tp pp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 4,
        tp: 4,
        pp: 2,
    }
    spatial_parallel_dims = [dp, tp]
    temporal_parallel_dims = [pp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, 2)
    transformer = transformer_fn(stack, 2)
    transformer_updated_grad = GradUpdater.apply(transformer)
    split_tensor1 = list()
    for tensor in transformer_updated_grad.tensors:
        if "stack_1_mha_x" in tensor.id:
            split_tensor1.append(tensor)
    split_tensor2 = list()
    for tensor in transformer_updated_grad.tensors:
        if "stack_1_mha_d_x" in tensor.id:
            split_tensor2.append(tensor)
    upper_graph1, lower_graph, split_shadow1 = GraphSpliter.apply(
        transformer_updated_grad, split_tensor1
    )

    lower_graph, upper_graph2, split_shadow2 = GraphSpliter.apply(
        lower_graph, split_tensor2
    )

    upper_graph = ConnectGraph([upper_graph1, upper_graph2], {})

    print("upper_graph1")
    for tensor in upper_graph1.tensors:
        print(tensor)
    print("upper_graph2")
    for tensor in upper_graph2.tensors:
        print(tensor)
    print("lower graph")
    for tensor in lower_graph.tensors:
        print(tensor)
    print("split_shadow1")
    print(split_shadow1)
    print("split_shadow2")
    print(split_shadow2)

    hybrid_graph_upper = ConvertChakra.apply(
        upper_graph, symbol_map_value, spatial_parallel_dims
    )
    hybrid_graph_lower = ConvertChakra.apply(
        lower_graph, symbol_map_value, spatial_parallel_dims
    )
    nodes = hybrid_graph_upper.get_nodes()
    Node.readout_nodes(nodes, "test_upper.0.eg")
    Node.readout_nodes(nodes, "test_upper.json", backend=JsonBackend)
    nodes = hybrid_graph_lower.get_nodes()
    Node.readout_nodes(nodes, "test_lower.0.eg")
    Node.readout_nodes(nodes, "test_lower.json", backend=JsonBackend)


def test6():
    num_stacks = 2
    dp, tp, pp = sp.symbols("dp tp pp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {
        Din: 51200,
        Dout: 25600,
        Dmodel: 25600,
        Dff: 25600 * 4,
        Batch: 1024,
        Seq: 1024,
        Head: 1024,
        dp: 8,
        tp: 8,
        pp: 1,
    }
    spatial_parallel_dims = [dp, tp]
    temporal_parallel_dims = [pp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/divya/feed_forward_network.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    # transformer = transformer_fn(in_emb, out_emb, stack, num_stacks)
    transformer = transformer_fn(stack, num_stacks)
    transformer_updated_grad = GradUpdater.apply(transformer)

    def _create_tensor_map(_tensors, _temporal_parallel_dims, _symbol_map_value):
        _tensor_map = dict()
        assert len(_temporal_parallel_dims) == 1
        parallel_dim = _temporal_parallel_dims[0]
        range_ = _symbol_map_value[parallel_dim]
        for tensor in _tensors:
            for num_stack in range(num_stacks):
                if f"stack_{num_stack}" in tensor.id:
                    _tensor_map[tensor.id] = {parallel_dim: (num_stack+1) % range_}
                    break
            if "in_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: 0}
            elif "out_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: (num_stacks+1) % range_}
        return _tensor_map

    hook = 0
    tensor_map = _create_tensor_map(
        transformer_updated_grad.tensors, temporal_parallel_dims, symbol_map_value
    )
    bundled_graph = GraphDistributer.apply(
        transformer_updated_grad,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        tensor_map,
    )
    hook = 1
    bundled_graph.remote_parent_shadow_pairs = []
    bundled_hybrid_graph = BundledConvertChakra.apply(bundled_graph, symbol_map_value)
    hook = 2
    bundled_hybrid_graph.readout("transformer_2stack.%d.eg")
    bundled_hybrid_graph.readout("transformer_2stack.%d.json", backend=JsonBackend)
    hook = 3


def test7(symbol_map_value, output_filename):
    num_stacks = 2
    dp, tp, pp = sp.symbols("dp tp pp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )

    spatial_parallel_dims = [dp, tp]
    temporal_parallel_dims = [pp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    transformer = transformer_fn(in_emb, out_emb, stack, num_stacks)
    # transformer = transformer_fn(stack, num_stacks)
    transformer_updated_grad = GradUpdater.apply(transformer)

    def _create_tensor_map(_tensors, _temporal_parallel_dims, _symbol_map_value):
        _tensor_map = dict()
        assert len(_temporal_parallel_dims) == 1
        parallel_dim = _temporal_parallel_dims[0]
        range_ = _symbol_map_value[parallel_dim]
        for tensor in _tensors:
            for num_stack in range(num_stacks):
                if f"stack_{num_stack}" in tensor.id:
                    _tensor_map[tensor.id] = {parallel_dim: (num_stack+1) % range_}
                    break
            if "in_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: 0}
            elif "out_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: (num_stacks+1) % range_}
        return _tensor_map
    hook = 0
    tensor_map = _create_tensor_map(
        transformer_updated_grad.tensors, temporal_parallel_dims, symbol_map_value
    )
    bundled_graph = GraphDistributer.apply(
        transformer_updated_grad,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        tensor_map,
    )
    hook = 1
    bundled_graph.remote_parent_shadow_pairs = []
    bundled_hybrid_graph = BundledConvertChakra.apply(bundled_graph, symbol_map_value)
    hook = 2
    bundled_hybrid_graph.readout(output_filename)
    hook = 3


def test_comm_group():
    num_stacks = 2
    dp, tp, pp, ssp = sp.symbols("dp tp pp sp")
    Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
        "Din Dout Dmodel Dff Batch Seq Head"
    )
    symbol_map_value = {dp: 2, tp: 3, pp: 5, ssp: 1}

    spatial_parallel_dims = [dp, tp, ssp]
    temporal_parallel_dims = [pp]
    mha = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/multi_head_attention.csv"
    )
    ffn = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/feed_forward_network.csv"
    )
    in_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/embedding.csv"
    )
    out_emb = TensorGraph.load_tensor_graph(
        "./sharding_spreadsheets/module/fully_sharded_divya/embedding.csv"
    )
    stack = transformer_stack_fn(mha, ffn)
    transformer = transformer_fn(in_emb, out_emb, stack, num_stacks)
    # transformer = transformer_fn(stack, num_stacks)
    transformer_updated_grad = GradUpdater.apply(transformer)

    def _create_tensor_map(_tensors, _temporal_parallel_dims, _symbol_map_value):
        _tensor_map = dict()
        assert len(_temporal_parallel_dims) == 1
        parallel_dim = _temporal_parallel_dims[0]
        range_ = _symbol_map_value[parallel_dim]
        for tensor in _tensors:
            for num_stack in range(num_stacks):
                if f"stack_{num_stack}" in tensor.id:
                    _tensor_map[tensor.id] = {parallel_dim: (num_stack+1) % range_}
                    break
            if "in_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: 0}
            elif "out_emb" in tensor.id:
                _tensor_map[tensor.id] = {parallel_dim: (num_stacks+1) % range_}
        return _tensor_map
    hook = 0
    tensor_map = _create_tensor_map(
        transformer_updated_grad.tensors, temporal_parallel_dims, symbol_map_value
    )
    bundled_graph = GraphDistributer.apply(
        transformer_updated_grad,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        tensor_map,
    )
    
    comm_groups = GraphDistributer._create_comm_groups(spatial_parallel_dims, temporal_parallel_dims, symbol_map_value)
    hook = 1
    for graph_key in comm_groups.keys():
        print(f"{graph_key}: {comm_groups[graph_key]}")
        
    GraphDistributer._distribute_comm_groups(bundled_graph.graphs, comm_groups, spatial_parallel_dims)
    for graph_key in bundled_graph.graphs.keys():
        print(f"{graph_key}")
        comm_groups = bundled_graph.graphs[graph_key].comm_groups
        for key in comm_groups:
            print(f"\t{key}: {comm_groups[key]}")
    hook = 2

if __name__ == "__main__":
    test_comm_group()
    # import os

    # generated_root = ""
    # dp, tp, pp = sp.symbols("dp tp pp")
    # Din, Dout, Dmodel, Dff, Batch, Seq, Head = sp.symbols(
    #     "Din Dout Dmodel Dff Batch Seq Head"
    # )
    # symbol_map_value = {
    #     Din: 51200,
    #     Dout: 25600,
    #     Dmodel: 25600,
    #     Dff: 25600 * 4,
    #     Batch: 1024,
    #     Seq: 1024,
    #     Head: 1024,
    #     dp: 1,
    #     tp: 64,
    #     pp: 1,
    # }
    # # pp has to be one as there is bug in astrasim of send/recv pairs between stages of pipeline.
    # test7(
    #     symbol_map_value,
    #     os.path.join(
    #         generated_root,
    #         "workload_fs/transformer_2stack_dp1_mp64/transformer_2stack_dp1_mp64.%d.et",
    #     ),
    # )
    # symbol_map_value[dp] = 2
    # symbol_map_value[tp] = 32
    # test7(
    #     symbol_map_value,
    #     os.path.join(
    #         generated_root,
    #         "workload_fs/transformer_2stack_dp2_mp32/transformer_2stack_dp2_mp32.%d.et",
    #     ),
    # )
    # symbol_map_value[dp] = 4
    # symbol_map_value[tp] = 16
    # test7(
    #     symbol_map_value,
    #     os.path.join(
    #         generated_root,
    #         "workload_fs/transformer_2stack_dp4_mp16/transformer_2stack_dp4_mp16.%d.et",
    #     ),
    # )
    # symbol_map_value[dp] = 8
    # symbol_map_value[tp] = 8
    # test7(
    #     symbol_map_value,
    #     os.path.join(
    #         generated_root,
    #         "workload_fs/transformer_2stack_dp8_mp8/transformer_2stack_dp8_mp8.%d.et",
    #     ),
    # )
    # symbol_map_value[dp] = 16
    # symbol_map_value[tp] = 4
    # test7(
    #     symbol_map_value,
    #     os.path.join(
    #         generated_root,
    #         "workload_fs/transformer_2stack_dp16_mp4/transformer_2stack_dp16_mp4.%d.et",
    #     ),
    # )
    # symbol_map_value[dp] = 32
    # symbol_map_value[tp] = 2
    # test7(
    #     symbol_map_value,
    #     os.path.join(
    #         generated_root,
    #         "workload_fs/transformer_2stack_dp32_mp2/transformer_2stack_dp32_mp2.%d.et",
    #     ),
    # )
    # symbol_map_value[dp] = 64
    # symbol_map_value[tp] = 1
    # test7(
    #     symbol_map_value,
    #     os.path.join(
    #         generated_root,
    #         "workload_fs/transformer_2stack_dp64_mp1/transformer_2stack_dp64_mp1.%d.et",
    #     ),
    # )
