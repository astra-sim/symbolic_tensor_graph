import os

from tensor import Tensor
from graph_linker import GraphLinker
from grad_updater import GradUpdater


def transformer_stack():
    mha_fwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/multiHeadAttentionFwd.csv')
    mha_bwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/multiHeadAttentionBwd.csv')
    ffn_fwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/feedForwardNetworkFwd.csv')
    ffn_bwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/feedForwardNetworkBwd.csv')
        
    mha_fwd = GraphLinker.prefix_graph_impl(mha_fwd, "mha")
    mha_bwd = GraphLinker.prefix_graph_impl(mha_bwd, "mha")
    ffn_fwd = GraphLinker.prefix_graph_impl(ffn_fwd, "ffn")
    ffn_bwd = GraphLinker.prefix_graph_impl(ffn_bwd, "ffn")
        
    Tensor.visualize(mha_fwd, f'sharding_spreadsheets/fsdp/visualization/mha_fwd')
    Tensor.visualize(mha_bwd, f'sharding_spreadsheets/fsdp/visualization/mha_bwd')
    Tensor.visualize(ffn_fwd, f'sharding_spreadsheets/fsdp/visualization/ffn_fwd')
    Tensor.visualize(ffn_bwd, f'sharding_spreadsheets/fsdp/visualization/ffn_bwd')
        
    fwd = GraphLinker.link_graph_impl([mha_fwd, ffn_fwd], {"mha_norm": "ffn_x0"})
    bwd = GraphLinker.link_graph_impl([mha_bwd, ffn_bwd], {"ffn_d_x0": "mha_d_norm"})
    
    Tensor.visualize(fwd, 'sharding_spreadsheets/fsdp/visualization/stack_fwd')
    Tensor.visualize(bwd, 'sharding_spreadsheets/fsdp/visualization/stack_bwd')
    
    Tensor.to_records(fwd, 'sharding_spreadsheets/fsdp/processed_graphs/stackFwd.csv')
    Tensor.to_records(bwd, 'sharding_spreadsheets/fsdp/processed_graphs/stackBwd.csv')
    return 


def transformer_stacks(num_stacks):
    fwd_graphs = list()
    bwd_graphs = list()
    for i in range(num_stacks):
        if not os.path.exists('sharding_spreadsheets/fsdp/processed_graphs/stackFwd.csv'):
            transformer_stack()
        assert os.path.exists('sharding_spreadsheets/fsdp/processed_graphs/stackBwd.csv')
        fwd = Tensor.parse_records('sharding_spreadsheets/fsdp/processed_graphs/stackFwd.csv')
        bwd = Tensor.parse_records('sharding_spreadsheets/fsdp/processed_graphs/stackBwd.csv')
        
        fwd = GraphLinker.prefix_graph_impl(fwd, f"stack{i}")
        bwd = GraphLinker.prefix_graph_impl(bwd, f"stack{i}")
        fwd_graphs.append(fwd)
        bwd_graphs.append(bwd)
        
    fwd_links = dict()
    bwd_links = dict()
    for i in range(num_stacks-1):
        fwd_links[f"stack{i}_ffn_norm"] = f"stack{i+1}_mha_x"
        bwd_links[f"stack{i+1}_mha_d_x"] = f"stack{i}_ffn_d_norm"
    fwd = GraphLinker.link_graph_impl(fwd_graphs, fwd_links)
    bwd = GraphLinker.link_graph_impl(bwd_graphs, bwd_links)
    
    Tensor.visualize(fwd, f'sharding_spreadsheets/fsdp/visualization/stack_{num_stacks}_fwd_stacks')
    Tensor.visualize(bwd, f'sharding_spreadsheets/fsdp/visualization/stack_{num_stacks}_bwd_stacks')
    Tensor.to_records(fwd, f'sharding_spreadsheets/fsdp/processed_graphs/stack{num_stacks}Fwd.csv')
    Tensor.to_records(bwd, f'sharding_spreadsheets/fsdp/processed_graphs/stack{num_stacks}Bwd.csv')


def transformer(num_stacks):
    if not os.path.exists(f'sharding_spreadsheets/fsdp/processed_graphs/stack{num_stacks}Fwd.csv'):
        transformer_stacks(num_stacks)
    assert os.path.exists(f'sharding_spreadsheets/fsdp/processed_graphs/stack{num_stacks}Bwd.csv')
    stacks_fwd = Tensor.parse_records(f'sharding_spreadsheets/fsdp/processed_graphs/stack{num_stacks}Fwd.csv')
    stacks_bwd = Tensor.parse_records(f'sharding_spreadsheets/fsdp/processed_graphs/stack{num_stacks}Bwd.csv')
    in_embed_fwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/inEmbedFwd.csv')
    in_embed_bwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/inEmbedBwd.csv')
    out_embed_fwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/outEmbedFwd.csv')
    out_embed_bwd = Tensor.parse_records('sharding_spreadsheets/fsdp/graphs/outEmbedBwd.csv')
    
    fwd_links = dict()
    bwd_links = dict()
    fwd_links['inEmbY'] = 'stack0_mha_x'
    fwd_links[f'stack{num_stacks-1}_ffn_norm'] = 'outEmbedX'
    bwd_links['d_outEmbedX'] = f'stack{num_stacks-1}_ffn_d_norm'
    bwd_links['stack0_mha_d_x'] = 'd_inEmbY'
    fwd = GraphLinker.link_graph_impl([stacks_fwd, in_embed_fwd, out_embed_fwd], fwd_links)
    bwd = GraphLinker.link_graph_impl([stacks_bwd, in_embed_bwd, out_embed_bwd], bwd_links)
    Tensor.visualize(fwd, f'sharding_spreadsheets/fsdp/visualization/transformer_{num_stacks}_fwd')
    Tensor.visualize(bwd, f'sharding_spreadsheets/fsdp/visualization/transformer_{num_stacks}_bwd')
    Tensor.to_records(fwd, f'sharding_spreadsheets/fsdp/processed_graphs/transformer_{num_stacks}_fwd.csv')
    Tensor.to_records(bwd, f'sharding_spreadsheets/fsdp/processed_graphs/transformer_{num_stacks}_bwd.csv')
    
    grad_updater = GradUpdater(fwd, bwd)
    update_tensors = grad_updater.update_tensors()
    loop_links = dict()
    loop_links['outEmbedY'] = 'd_outEmbedY'
    loop = GraphLinker.link_graph_impl([fwd, bwd, update_tensors], loop_links)
    # Tensor.visualize(loop, f"sharding_spreadsheets/dp/visualization/transformer_{num_stacks}")
    Tensor.to_records(loop, f"sharding_spreadsheets/fsdp/processed_graphs/transformer_{num_stacks}.csv")
    
    return
    
    
if __name__ == '__main__':
    transformer(2)
    transformer(8)
    transformer(16)
    transformer(32)
    transformer(64)
    transformer(96)
