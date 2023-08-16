from tensor import Tensor
import re


class GraphLinker:
    def __init__(self, linker_file):
        self.linker_file = linker_file
        self.graphs = dict()
        pass
    
    def load_graph(self, cmd):
        # TODO:
        raise NotImplementedError()
    
    def save_graph(self, cmd):
        # TODO:
        raise NotImplementedError()
    
    def prefix_graph(self, cmd):
        def _parse_terms(cmd):
            op_name_pattern = re.compile('[ \t\r\n]*prefix[ \t\r\n]+')
            graphs_pattern = re.compile('[ \t\r\n]*[^ \t\r\n]+[ \t\r\n]+')
            prefix_pattern = re.compile('[ \t\r\n]*[^ \t\r\n]+[ \t\r\n;]+')
            
            op_name = op_name_pattern.match(cmd)
            assert not op_name is None
            cmd = cmd[op_name.end():]
            graph_name = graphs_pattern.match(cmd)
            assert not graph_name is None
            cmd = cmd[graph_name.end():]
            prefix_name = prefix_pattern.match(cmd)
            assert not prefix_name is None
            graph_name = graph_name.string[graph_name.start():graph_name.end()].strip()
            prefix_name = prefix_name.string[prefix_name.start():prefix_name.end()].replace(";", "").strip()
            return graph_name, prefix_name
                        
        graph_name, prefix_name = _parse_terms(cmd)
        graph = self.graphs[graph_name]
        graph = GraphLinker.prefix_graph_impl(graph, prefix_name)
        self.graphs[graph_name] = graph
        
    def link_graph(self, cmd):
        def _parse_terms(cmd):
            op_name_pattern = re.compile('[ \t\r\n]*link[ \t\r\n]+')
            graphs_pattern = re.compile('[ \t\r\n]*\[[^\]]*\][ \t\r\n]+')
            links_pattern = re.compile('[ \t\r\n]*\{[^\}]*\}[ \t\r\n]+')
            op_name = op_name_pattern.match(cmd)
            assert not op_name is None
            cmd = cmd[op_name.end():]
            graphs = graphs_pattern.match(cmd)
            assert not graphs is None
            cmd = cmd[graphs.end():]
            links = links_pattern.match(cmd)
            assert not links is None
            cmd = cmd[links.end():]
            name = cmd.replace(";", "").strip()
        
            graphs = graphs.string[graphs.start():graphs.end()]
            graphs = graphs.replace("[", "").replace("]", "").split(",")
            
            links = links.string[links.start():links.end()]
            links = links.replace("{", "").replace("}", "").split(",")
            
            return graphs, links, name
        
        def _parse_graphs(graphs_name):
            graphs = list()
            for graph_name in graphs_name:
                graph_name = graph_name.strip()
                assert graph_name in self.graphs
                graph = self.graphs[graph_name]
                graphs.append(graph)
            return graphs
        
        def _parse_links(links_str):
            links = dict()
            for link_str in links_str:
                key, value = link_str.strip().split("->")
                key, value = key.strip(), value.strip()
                assert not key in links
                links[key] = value
            return links
            
        graphs_name, links_str, name = _parse_terms(cmd)
        graphs = _parse_graphs(graphs_name)
        links = _parse_links(links_str)
        linked_graph = GraphLinker.link_graph_impl(graphs, links)
        self.graphs[name] = linked_graph
        
    @staticmethod
    def prefix_graph_impl(tensors, prefix=None):
        if prefix is None:
            prefix = ""
        else:
            prefix += "_"
        for tensor in tensors:
            tensor.id = prefix + tensor.id
            if not tensor.x1 is None:
                tensor.x1 = prefix + tensor.x1
            if not tensor.x2 is None:
                tensor.x2 = prefix + tensor.x2
        return tensors

    @staticmethod
    def link_graph_impl(graphs, link_node):
        named_graph = dict()
        for graph in graphs:
            for tensor in graph:
                assert tensor.id not in named_graph
                named_graph[tensor.id] = tensor
        for from_, to_ in link_node.items():
            print(from_)
            assert from_ in named_graph
            assert to_ in named_graph
            from_, to_ = named_graph[from_], named_graph[to_]
            assert to_.op_type == 'T'
            for tensor in named_graph.values():
                # do not allow non-leaf node to be from-ed, otherwise might referenced multiple times, 
                # each time will have a component of gradient, multiple components need to sumed,
                # and bwd graph breaks
                if tensor.x1 == from_.id:
                    assert False
                if tensor.x2 == from_.id:
                    assert False
            assert from_.require_grads == to_.require_grads
            # print(from_.shape, to_.shape)
            # assert from_.shape == to_.shape
            # assert from_.hidden == to_.hidden

            to_.x1, to_.x2 = from_.x1, from_.x2
            to_.op_type, to_.op_attr = from_.op_type, from_.op_attr
            to_.x1_shape, to_.x1_hidden = from_.x1_shape, from_.x1_hidden
            to_.x2_shape, to_.x2_hidden = from_.x2_shape, from_.x2_hidden
            to_.direct_output_shape, to_.direct_output_hidden = from_.direct_output_shape, from_.direct_output_hidden
            to_.post_communications = from_.post_communications
            to_.ops = from_.ops
            
            named_graph[to_.id] = to_
            del named_graph[from_.id]
        return named_graph.values()


def transformer(num_stacks):
    fwd_graphs = list()
    bwd_graphs = list()
    for i in range(num_stacks):
        mha_fwd = Tensor.parse_records('sharding_spreadsheets/dp/graphs/multiHeadAttentionFwd.csv')
        mha_bwd = Tensor.parse_records('sharding_spreadsheets/dp/graphs/multiHeadAttentionBwd.csv')
        ffn_fwd = Tensor.parse_records('sharding_spreadsheets/dp/graphs/feedForwardNetworkFwd.csv')
        ffn_bwd = Tensor.parse_records('sharding_spreadsheets/dp/graphs/feedForwardNetworkBwd.csv')
        
        mha_fwd = GraphLinker.prefix_graph_impl(mha_fwd, "mha")
        mha_bwd = GraphLinker.prefix_graph_impl(mha_bwd, "mha")
        ffn_fwd = GraphLinker.prefix_graph_impl(ffn_fwd, "ffn")
        ffn_bwd = GraphLinker.prefix_graph_impl(ffn_bwd, "ffn")
        
        Tensor.visualize(mha_fwd, "mha_fwd")
        Tensor.visualize(mha_bwd, "mha_bwd")
        Tensor.visualize(ffn_fwd, "ffn_fwd")
        Tensor.visualize(ffn_bwd, "ffn_bwd")
        
        fwd = GraphLinker.link_graph_impl([mha_fwd, ffn_fwd], {"mha_norm": "ffn_x0"})
        bwd = GraphLinker.link_graph_impl([mha_bwd, ffn_bwd], {"ffn_d_x0": "mha_d_norm"})
        
        fwd = GraphLinker.prefix_graph_impl(fwd, f"stack{i}")
        bwd = GraphLinker.prefix_graph_impl(bwd, f"stack{i}")
        fwd_graphs.append(fwd)
        bwd_graphs.append(bwd)
    
    Tensor.visualize(fwd_graphs[0], f'sharding_spreadsheets/dp/processed_graphs/fwd_stack')
    Tensor.visualize(bwd_graphs[0], f'sharding_spreadsheets/dp/processed_graphs/bwd_stack')
        
    fwd_links = dict()
    bwd_links = dict()
    for i in range(num_stacks-1):
        fwd_links[f"stack{i}_ffn_norm"] = f"stack{i+1}_mha_x"
        bwd_links[f"stack{i+1}_mha_d_x"] = f"stack{i}_ffn_d_norm"
    fwd = GraphLinker.link_graph_impl(fwd_graphs, fwd_links)
    bwd = GraphLinker.link_graph_impl(bwd_graphs, bwd_links)
    Tensor.visualize(fwd, f'sharding_spreadsheets/dp/processed_graphs/stack_{num_stacks}_fwd_stacks')
    Tensor.visualize(bwd, f'sharding_spreadsheets/dp/processed_graphs/stack_{num_stacks}_bwd_stacks')
    Tensor.to_records(fwd, f'sharding_spreadsheets/dp/processed_graphs/stack{num_stacks}Fwd.csv')
    Tensor.to_records(bwd, f'sharding_spreadsheets/dp/processed_graphs/stack{num_stacks}Bwd.csv')


if __name__ == '__main__':
    transformer(2)
    transformer(32)
        