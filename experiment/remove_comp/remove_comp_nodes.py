#!/usr/bin/env python

from chakra.et_def.et_def_pb2 import *
from chakra.third_party.utils.protolib import decodeMessage as decode_message
from chakra.third_party.utils.protolib import encodeMessage as encode_message
from chakra.third_party.utils.protolib import openFileRd as open_file_rd
import networkx as nx
import copy
import os
import itertools
import multiprocessing


def keep_coll_comm(node: Node):
    return node.type == NodeType.COMM_COLL_NODE

def keep_comp(node: Node):
    if node.type != NodeType.COMP_NODE:
        return False
    for attr in node.attr:
        if attr.name == "num_ops" and attr.uint64_val<10:
            return False
    return True


def remove_shortcuts(graph: nx.DiGraph):
    graph_removed = copy.deepcopy(graph)
    for from_, to_ in graph.edges():
        for child in graph.successors(from_):
            if child == to_:
                continue
            if nx.has_path(graph, child, to_):
                graph_removed.remove_edge(from_, to_)
                break
    return graph_removed


class RemoveChakraNodes:
    def __init__(self, keep_fn=None):
        self.keep_fn = keep_fn

    def apply(self, chakra_input_filename, chakra_output_filename):
        nodes = list()
        graph = nx.DiGraph()
        with open_file_rd(chakra_input_filename) as et:
            global_metadata = GlobalMetadata()
            node = Node()
            decode_message(et, global_metadata)
            while decode_message(et, node):
                nodes.append(copy.deepcopy(node))
                graph.add_node(node.id)
            for node in nodes:
                node_id = node.id
                for parent_id in node.data_deps:
                    graph.add_edge(parent_id, node_id)
        
        # graph = remove_shortcuts(graph)
        keeped_nodes = list()
        for node in nodes:
            node_id = node.id
            if self.keep_fn(node):
                keeped_nodes.append(node)
                continue
            parents_id = copy.deepcopy(graph.predecessors(node_id))
            children_id = copy.deepcopy(graph.successors(node_id))
            for parent_id in parents_id:
                graph.remove_edge(parent_id, node_id)
            for child_id in children_id:
                graph.remove_edge(node_id, child_id)
            for from_, to_ in itertools.product(parents_id, children_id):
                if graph.has_edge(from_, to_):
                    continue
                graph.add_edge(from_, to_)
        
        with open(chakra_output_filename, "wb") as et:
            encode_message(et, global_metadata)
            for node in keeped_nodes:
                node_id = node.id
                del node.data_deps[:]
                for parent in graph.predecessors(node_id):
                    node.data_deps.append(parent)
                encode_message(et, node)


def get_all_chakra_ets(root):
    files = os.listdir(root)
    valid_chakra_files = list()
    for file in files:
        if not file.endswith(".et"):
            continue
        valid_chakra_files.append(os.path.join(root, file))
    return valid_chakra_files


def prepare_batch_run_args(ori_root, keep="comp"):
    if keep == "comp":
        filter_fn = keep_comp
        output_root = "comp"
    elif keep == "coll_comm":
        filter_fn = keep_coll_comm
        output_root = "coll_comm"
    else:
        raise NotImplementedError()
    valid_chakra_files = get_all_chakra_ets(ori_root)
    args = list()
    for valid_chakra_file in valid_chakra_files:
        chakra_input_filename = valid_chakra_file
        chakra_output_filename = os.path.join(output_root, os.path.split(chakra_input_filename)[-1])
        args.append((chakra_input_filename, chakra_output_filename, filter_fn))
    return args


def run(arg):
    chakra_input_filename, chakra_output_filename, filter_fn = arg
    RemoveChakraNodes(filter_fn).apply(chakra_input_filename, chakra_output_filename)
    return True


if __name__ == '__main__':
    args = prepare_batch_run_args("./raw", keep="comp")
    args1 = prepare_batch_run_args("./raw", keep="coll_comm")
    with multiprocessing.Pool(int(multiprocessing.cpu_count()*0.8)) as pool:
        # ret = map(run, args)
        ret = pool.map(run, args)
        ret = pool.map(run, args1)
        pool.close()
        pool.join()
    
