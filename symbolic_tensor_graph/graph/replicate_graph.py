import copy
import typing
import sympy as sp
from .graph import TensorGraph
from ..ops import Slice, BroadcastReduce, Customized


class ReplicateGraph:
    @classmethod
    def _update_tensor_name(cls, graph, tensor_name_template, inplace=False):
        if not inplace:
            graph = copy.deepcopy(graph)
        assert isinstance(graph, TensorGraph)
        tensors = graph.tensors
        for tensor in tensors:
            tensor.name = tensor_name_template % (tensor.name,)
        return graph

    @classmethod
    def _update_tensor_revision(cls, graph, new_revision, inplace=False):
        if not inplace:
            graph = copy.deepcopy(graph)
        assert isinstance(graph, TensorGraph)
        tensors = graph.tensors
        if isinstance(new_revision, str):
            new_revision = lambda old_revision: new_revision
        elif isinstance(new_revision, typing.Callable):
            pass
        else:
            assert False

        for tensor in tensors:
            tensor.revision = new_revision(tensor.revision)
        return graph

    @classmethod
    def _update_symbols(cls, graph, old_symbol_map_new_symbol, inplace=False):
        if not inplace:
            graph = copy.deepcopy(graph)
        assert isinstance(graph, TensorGraph)
        for from_, to_ in old_symbol_map_new_symbol.items():
            if isinstance(from_, str):
                from_ = sp.parse_expr(from_)
            if isinstance(to_, str):
                to_ = sp.parse_expr(to_)
            for tensor in graph.tensors:
                if not tensor.x1_shape is None:
                    for i, dim in enumerate(tensor.x1_shape):
                        tensor.x1_shape[i] = dim.replace(from_, to_)
                if not tensor.x1_hidden is None:
                    for i, dim in enumerate(tensor.x1_hidden):
                        tensor.x1_hidden[i] = dim.replace(from_, to_)
                if not tensor.x2_shape is None:
                    for i, dim in enumerate(tensor.x2_shape):
                        tensor.x2_shape[i] = dim.replace(from_, to_)
                if not tensor.x2_hidden is None:
                    for i, dim in enumerate(tensor.x2_hidden):
                        tensor.x2_hidden[i] = dim.replace(from_, to_)
                if tensor.op_type in {Slice.type_name, BroadcastReduce.type_name, Customized.type_name}:
                    tensor.op_attr = tensor.op_attr.replace(
                        f"{str(from_)}", f"({str(to_)})"
                    )
        return graph

    @classmethod
    def apply(
        cls,
        graph,
        tensor_name_template=None,
        new_revision=None,
        old_symbol_map_new_symbol=None,
        inplace=False,
    ):
        if not inplace:
            graph = copy.deepcopy(graph)
        if not tensor_name_template is None:
            cls._update_tensor_name(graph, tensor_name_template, inplace=True)
        if not new_revision is None:
            cls._update_tensor_revision(graph, new_revision, inplace=True)
        if not old_symbol_map_new_symbol is None:
            cls._update_symbols(graph, old_symbol_map_new_symbol, inplace=True)
        return graph
