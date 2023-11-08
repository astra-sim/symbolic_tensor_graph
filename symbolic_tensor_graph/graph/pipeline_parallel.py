import copy
from ..tensor import Tensor
from ..ops import PlaceHolder
from ..graph.graph import TensorGraph


class GraphDistributer:
    @classmethod
    def _sanity_check(
        cls,
        tensor_graph,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        tensor_maps,
    ):
        for symbol in tensor_graph.get_symbols():
            assert symbol in symbol_map_value
        for symbol in spatial_parallel_dims:
            assert symbol in symbol_map_value
        for symbol in temporal_parallel_dims:
            assert symbol in symbol_map_value
        mapped_tensors = list()
        assert len(temporal_parallel_dims) == len(tensor_maps.keys())
        for dim in tensor_maps:
            assert dim in temporal_parallel_dims
            map_this_dim = tensor_maps[dim]
            assert len(map_this_dim) == symbol_map_value[dim]
            for i, bucket in enumerate(map_this_dim):
                for tensor in bucket:
                    assert tensor in tensor_graph.tensors
                    assert not tensor in mapped_tensors
                    mapped_tensors.append(tensor)
        assert len(mapped_tensors) == len(tensor_graph.tensors)

    @classmethod
    def _split_graph(cls, tensor_graph, tensor_maps):
        for dim in tensor_maps:
            map_this_dim = tensor_maps[dim]
        raise NotImplementedError()

    @classmethod
    def apply(
        cls,
        tensor_graph,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        tensor_maps,
    ):
        cls._sanity_check(
            tensor_graph,
            symbol_map_value,
            spatial_parallel_dims,
            temporal_parallel_dims,
            tensor_maps,
        )


class GraphSpliter:
    @classmethod
    def apply(cls, tensor_graph, split_tensors, inplace=False):
        if not inplace:
            tensor_graph = copy.deepcopy(tensor_graph)
        tensor_id_map_tensor = tensor_graph.get_tensor_id_map_tensor()
        tensor_parent_to_child = tensor_graph.get_tensor_parent_to_child_links()
        new_shadow_tensors = list()
        for tensor in split_tensors:
            splitted_shadow_tensor = cls._split_tensor(
                tensor, tensor_parent_to_child, tensor_id_map_tensor
            )
            new_shadow_tensors.append(splitted_shadow_tensor)
        for tensor in new_shadow_tensors:
            tensor_graph.tensors.append(tensor)

        upper_graph_tensors = cls._find_upper_graph_tensors(
            tensor_graph, split_tensors, new_shadow_tensors
        )
        lower_graph_tensors = cls._get_complement(
            tensor_graph.tensors, upper_graph_tensors
        )

        in_tensors = tensor_graph.in_tensors + new_shadow_tensors
        out_tensors = tensor_graph.out_tensors + split_tensors

        (
            upper_in_tensors,
            upper_out_tensors,
            lower_in_tensors,
            lower_out_tensors,
        ) = cls._split_in_out_tensors(
            upper_graph_tensors, lower_graph_tensors, in_tensors, out_tensors
        )

        upper_graph = TensorGraph(
            upper_graph_tensors, upper_in_tensors, upper_out_tensors
        )
        lower_graph = TensorGraph(
            lower_graph_tensors, lower_in_tensors, lower_out_tensors
        )
        return upper_graph, lower_graph, new_shadow_tensors

    @classmethod
    def _split_tensor(cls, tensor, tensor_parent_to_child, tensor_id_map_tensor):
        if isinstance(tensor, str):
            if not "@" in tensor:
                tensor += "@0"
            tensor = tensor_id_map_tensor[tensor]
        assert not tensor.require_grads
        splitted_shadow_tensor = Tensor(create_empty=True)
        splitted_shadow_tensor.name = f"shadow{tensor.name}"
        splitted_shadow_tensor.require_grads = tensor.require_grads
        splitted_shadow_tensor.op_type = PlaceHolder.type_name
        splitted_shadow_tensor.x1_shape = tensor.y_shape
        splitted_shadow_tensor.x1_hidden = tensor.y_hidden
        splitted_shadow_tensor.revision = tensor.revision
        for child_id in tensor_parent_to_child[tensor.id]:
            child = tensor_id_map_tensor[child_id]
            if child.x1 == tensor:
                child.x1 = splitted_shadow_tensor
            if child.x2 == tensor:
                child.x2 = splitted_shadow_tensor
        splitted_shadow_tensor._shadow_of = tensor.id
        tensor._shadow = splitted_shadow_tensor.id
        return splitted_shadow_tensor

    @classmethod
    def _reachable(cls, from_, to_, parent_to_child_links):
        if from_ == to_:
            return True
        for child in parent_to_child_links[from_]:
            if cls._reachable(child, to_):
                return True
        return False

    @classmethod
    def _find_upper_graph_tensors(
        cls, tensor_graph, split_real_tensors, split_shadow_tensors
    ):
        upper_graph_tensors = list()
        parent_to_child_links = tensor_graph.get_tensor_parent_to_child_link()
        for tensor in tensor_graph.tensors:
            for split_tensor in split_real_tensors:
                if cls._reachable(tensor.id, split_tensor.id, parent_to_child_links):
                    upper_graph_tensors.append(tensor)
                    break
            for split_shadow_tensor in split_shadow_tensors:
                assert not cls._reachable(
                    tensor.id, split_shadow_tensor.id, parent_to_child_links
                )
        return upper_graph_tensors

    @classmethod
    def _get_complement(cls, all_tensors, one_set_tensors):
        all_tensors = set(all_tensors)
        one_set_tensors = set(one_set_tensors)
        another_set_tensors = all_tensors - one_set_tensors
        return list(another_set_tensors)

    @classmethod
    def _split_in_out_tensors(
        cls, upper_tensors, lower_tensors, in_tensors, out_tensors
    ):
        upper_in_tensors = list()
        upper_out_tensors = list()
        lower_in_tensors = list()
        lower_out_tensors = list()

        for tensor in in_tensors:
            if tensor in upper_tensors:
                upper_in_tensors.append(tensor)
            else:
                assert tensor in lower_tensors
                lower_in_tensors.append(tensor)

        for tensor in out_tensors:
            if tensor in upper_tensors:
                upper_out_tensors.append(tensor)
            else:
                assert tensor in lower_tensors
                lower_out_tensors.append(tensor)
        return upper_in_tensors, upper_out_tensors, lower_in_tensors, lower_out_tensors
