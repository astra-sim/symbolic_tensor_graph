import copy
from ..graph.graph import TensorGraph, BundledTensorGraph
from ..tensor import Tensor
from ..ops import Shadow


class GraphDistributer:
    @classmethod
    def _sanity_check(
        cls,
        tensor_graph,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        tensor_id_temporal_map,
    ):
        for symbol in spatial_parallel_dims:
            assert symbol in symbol_map_value
        for symbol in temporal_parallel_dims:
            assert symbol in symbol_map_value
        assert len(tensor_id_temporal_map) == len(tensor_graph.tensors)
        for tensor in tensor_id_temporal_map.keys():
            position = tensor_id_temporal_map[tensor]
            assert len(position.keys()) == len(temporal_parallel_dims)
            for asked_dim in position.keys():
                asked_rank = position[asked_dim]
                assert asked_dim in temporal_parallel_dims
                assert asked_rank < symbol_map_value[asked_dim] and asked_rank >= 0

    @classmethod
    def _temporal_dispatch_tensors(cls, tensors, tensor_id_temporal_map):
        buckets = dict()
        for tensor in tensors:
            tensor_id = tensor.id
            position = tensor_id_temporal_map[tensor_id]
            target_bucket_key = cls._mapping_dict_to_tuple(position)
            if not target_bucket_key in buckets:
                buckets[target_bucket_key] = list()
            buckets[target_bucket_key].append(tensor)
        return buckets

    @classmethod
    def _mapping_dict_to_tuple(cls, dict_):
        ret = tuple()
        for key in dict_:
            ret += ((key, dict_[key]),)
        return ret

    @classmethod
    def _fix_cross_bucket_data_dependancies(cls, buckets, tensor_id_temporal_map):
        remote_parent_shadow_pairs = list()
        for bucket_key in buckets:
            remote_parent_map_shadow = dict()
            bucket = buckets[bucket_key]
            for tensor in bucket:
                if not tensor.x1 is None:
                    if not tensor.x1 in bucket:
                        remote = tensor.x1
                        child = tensor
                        if not remote in remote_parent_map_shadow:
                            shadow = cls._create_shadow(remote)
                            remote_parent_map_shadow[remote] = shadow
                        shadow = remote_parent_map_shadow[remote]
                        child.x1 = shadow
                if not tensor.x2 is None:
                    if not tensor.x2 in bucket:
                        remote = tensor.x2
                        child = tensor
                        if not remote in remote_parent_map_shadow:
                            shadow = cls._create_shadow(remote)
                            remote_parent_map_shadow[remote] = shadow
                        shadow = remote_parent_map_shadow[remote]
                        child.x2 = shadow
            for remote in remote_parent_map_shadow:
                remote_map = cls._mapping_dict_to_tuple(
                    tensor_id_temporal_map[remote.id]
                )
                shadow = remote_parent_map_shadow[remote]
                shadow_map = bucket_key
                remote_parent_shadow_pairs.append(
                    (
                        (remote_map, remote.id),
                        (shadow_map, shadow.id),
                    )
                )
                bucket.append(shadow)
        return buckets, remote_parent_shadow_pairs

    @classmethod
    def _spatial_copy_graphs(
        cls,
        buckets,
        remote_parent_shadow_pairs,
        spatial_parallel_dims,
        symbol_map_value,
    ):
        if len(spatial_parallel_dims) == 0:
            return buckets, remote_parent_shadow_pairs
        dim = spatial_parallel_dims[0]
        spatial_parallel_dims = spatial_parallel_dims[1:]
        buckets, remote_parent_shadow_pairs = cls._spatial_copy_graphs(
            buckets, remote_parent_shadow_pairs, spatial_parallel_dims, symbol_map_value
        )
        new_buckets = dict()
        for key in buckets.keys():
            for rank in range(symbol_map_value[dim]):
                new_key = key + ((dim, rank),)
                stub_graph = TensorGraph(buckets[key])
                stub_graph_copied = copy.deepcopy(stub_graph)
                new_buckets[new_key] = stub_graph_copied.tensors
        new_remote_parent_shadow_pairs = list()
        for item in remote_parent_shadow_pairs:
            for rank in range(symbol_map_value[dim]):
                (remote_map, remote_id), (shadow_map, shadow_id) = item
                remote_map = remote_map + ((dim, rank),)
                shadow_map = shadow_map + ((dim, rank),)
                new_item = (remote_map, remote_id), (shadow_map, shadow_id)
                new_remote_parent_shadow_pairs.append(new_item)
        return new_buckets, new_remote_parent_shadow_pairs

    @classmethod
    def _create_shadow(cls, remote):
        shadow = Tensor(create_empty=True)
        shadow.name = f"shadow_{remote.name}"
        shadow.revision = remote.revision
        shadow.op_type = Shadow.type_name
        shadow.require_grads = False
        shadow.x1_shape = remote.y_shape
        shadow.x1_hidden = remote.y_hidden
        return shadow

    @classmethod
    def apply(
        cls,
        tensor_graph,
        symbol_map_value,
        spatial_parallel_dims,
        temporal_parallel_dims,
        tensor_id_temporal_map,
        inplace=False,
    ):
        if not inplace:
            tensor_graph = copy.deepcopy(tensor_graph)
        cls._sanity_check(
            tensor_graph,
            symbol_map_value,
            spatial_parallel_dims,
            temporal_parallel_dims,
            tensor_id_temporal_map,
        )

        buckets = cls._temporal_dispatch_tensors(
            tensor_graph.tensors, tensor_id_temporal_map
        )
        buckets, remote_parent_shadow_pairs = cls._fix_cross_bucket_data_dependancies(
            buckets, tensor_id_temporal_map
        )
        buckets, remote_parent_shadow_pairs = cls._spatial_copy_graphs(
            buckets, remote_parent_shadow_pairs, spatial_parallel_dims, symbol_map_value
        )
        graphs = dict()
        for key in buckets.keys():
            graphs[key] = TensorGraph(buckets[key])

        bundled_tensor_graph = BundledTensorGraph(
            graphs,
            remote_parent_shadow_pairs,
            spatial_parallel_dims,
            temporal_parallel_dims,
            symbol_map_value,
        )
        return bundled_tensor_graph
