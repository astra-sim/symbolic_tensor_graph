from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.graph.convert_chakra import ConvertChakra

# ------------------------------------------------------------------
# Helper utilities for VRAM accounting
# ------------------------------------------------------------------
def _tensor_mem_class(tensor):
    """Classify tensor for persistent VRAM footprint.

    weight : parameter shards stored persistently (exclude *_assembled_weight*)
    grad   : persistent gradient shards (only *_sharded_grad*)
    act    : forward activations that are kept (exclude backward/temp).
    None   : skip (temporary tensors not expected to occupy persistent VRAM)
    """

    name = tensor.name or ""

    # There are a few tensor variants that should NOT be counted towards
    # persistent VRAM because they are ephemeral (assembled buffers used
    # inside FSDP, micro-batch intermediate shards, etc.).
    # We filter those out *before* the normal classification.

    # 0) Explicitly ignore assembled / temporary buffers
    tmp_keywords = [
        "_assembled_weight",  # FSDP full-weight buffer
        "_assembled_weight_backward",  # its backward shadow
        "_assembled_grad",  # full gradient before shard/RS
    ]
    for kw in tmp_keywords:
        if kw in name:
            return None

    # 1) Parameters (persistent weight shards)
    # Note: Only the sharded parameters (require_grads == True) are stored
    # permanently. The full-precision assembled weights have been excluded
    # above.
    if tensor.require_grads:
        return "weight"

    # 2) Gradients (persistent)
    # We only keep the per-rank sharded gradients (post reduce-scatter).
    # Micro-batch private gradients (prefixed with "mb<i>.") are local
    # temporaries and should be skipped.
    if tensor.grad_of is not None and "_sharded_grad" in name:
        return "grad"

    # Heuristics to exclude large backward / other temporaries
    lower_name = name.lower()
    if "_backward" in lower_name or lower_name.split(".")[-1].startswith(
        ("d", "dw", "dq", "dk", "dv")
    ):
        return None  # skip temp

    # Everything else => activation
    return "act"


def _weight_and_opt_sizes(tensor, symbol_map, mixed_precision=False):
    """Return (weight_bytes, optimizer_state_bytes) for a parameter tensor.

    The logic mirrors ConvertChakra._create_IOInfo so reported numbers add
    up to the same total size but we split the optimizer state out.
    """
    from symbolic_tensor_graph.tensor import Tensor as _Tensor

    elem_cnt = _Tensor.eval_expr(_Tensor.eval_size(tensor.y_shape), symbol_map)

    # Weight representation bytes
    if mixed_precision:
        # 2 bytes (fp16) + 4 bytes (fp32 master) = 6 bytes per element
        weight_bytes = int(elem_cnt * 1.5) * 4
    else:
        # standard fp32 => 4 bytes per element
        weight_bytes = elem_cnt * 4

    # Optimizer state bytes (Adam: m & v, fp32): 2 * 4 bytes
    opt_bytes = (
        elem_cnt * 4 * 2
    )  # multiplier 2 already folded as 4 in original? wait orig used *4 only.
    # Note: ConvertChakra used *4 ( not *8 ) – but that code path is inconsistent wrt bytes.
    # For consistency with their total we keep the same (single replica) so /2.
    opt_bytes = elem_cnt * 4  # to match _create_IOInfo logic

    return weight_bytes, opt_bytes


def _tensor_size_bytes(tensor, symbol_map, mixed_precision=False):
    """Total size as computed by ConvertChakra (weight+opt if param)."""
    info = ConvertChakra._create_IOInfo(
        tensor, symbol_map, mixed_precision, fsdp_enabled=symbol_map.get("fsdp", 0) > 1
    )
    return info["size"]


def _print_gpu_vram(bundle_graph, symbol_map, mixed_precision=False, header=""):
    GiB = 1024**3
    for rank_key, tg in bundle_graph.graphs.items():
        stats = {"weight": 0, "opt": 0, "act": 0, "grad": 0}
        tensor_details = []  # Store details for sorting
        for tensor in tg.tensors:
            cls = _tensor_mem_class(tensor)
            if cls is None:
                continue
            if cls == "weight":
                w_b, opt_b = _weight_and_opt_sizes(tensor, symbol_map, mixed_precision)
                stats["weight"] += w_b
                stats["opt"] += opt_b
                tensor_details.append(
                    (cls, w_b + opt_b, tensor.id, Tensor.stringfy_shape(tensor.y_shape))
                )
            else:
                size_b = _tensor_size_bytes(tensor, symbol_map, mixed_precision)
                stats[cls] += size_b
                tensor_details.append(
                    (cls, size_b, tensor.id, Tensor.stringfy_shape(tensor.y_shape))
                )
        total = sum(stats.values())
        rk_str = ",".join([f"{d[0]}={d[1]}" for d in rank_key])
        print(
            f"{header}[GPU {rk_str}] total={total / GiB:.3f} GiB | "
            f"weights={stats['weight'] / GiB:.3f} | "
            f"opt={stats['opt'] / GiB:.3f} | "
            f"acts={stats['act'] / GiB:.3f} | "
            f"grads={stats['grad'] / GiB:.3f}"
        )
        # Print top 5 largest tensors by size for this rank
        # tensor_details.sort(key=lambda x: x[1], reverse=True)
        # print(f"    Top 5 Tensors for GPU {rk_str}:")
        # for i in range(min(5, len(tensor_details))):
        #     cls, size_b, t_id, shape_str = tensor_details[i]
        #     print(f"      {i+1}. Type={cls}, Size={size_b / GiB:.4f} GiB, ID={t_id}, Shape={shape_str}")

