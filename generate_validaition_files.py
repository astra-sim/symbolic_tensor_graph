import os
from models.transformer import transformer
from multiprocessing import Pool, cpu_count
from symbolic_tensor_graph.symbolic2chakra_converter import (
    Symbolic2ChakraConverter
)


def _convert(_symbolic_value_map, symbolic, eg, offload=None):
    if offload is None:
        converter = Symbolic2ChakraConverter(
            symbolic, eg, _symbolic_value_map, _symbolic_value_map["bp"] * _symbolic_value_map["mp"]
        )
    else:
        pass
        # converter = Symbolic2ChakraConverterWithOffload(
        #     symbolic,
        #     offload,
        #     eg,
        #     _symbolic_value_map["bp"] * _symbolic_value_map["mp"],
        # )
    converter.convert()

    return True


if __name__ == "__main__":
    symbol_value_map = {
        "bp": 1024,
        "mp": 1,
        "B": 32 * 1024,
        "Seq": 1024,
        "H": 256,
        "D": 100,
        "DF": 400,
        "DI": 200,
        "DO": 100,
    }
    os.makedirs("sharding_spreadsheets/transformer/validation", exist_ok=True)
    pool = Pool(int(cpu_count() * 0.8))
    rets = list()

    # rets.append(pool.apply_async(transformer, (128, "sharding_spreadsheets/transformer/dp")))
    # rets.append(pool.apply_async(transformer, (128, "sharding_spreadsheets/transformer/fsdp")))
    # rets.append(
    #     pool.apply_async(transformer, (128, "sharding_spreadsheets/transformer/divya_parallel"))
    # )
    rets.append(pool.apply_async(transformer, (2, "sharding_spreadsheets/transformer/dp")))
    rets.append(pool.apply_async(transformer, (2, "sharding_spreadsheets/transformer/fsdp")))
    rets.append(
        pool.apply_async(transformer, (2, "sharding_spreadsheets/transformer/divya_parallel"))
    )
    for ret in rets:
        ret.get()
    rets.clear()

    # rets.append(pool.apply_async(transformer, (128, "sharding_spreadsheets/transformer/dp")))
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (128, "sharding_spreadsheets/transformer/dp", 1, 0, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (128, "sharding_spreadsheets/transformer/dp", 1, 1, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (128, "sharding_spreadsheets/transformer/dp", 1, 1, 1)
    #     )
    # )

    # rets.append(pool.apply_async(transformer, (128, "sharding_spreadsheets/transformer/fsdp")))
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (128, "sharding_spreadsheets/transformer/fsdp", 1, 0, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (128, "sharding_spreadsheets/transformer/fsdp", 1, 1, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (128, "sharding_spreadsheets/transformer/fsdp", 1, 1, 1)
    #     )
    # )

    # rets.append(
    #     pool.apply_async(transformer, (128, "sharding_spreadsheets/transformer/divya_parallel"))
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy,
    #         (128, "sharding_spreadsheets/transformer/divya_parallel", 1, 0, 0),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy,
    #         (128, "sharding_spreadsheets/transformer/divya_parallel", 1, 1, 0),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy,
    #         (128, "sharding_spreadsheets/transformer/divya_parallel", 1, 1, 1),
    #     )
    # )

    rets.append(pool.apply_async(transformer, (2, "sharding_spreadsheets/transformer/dp")))
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (2, "sharding_spreadsheets/transformer/dp", 1, 0, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (2, "sharding_spreadsheets/transformer/dp", 1, 1, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (2, "sharding_spreadsheets/transformer/dp", 1, 1, 1)
    #     )
    # )

    rets.append(pool.apply_async(transformer, (2, "sharding_spreadsheets/transformer/fsdp")))
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (2, "sharding_spreadsheets/transformer/fsdp", 1, 0, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (2, "sharding_spreadsheets/transformer/fsdp", 1, 1, 0)
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy, (2, "sharding_spreadsheets/transformer/fsdp", 1, 1, 1)
    #     )
    # )

    rets.append(
        pool.apply_async(transformer, (2, "sharding_spreadsheets/transformer/divya_parallel"))
    )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy,
    #         (2, "sharding_spreadsheets/transformer/divya_parallel", 1, 0, 0),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy,
    #         (2, "sharding_spreadsheets/transformer/divya_parallel", 1, 1, 0),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         transformer_offload_strategy,
    #         (2, "sharding_spreadsheets/transformer/divya_parallel", 1, 1, 1),
    #     )
    # )
    for ret in rets:
        ret.get()
    rets.clear()

    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w0l0i0.dp",
    #             None,
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l0i0.dp",
    #             "sharding_spreadsheets/transformer/dp/offload_strategy/transformer_128_w1_l0_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l1i0.dp",
    #             "sharding_spreadsheets/transformer/dp/offload_strategy/transformer_128_w1_l1_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l1i1.dp",
    #             "sharding_spreadsheets/transformer/dp/offload_strategy/transformer_128_w1_l1_i1.csv",
    #         ),
    #     )
    # )

    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w0l0i0.fsdp",
    #             None,
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l0i0.fsdp",
    #             "sharding_spreadsheets/transformer/fsdp/offload_strategy/transformer_128_w1_l0_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l1i0.fsdp",
    #             "sharding_spreadsheets/transformer/fsdp/offload_strategy/transformer_128_w1_l1_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l1i1.fsdp",
    #             "sharding_spreadsheets/transformer/fsdp/offload_strategy/transformer_128_w1_l1_i1.csv",
    #         ),
    #     )
    # )

    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w0l0i0.divya",
    #             None,
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l0i0.divya",
    #             "sharding_spreadsheets/transformer/divya_parallel/offload_strategy/transformer_128_w1_l0_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l1i0.divya",
    #             "sharding_spreadsheets/transformer/divya_parallel/offload_strategy/transformer_128_w1_l1_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_128.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer1T.w1l1i1.divya",
    #             "sharding_spreadsheets/transformer/divya_parallel/offload_strategy/transformer_128_w1_l1_i1.csv",
    #         ),
    #     )
    # )

    rets.append(
        pool.apply_async(
            _convert,
            (
                symbol_value_map,
                "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_2.csv",
                "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp",
                None,
            ),
        )
    )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l0i0.dp",
    #             "sharding_spreadsheets/transformer/dp/offload_strategy/transformer_2_w1_l0_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l1i0.dp",
    #             "sharding_spreadsheets/transformer/dp/offload_strategy/transformer_2_w1_l1_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l1i1.dp",
    #             "sharding_spreadsheets/transformer/dp/offload_strategy/transformer_2_w1_l1_i1.csv",
    #         ),
    #     )
    # )

    rets.append(
        pool.apply_async(
            _convert,
            (
                symbol_value_map,
                "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_2.csv",
                "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.fsdp",
                None,
            ),
        )
    )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l0i0.fsdp",
    #             "sharding_spreadsheets/transformer/fsdp/offload_strategy/transformer_2_w1_l0_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l1i0.fsdp",
    #             "sharding_spreadsheets/transformer/fsdp/offload_strategy/transformer_2_w1_l1_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l1i1.fsdp",
    #             "sharding_spreadsheets/transformer/fsdp/offload_strategy/transformer_2_w1_l1_i1.csv",
    #         ),
    #     )
    # )

    rets.append(
        pool.apply_async(
            _convert,
            (
                symbol_value_map,
                "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_2.csv",
                "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.divya",
                None,
            ),
        )
    )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l0i0.divya",
    #             "sharding_spreadsheets/transformer/divya_parallel/offload_strategy/transformer_2_w1_l0_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l1i0.divya",
    #             "sharding_spreadsheets/transformer/divya_parallel/offload_strategy/transformer_2_w1_l1_i0.csv",
    #         ),
    #     )
    # )
    # rets.append(
    #     pool.apply_async(
    #         _convert,
    #         (
    #             symbol_value_map,
    #             "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_2.csv",
    #             "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w1l1i1.divya",
    #             "sharding_spreadsheets/transformer/divya_parallel/offload_strategy/transformer_2_w1_l1_i1.csv",
    #         ),
    #     )
    # )

    for ret in rets:
        ret.get()
    rets.clear()
