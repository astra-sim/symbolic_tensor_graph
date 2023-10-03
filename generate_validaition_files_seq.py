import os
from models.transformer import transformer
from multiprocessing import Pool, cpu_count
from symbolic_tensor_graph.symbolic2chakra_converter import Symbolic2ChakraConverter


def _convert(_symbolic_value_map, symbolic, eg, offload=None):
    if offload is None:
        converter = Symbolic2ChakraConverter(
            symbolic,
            eg,
            _symbolic_value_map,
            _symbolic_value_map["bp"] * _symbolic_value_map["mp"],
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

    transformer(2, "sharding_spreadsheets/transformer/dp")
    transformer(2, "sharding_spreadsheets/transformer/fsdp")
    transformer(2, "sharding_spreadsheets/transformer/divya_parallel")

    _convert(
        symbol_value_map,
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_2.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp",
        None,
    )

    _convert(
        symbol_value_map,
        "sharding_spreadsheets/transformer/fsdp/processed_graphs/transformer_2.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.fsdp",
        None,
    )
    _convert(
        symbol_value_map,
        "sharding_spreadsheets/transformer/divya_parallel/processed_graphs/transformer_2.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.divya",
        None,
    )
