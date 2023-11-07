import os, random
from models.old.transformer import transformer, transformer_offload_strategy
from multiprocessing import Pool, cpu_count
from symbolic2chakra_converter import (
    Symbolic2ChakraConverter,
    Symbolic2ChakraConverterWithOffload,
)


def _convert(_symbolic_value_map, symbolic, eg, offload=None):
    if offload is None:
        converter = Symbolic2ChakraConverter(
            symbolic, eg, _symbolic_value_map["bp"] * _symbolic_value_map["mp"]
        )
    else:
        converter = Symbolic2ChakraConverterWithOffload(
            symbolic,
            offload,
            eg,
            _symbolic_value_map["bp"] * _symbolic_value_map["mp"],
        )
    converter.symbol_value_map = _symbolic_value_map
    converter.convert()

    return True


def _sample_symbol_value_map(symbol_value_map_template):
    symbol_value_map = dict()
    for key in symbol_value_map_template:
        if key == "bp":
            key = 2 ** random.randint(low, high)
            continue
        low, high = symbol_value_map_template[key]
        symbol_value_map[key] = random.randint(low, high)
    if "B" in symbol_value_map and "bp" in symbol_value_map:
        symbol_value_map["B"] *= symbol_value_map["bp"]
    if "bp" in symbol_value_map and "mp" in symbol_value_map:
        symbol_value_map["mp"] = 256 // symbol_value_map["bp"]

    return symbol_value_map


if __name__ == "__main__":
    symbol_value_map_template = {
        "bp": (0, 8),
        "mp": (0, 8),
        "B": (1, 2048),
        "Seq": (32, 4096),
        "H": (4, 1024),
        "D": (10, 1000),
        "DF": (40, 4000),
        "DI": (20, 2000),
        "DO": (10, 1000),
    }
