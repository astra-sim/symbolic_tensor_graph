import os
from models.old.transformer import transformer
from multiprocessing import Pool, cpu_count
from symbolic_tensor_graph.symbolic2chakra_converter import Symbolic2ChakraConverter


def _convert(_symbolic_value_map, symbolic, eg, offload=None):
    if offload is None:
        converter = Symbolic2ChakraConverter(
            symbolic,
            eg,
            _symbolic_value_map,
            _symbolic_value_map["bp"] * _symbolic_value_map["tp"],
        )
    else:
        pass
    converter.convert_and_readout()

    return True


if __name__ == "__main__":
    symbol_value_map = {
        "bp": 16,
        "tp": 1,
        "B": 32 * 16,
        "Seq": 256,
        "H": 16,
        "D": 100,
        "DF": 400,
        "DI": 200,
        "DO": 100,
    }
    os.makedirs("sharding_spreadsheets/transformer/validation", exist_ok=True)
    pool = Pool(int(cpu_count() * 0.8))
    rets = list()

    rets.append(
        pool.apply_async(transformer, (8, "sharding_spreadsheets/transformer/dp"))
    )
    for ret in rets:
        ret.get()
    rets.clear()

    rets.append(
        pool.apply_async(
            _convert,
            (
                symbol_value_map,
                "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_8.csv",
                "sharding_spreadsheets/transformer/validation/transformer8.dp",
                None,
            ),
        )
    )
    for ret in rets:
        ret.get()
    rets.clear()
