import sys, os, random, json, time

sys.path.append("../../symbolic_tensor_graph")
sys.path.append("../../chakra/et_def/")
sys.path.append("../../chakra/third_party/utils/")

from multiprocessing import Pool, cpu_count

from models.transformer import transformer
from symbolic2chakra_converter import Symbolic2ChakraConverterWithOffload
from offload_strategy import OffloadStrategy
from tensor import Tensor

num_samples = 2048
et_dir = "./run_file/ets"
symbolic_dir = "../../sharding_spreadsheets/dp"
symbol_value_map = {
    "bp": 256,
    "B": 1024,
    "Seq": 1024,
    "H": 256,
    "D": 100,
    "DF": 400,
    "DI": 200,
    "DO": 100,
}
os.makedirs(et_dir, exist_ok=True)
pool = Pool(int(cpu_count() * 1))


def prepare_templates(symbolic_dir_):
    transformer(24, symbolic_dir_)
    tensors = Tensor.parse_records(
        os.path.join(symbolic_dir_, "processed_graphs", "transformer_24.csv")
    )
    offload_strategy = OffloadStrategy.create_blank(tensors)
    offload_strategy.to_records(
        os.path.join(symbolic_dir_, "offload_strategy", "blank", "transformer_24.csv")
    )
    return


def sample_symbol_value_map(i_, symbolic_dir_, et_dir_):
    offload_strategy = OffloadStrategy.parse_records(
        os.path.join(symbolic_dir_, "offload_strategy", "blank", "transformer_24.csv")
    )
    for key in offload_strategy.offload_strategy.keys():
        offload_strategy.offload_strategy[key] = random.randint(0, 1)
    offload_strategy.to_records(os.path.join(et_dir_, f"{i_}", "offload.csv"))
    print("sampled")


def convert_task(i_, symbolic_dir_, et_dir_, symbol_value_map_):
    os.makedirs(os.path.join(et_dir_, f"{i_}"), exist_ok=True)
    random.seed(time.time_ns())
    print(i_)
    sample_symbol_value_map(i_, symbolic_dir_, et_dir_)
    converter = Symbolic2ChakraConverterWithOffload(
        os.path.join(symbolic_dir_, "processed_graphs", f"transformer_24.csv"),
        os.path.join(et_dir_, f"{i_}", "offload.csv"),
        os.path.join(et_dir, f"{i_}", f"transformer"),
        256,
    )
    print(f"converter_created{i_}")
    converter.symbol_value_map = symbol_value_map_
    print(f"mapped{i_}")
    converter.convert()
    print(f"convertered{i_}")
    return True


def convert(symbolic_dir_, et_dir_, symbol_value_map_):
    ret = list()
    for i in range(num_samples):
        ret.append(
            pool.apply_async(
                convert_task, (i, symbolic_dir_, et_dir_, symbol_value_map_)
            )
        )
    for i in ret:
        i.get()
    return True


if __name__ == "__main__":
    prepare_templates(symbolic_dir_=symbolic_dir)
    convert_task(0, symbolic_dir, et_dir, symbol_value_map)
    ret = convert(
        symbolic_dir_=symbolic_dir, et_dir_=et_dir, symbol_value_map_=symbol_value_map
    )

    pool.close()
    pool.join()
