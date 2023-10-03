import sys, os, random, json, time

sys.path.append("../../symbolic_tensor_graph")
sys.path.append("../../chakra/et_def/")
sys.path.append("../../chakra/third_party/utils/")

from multiprocessing import Pool, cpu_count

import design_space
from models.transformer import transformer
from symbolic2chakra_converter import Symbolic2ChakraConverter

num_samples = 2048
et_dir = "./run_file/ets/1"
symbolic_dir = "../../sharding_spreadsheets/dp"
os.makedirs(et_dir, exist_ok=True)


def prepare_templates(design_space_, num_layers_):
    pool = Pool(int(cpu_count() * 1))
    handler = list()
    for num_layers in num_layers_:
        handler.append(pool.apply_async(transformer, (num_layers, symbolic_dir)))
    for h in handler:
        h.get()
    handler.clear()
    pool.close()
    pool.join()
    return


def sample_symbol_value_map(design_space_):
    symbol_value_map = dict()
    for key in design_space_:
        value = random.choice(design_space_[key])
        symbol_value_map[key] = value
    return symbol_value_map


def convert_task(i_, design_space_, num_layers):
    os.makedirs(os.path.join(et_dir, f"{i_}"), exist_ok=True)
    random.seed(time.time_ns())
    print(i_)
    num_layers = random.choice(num_layers)
    converter = Symbolic2ChakraConverter(
        os.path.join(symbolic_dir, "processed_graphs", f"transformer_{num_layers}.csv"),
        os.path.join(et_dir, f"{i_}", f"transformer_{i_}"),
        256,
    )
    symbol_value_map = sample_symbol_value_map(design_space_)
    print(symbol_value_map)
    converter.symbol_value_map = symbol_value_map
    converter.convert()
    symbol_value_map["num_layers"] = num_layers
    return symbol_value_map


def convert(design_space_, num_layers):
    pool = Pool(int(cpu_count() * 1))
    cfgs = dict()
    for i in range(num_samples):
        cfgs[i] = pool.apply_async(convert_task, (i, design_space_, num_layers))
    for i in cfgs.keys():
        cfgs[i] = cfgs[i].get()
    pool.close()
    pool.join()
    return cfgs


if __name__ == "__main__":
    design_space_ = design_space.symbol_value_map
    num_layers = design_space.num_layers
    prepare_templates(design_space_=design_space_, num_layers_=num_layers)
    cfgs = convert(design_space_=design_space_, num_layers=num_layers)

    f = open("./run_file/cfgs.json", "w")
    json.dump(cfgs, f)
    f.close()
