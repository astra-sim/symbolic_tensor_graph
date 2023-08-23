import sys, os, random, json, time
sys.path.append('../../')
sys.path.append('../../../../chakra/et_def/')
sys.path.append('../../../../chakra/third_party/utils/')

from multiprocessing import Pool, cpu_count

import design_space
from models.transformer import transformer
from symbolic2chakra_converter import Symbolic2ChakraConverter

num_samples = 2048
et_dir = './run_file/ets'
symbolic_dir = '../../sharding_spreadsheets/dp'
os.makedirs(et_dir, exist_ok=True)
pool = Pool(int(cpu_count()*1))


def prepare_templates(design_space_):
    handler = list()
    for num_layers in design_space_.num_layers:
        handler.append(pool.apply_async(transformer, (num_layers, symbolic_dir)))
    for h in handler:
        h.get()
    handler.clear()
    return


def sample_symbol_value_map(design_space_):
    symbol_value_map = dict()
    for key in design_space_.symbol_value_map:
        value = random.choice(design_space_.symbol_value_map[key])
        symbol_value_map[key] = value
    return symbol_value_map
    

def convert_task(i_, design_space_):
    os.makedirs(os.path.join(et_dir, f"{i_}"), exist_ok=True)
    random.seed(time.time_ns())
    print(i_)
    num_layers = random.choice(design_space_.num_layers)
    converter = Symbolic2ChakraConverter(
        os.path.join(symbolic_dir, "processed_graphs", f"transformer_{num_layers}.csv"),
        os.path.join(et_dir, f"{i_}", f"transformer_{i_}"),
        256
    )
    symbol_value_map = sample_symbol_value_map(design_space_)
    print(symbol_value_map)
    converter.symbol_value_map = symbol_value_map
    converter.convert()
    symbol_value_map['num_layers'] = num_layers
    return symbol_value_map


def convert(design_space_):
    cfgs = dict()
    for i in range(num_samples):
        cfgs[i] = pool.apply_async(convert_task, (i, design_space_))
    for i in cfgs.keys():
        cfgs[i] = cfgs[i].get()
    return cfgs


if __name__ == '__main__':
    prepare_templates(design_space_=design_space)
    cfgs = convert(design_space_=design_space)
    
    pool.close()
    pool.join()
    
    f = open('./run_file/cfgs.json', 'w')
    json.dump(cfgs, f)
    f.close()
