import copy
from eg_simulator.executor.astrasim_executor import AstraSIMExecutor
from eg_simulator.runtime_database.astrasim_runtime_database import (
    AstrasimNodeRuntimeDatabase,
)
from eg_simulator.node_runner import NodeRunner
from symbolic_tensor_graph.symbolic2chakra_converter import Symbolic2ChakraConverter
from models.old.transformer import transformer


if __name__ == "__main__":
    # transformer(1024, "sharding_spreadsheets/transformer/dp")
    workloads = None
    system = "../astra-sim/inputs/system/sample_fully_connected_sys_roofline.txt"
    network = "../astra-sim/inputs/network/analytical/fully_connected.json"
    memory = "../astra-sim/inputs/remote_memory/analytical/no_memory_expansion.json"
    astrasim_bin = (
        "../astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    )

    executor = AstraSIMExecutor(system, network, memory, workloads, astrasim_bin)
    database = AstrasimNodeRuntimeDatabase(system, network, memory, astrasim_bin)
    node_runner = NodeRunner(executor, database)

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

    converter = Symbolic2ChakraConverter(
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_1024.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer1024.w0l0i0.dp",
        symbol_value_map,
        symbol_value_map["bp"] * symbol_value_map["mp"],
    )
    converter.convert()
    nodes = converter.get_nodes()
    runtime = node_runner.run_nodes(nodes)
    hook = 1
    runtime2 = node_runner.run_nodes(nodes)
    hook = 2
