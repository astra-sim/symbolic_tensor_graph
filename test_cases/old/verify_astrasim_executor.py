import copy
from eg_simulator.executor.astrasim_executor import AstraSIMExecutor
from symbolic_tensor_graph.symbolic2chakra_converter import Symbolic2ChakraConverter


if __name__ == "__main__":
    workloads = [
        f"./sharding_spreadsheets/transformer/validation/symbolic_transformer32.w0l0i0.dp.{npu_id}.eg"
        for npu_id in range(64)
    ]
    system = "../astra-sim/inputs/system/sample_fully_connected_sys_roofline.txt"
    network = "../astra-sim/inputs/network/analytical/fully_connected.json"
    memory = "../astra-sim/inputs/remote_memory/analytical/no_memory_expansion.json"
    astrasim_bin = (
        "../astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    )

    runner = AstraSIMExecutor(system, network, memory, workloads, astrasim_bin)
    cycles = runner.run()
    print(cycles)
    hook = 0

    symbol_value_map = {
        "bp": 1024,
        "tp": 1,
        "B": 32 * 1024,
        "Seq": 1024,
        "H": 256,
        "D": 100,
        "DF": 400,
        "DI": 200,
        "DO": 100,
    }

    converter = Symbolic2ChakraConverter(
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_32.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer32.w0l0i0.dp",
        symbol_value_map,
        symbol_value_map["bp"] * symbol_value_map["tp"],
    )
    converter.convert()
    nodes = converter.get_nodes()
    runner.update_workload(nodes)
    cycles = runner.run()
    print(cycles)
    hook = 1

    for node in nodes:
        node = copy.deepcopy(node)
        while len(node.parent) > 0:
            node.parent.pop()
        runner.update_workload([node])
        cycles = runner.run()
        print(f"{node.id}: {cycles}")
    hook = 2
