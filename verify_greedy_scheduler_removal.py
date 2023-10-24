import copy
from symbolic_tensor_graph.chakra_remove_shortcuts import ChakraShortcutRemover
from symbolic_tensor_graph.scheduler.scheduler import Scheduler as GreedyScheduler

# from symbolic_tensor_graph.scheduler.greedy_scheduler import GreedyScheduler
from symbolic_tensor_graph.symbolic2chakra_converter import Symbolic2ChakraConverter
from eg_simulator.executor.astrasim_executor import AstraSIMExecutor
from eg_simulator.runtime_database.astrasim_runtime_database import (
    AstrasimNodeRuntimeDatabase,
)
from eg_simulator.node_runner import NodeRunner


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

    converter = Symbolic2ChakraConverter(
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_2.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp",
        symbol_value_map,
        symbol_value_map["bp"] * symbol_value_map["mp"],
    )
    converter.convert()
    ori_nodes = converter.get_nodes()
    ori_nodes = copy.deepcopy(ori_nodes)

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

    nodes_runtime_list = node_runner.run_nodes(ori_nodes)
    nodes_runtime = dict()
    for node, node_runtime in zip(ori_nodes, nodes_runtime_list):
        assert not node.id in nodes_runtime
        nodes_runtime[node.id] = node_runtime

    # converter.readout()

    # greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes))
    # g1_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g1_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g1"
    # converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes), queues_function=[None, None]
    # )
    # g2_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g2_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g2"
    # converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes), queues_function=[None, None, None]
    # )
    # g3_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g3_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g3"
    # converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes), queues_function=[None, None, None, None, None]
    # )
    # g5_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g5_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g5"
    # converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes),
    #     queues_function=[None, None, None, None, None, None, None, None, None, None],
    # )
    # g10_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g10_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g10"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g1_nodes))
    # g1s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g1s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g1s"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g2_nodes))
    # g2s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g2s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g2s"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g3_nodes))
    # g3s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g3s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g3s"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g5_nodes))
    # g5s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g5s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g5s"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g10_nodes))
    # g10s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g10s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g10s"
    # converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes), node_runtime=nodes_runtime
    # )
    # g1_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g1_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr1"
    # converter.readout()

    greedy_scheduler = GreedyScheduler(
        copy.deepcopy(ori_nodes),
        node_runtime=nodes_runtime,
        queues_function=[None, None],
    )
    g2_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g2_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr2"
    converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes),
    #     node_runtime=nodes_runtime,
    #     queues_function=[None, None, None],
    # )
    # g3_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g3_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr3"
    # converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes),
    #     node_runtime=nodes_runtime,
    #     queues_function=[None, None, None, None, None],
    # )
    # g5_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g5_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr5"
    # converter.readout()

    # greedy_scheduler = GreedyScheduler(
    #     copy.deepcopy(ori_nodes),
    #     node_runtime=nodes_runtime,
    #     queues_function=[None, None, None, None, None, None, None, None, None, None],
    # )
    # g10_nodes = greedy_scheduler.apply()
    # converter.replace_nodes(g10_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr10"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g1_nodes))
    # g1s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g1s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr1s"
    # converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g2_nodes))
    g2s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g2s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr2s"
    converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g3_nodes))
    # g3s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g3s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr3s"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g5_nodes))
    # g5s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g5s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr5s"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g10_nodes))
    # g10s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g10s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr10s"
    # converter.readout()
