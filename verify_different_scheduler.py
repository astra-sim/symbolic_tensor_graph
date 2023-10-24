import copy
from symbolic_tensor_graph.chakra_remove_shortcuts import ChakraShortcutRemover
from symbolic_tensor_graph.scheduler.baseline_greedy_scheduler import (
    BaselineGreedyScheduler,
)
from symbolic_tensor_graph.scheduler.random_scheduler import RandomScheduler

# from symbolic_tensor_graph.scheduler.improved_greedy_scheduler import (
#     ImprovedGreedyScheduler,
# )
# from symbolic_tensor_graph.scheduler.monotonous_greedy_scheduler import (
#     MonotonousGreedyScheduler,
# )
# from symbolic_tensor_graph.scheduler.monotonous_nearest_greedy_scheduler import (
#     MonotonousNearstGreedyScheduler,
# )

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

    converter.readout()

    # baseline_greedy_scheduler = BaselineGreedyScheduler(
    #     copy.deepcopy(ori_nodes), queues_function=[None, None]
    # )
    # g2_nodes = baseline_greedy_scheduler.apply()
    # converter.replace_nodes(g2_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g2"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g2_nodes))
    # g2s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g2s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g2s"
    # converter.readout()

    mono_greedy_scheduler = RandomScheduler(
        copy.deepcopy(ori_nodes), queues_function=[None, None]
    )
    mg2_nodes = mono_greedy_scheduler.apply()
    converter.replace_nodes(mg2_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mg2"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(mg2_nodes))
    mg2s_nodes = shortcut_remover.apply()
    converter.replace_nodes(mg2s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mg2s"
    converter.readout()

    # mono_nearst_greedy_scheduler = MonotonousNearstGreedyScheduler(
    #     copy.deepcopy(ori_nodes), queues_function=[None, None]
    # )
    # mng2_nodes = mono_nearst_greedy_scheduler.apply()
    # converter.replace_nodes(mng2_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mng2"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(mng2_nodes))
    # mng2s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(mng2s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mng2s"
    # converter.readout()

    # baseline_greedy_scheduler = BaselineGreedyScheduler(
    #     copy.deepcopy(ori_nodes),
    #     queues_function=[None, None],
    #     node_runtime=nodes_runtime,
    # )
    # g2_nodes = baseline_greedy_scheduler.apply()
    # converter.replace_nodes(g2_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr2"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g2_nodes))
    # g2s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(g2s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.gr2s"
    # converter.readout()

    # mono_greedy_scheduler = MonotonousGreedyScheduler(
    #     copy.deepcopy(ori_nodes),
    #     queues_function=[None, None],
    #     node_runtime=nodes_runtime,
    # )
    # mg2_nodes = mono_greedy_scheduler.apply()
    # converter.replace_nodes(mg2_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mgr2"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(mg2_nodes))
    # mg2s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(mg2s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mgr2s"
    # converter.readout()

    # mono_nearst_greedy_scheduler = MonotonousNearstGreedyScheduler(
    #     copy.deepcopy(ori_nodes),
    #     queues_function=[None, None],
    #     node_runtime=nodes_runtime,
    # )
    # mng2_nodes = mono_nearst_greedy_scheduler.apply()
    # converter.replace_nodes(mng2_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mngr2"
    # converter.readout()

    # shortcut_remover = ChakraShortcutRemover(copy.deepcopy(mng2_nodes))
    # mng2s_nodes = shortcut_remover.apply()
    # converter.replace_nodes(mng2s_nodes)
    # converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.mngr2s"
    # converter.readout()
