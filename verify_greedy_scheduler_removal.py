import copy
from symbolic_tensor_graph.chakra_remove_shortcuts import ChakraShortcutRemover
from symbolic_tensor_graph.scheduler.greedy_scheduler import GreedyScheduler
from symbolic_tensor_graph.symbolic2chakra_converter import Symbolic2ChakraConverter


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
    converter.readout()

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes))
    g1_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g1_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g1"
    converter.readout()

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes), num_queue=2)
    g2_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g2_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g2"
    converter.readout()

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes), num_queue=3)
    g3_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g3_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g3"
    converter.readout()

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes), num_queue=5)
    g5_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g5_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g5"
    converter.readout()

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes), num_queue=10)
    g10_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g10_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g10"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g1_nodes))
    g1s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g1s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g1s"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g2_nodes))
    g2s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g2s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g2s"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g3_nodes))
    g3s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g3s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g3s"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g5_nodes))
    g5s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g5s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g5s"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g10_nodes))
    g10s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g10s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g10s"
    converter.readout()
