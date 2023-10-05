import copy
from symbolic_tensor_graph.chakra_remove_shortcuts import ChakraShortcutRemover
from symbolic_tensor_graph.scheduler import GreedyScheduler
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

    shortcut_remover = ChakraShortcutRemover(ori_nodes)
    sc_nodes = shortcut_remover.apply()
    converter.replace_nodes(sc_nodes)
    converter.eg_file = (
        "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.s"
    )

    converter.readout()

    greedy_scheduler = GreedyScheduler(copy.deepcopy(sc_nodes))
    scg1_nodes = greedy_scheduler.apply()
    converter.replace_nodes(scg1_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.sg1"
    converter.readout()

    greedy_scheduler = GreedyScheduler(copy.deepcopy(sc_nodes), num_queue=2)
    scg2_nodes = greedy_scheduler.apply()
    converter.replace_nodes(scg2_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.sg2"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(scg1_nodes))
    scg1s_nodes = shortcut_remover.apply()
    converter.replace_nodes(scg1s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.sg1s"
    converter.readout()

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(scg2_nodes))
    scg2s_nodes = shortcut_remover.apply()
    converter.replace_nodes(scg2s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.sg2s"
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
