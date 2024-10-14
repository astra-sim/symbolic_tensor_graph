import copy
from symbolic_tensor_graph.chakra_remove_shortcuts import ChakraShortcutRemover

# from symbolic_tensor_graph.scheduler.greedy_scheduler import GreedyScheduler
from symbolic_tensor_graph.scheduler.baseline_greedy_scheduler import (
    BaselineGreedyScheduler as GreedyScheduler,
)
from symbolic_tensor_graph.symbolic2chakra_converter import Symbolic2ChakraConverter
from symbolic_tensor_graph.tensor_lifetime_extractor import TensorLifetimeExtractor
from models.old.transformer import transformer


if __name__ == "__main__":
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

    transformer(2, "sharding_spreadsheets/transformer/dp")
    converter = Symbolic2ChakraConverter(
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_2.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp",
        symbol_value_map,
        symbol_value_map["bp"] * symbol_value_map["tp"],
    )
    converter.convert()
    converter.readout()
    ori_nodes = converter.get_nodes()
    ori_nodes = copy.deepcopy(ori_nodes)

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes))
    g1_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g1_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g1"
    converter.readout()

    assert len(greedy_scheduler.queues) == 1
    tensor_lifetime_extractor = TensorLifetimeExtractor(
        converter.tensors, converter.tensor_node_maps, greedy_scheduler.queues[0]
    )
    tensor_lifetime_extractor.analysis_memory_operations()
    tensor_lifetime_extractor.to_records("symbolic_transformer2.w0l0i0.dp.g1.tlt.json")
    tensor_lifetime_extractor.visualize("symbolic_transformer2.w0l0i0.dp.g1.tvis.json")
    tensor_lifetime_extractor.parse_records(
        "symbolic_transformer2.w0l0i0.dp.g1.tlt.json"
    )
    tensor_lifetime_extractor.to_records(
        "symbolic_transformer2.w0l0i0.dp.g1.tlt.redump.json"
    )

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g1_nodes))
    g1s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g1s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer2.w0l0i0.dp.g1s"
    converter.readout()

    transformer(8, "sharding_spreadsheets/transformer/dp")
    converter = Symbolic2ChakraConverter(
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_8.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer8.w0l0i0.dp",
        symbol_value_map,
        symbol_value_map["bp"] * symbol_value_map["tp"],
    )
    converter.convert()
    converter.readout()
    ori_nodes = converter.get_nodes()
    ori_nodes = copy.deepcopy(ori_nodes)

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes))
    g1_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g1_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer8.w0l0i0.dp.g1"
    converter.readout()

    assert len(greedy_scheduler.queues) == 1
    tensor_lifetime_extractor = TensorLifetimeExtractor(
        converter.tensors, converter.tensor_node_maps, greedy_scheduler.queues[0]
    )
    tensor_lifetime_extractor.analysis_memory_operations()
    tensor_lifetime_extractor.to_records("symbolic_transformer8.w0l0i0.dp.g1.tlt.json")
    tensor_lifetime_extractor.visualize("symbolic_transformer8.w0l0i0.dp.g1.tvis.json")
    tensor_lifetime_extractor.parse_records(
        "symbolic_transformer8.w0l0i0.dp.g1.tlt.json"
    )
    tensor_lifetime_extractor.to_records(
        "symbolic_transformer8.w0l0i0.dp.g1.tlt.redump.json"
    )

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g1_nodes))
    g1s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g1s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer8.w0l0i0.dp.g1s"
    converter.readout()

    transformer(32, "sharding_spreadsheets/transformer/dp")
    converter = Symbolic2ChakraConverter(
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_32.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer32.w0l0i0.dp",
        symbol_value_map,
        symbol_value_map["bp"] * symbol_value_map["tp"],
    )
    converter.convert()
    converter.readout()
    ori_nodes = converter.get_nodes()
    ori_nodes = copy.deepcopy(ori_nodes)

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes))
    g1_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g1_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer32.w0l0i0.dp.g1"
    converter.readout()

    assert len(greedy_scheduler.queues) == 1
    tensor_lifetime_extractor = TensorLifetimeExtractor(
        converter.tensors, converter.tensor_node_maps, greedy_scheduler.queues[0]
    )
    tensor_lifetime_extractor.analysis_memory_operations()
    tensor_lifetime_extractor.to_records("symbolic_transformer32.w0l0i0.dp.g1.tlt.json")
    tensor_lifetime_extractor.visualize("symbolic_transformer32.w0l0i0.dp.g1.tvis.json")
    tensor_lifetime_extractor.parse_records(
        "symbolic_transformer32.w0l0i0.dp.g1.tlt.json"
    )
    tensor_lifetime_extractor.to_records(
        "symbolic_transformer32.w0l0i0.dp.g1.tlt.redump.json"
    )

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g1_nodes))
    g1s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g1s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer32.w0l0i0.dp.g1s"
    converter.readout()

    transformer(128, "sharding_spreadsheets/transformer/dp")
    converter = Symbolic2ChakraConverter(
        "sharding_spreadsheets/transformer/dp/processed_graphs/transformer_128.csv",
        "sharding_spreadsheets/transformer/validation/symbolic_transformer128.w0l0i0.dp",
        symbol_value_map,
        symbol_value_map["bp"] * symbol_value_map["tp"],
    )
    converter.convert()
    converter.readout()
    ori_nodes = converter.get_nodes()
    ori_nodes = copy.deepcopy(ori_nodes)

    greedy_scheduler = GreedyScheduler(copy.deepcopy(ori_nodes))
    g1_nodes = greedy_scheduler.apply()
    converter.replace_nodes(g1_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer128.w0l0i0.dp.g1"
    converter.readout()

    assert len(greedy_scheduler.queues) == 1
    tensor_lifetime_extractor = TensorLifetimeExtractor(
        converter.tensors, converter.tensor_node_maps, greedy_scheduler.queues[0]
    )
    tensor_lifetime_extractor.analysis_memory_operations()
    tensor_lifetime_extractor.to_records(
        "symbolic_transformer128.w0l0i0.dp.g1.tlt.json"
    )
    tensor_lifetime_extractor.visualize(
        "symbolic_transformer128.w0l0i0.dp.g1.tvis.json"
    )
    tensor_lifetime_extractor.parse_records(
        "symbolic_transformer128.w0l0i0.dp.g1.tlt.json"
    )
    tensor_lifetime_extractor.to_records(
        "symbolic_transformer128.w0l0i0.dp.g1.tlt.redump.json"
    )

    shortcut_remover = ChakraShortcutRemover(copy.deepcopy(g1_nodes))
    g1s_nodes = shortcut_remover.apply()
    converter.replace_nodes(g1s_nodes)
    converter.eg_file = "sharding_spreadsheets/transformer/validation/symbolic_transformer128.w0l0i0.dp.g1s"
    converter.readout()
