import unittest
import logging

logger = logging.getLogger("graph/test_graph")


class TestTensorGraph(unittest.TestCase):
    def test_tensor_graph_load_save1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear_layer.csv"
        resave_1_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save1_resave1.csv"
        resave_2_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save1_resave2.csv"

        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.save_tensor_graph(resave_1_file)
        graph2 = TensorGraph.load_tensor_graph(resave_1_file)
        self.assertEqual(graph, graph2)
        graph2.save_tensor_graph(resave_2_file)

    def test_tensor_graph_load_save2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/mlp2.csv"
        resave_1_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save2_resave1.csv"
        resave_2_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save2_resave2.csv"

        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.save_tensor_graph(resave_1_file)
        graph2 = TensorGraph.load_tensor_graph(resave_1_file)
        self.assertEqual(graph, graph2)
        graph2.save_tensor_graph(resave_2_file)

    def test_tensor_graph_load_save3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/test_ops.csv"
        resave_1_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save3_resave1.csv"
        resave_2_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save3_resave2.csv"

        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.save_tensor_graph(resave_1_file)
        graph2 = TensorGraph.load_tensor_graph(resave_1_file)
        self.assertEqual(graph, graph2)
        graph2.save_tensor_graph(resave_2_file)

    def test_tensor_graph_get_tensor_child_to_parent_link1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear_layer.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        child_to_parent = graph.get_tensor_child_to_parent_link()
        logger.debug(child_to_parent)

    def test_tensor_graph_get_tensor_child_to_parent_link2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/mlp2.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        child_to_parent = graph.get_tensor_child_to_parent_link()
        logger.debug(child_to_parent)

    def test_tensor_graph_get_tensor_child_to_parent_link3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/test_ops.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        child_to_parent = graph.get_tensor_child_to_parent_link()
        logger.debug(child_to_parent)
