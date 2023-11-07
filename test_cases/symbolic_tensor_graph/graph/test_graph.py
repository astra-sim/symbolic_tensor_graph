import unittest
import logging

logger = logging.getLogger("test/graph/test_graph")


class TestTensorGraph(unittest.TestCase):
    def test_tensor_graph_load_save1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear.csv"
        resave_1_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save1_resave1.csv"
        resave_2_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save1_resave2.csv"

        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.save_tensor_graph(resave_1_file)
        graph2 = TensorGraph.load_tensor_graph(resave_1_file)
        for tensor1, tensor2 in zip(graph.tensors, graph2.tensors):
            self.assertEqual(tensor1._to_record(), tensor2._to_record())
        graph2.save_tensor_graph(resave_2_file)

    def test_tensor_graph_load_save2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        resave_1_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save2_resave1.csv"
        resave_2_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save2_resave2.csv"

        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.save_tensor_graph(resave_1_file)
        graph2 = TensorGraph.load_tensor_graph(resave_1_file)
        for tensor1, tensor2 in zip(graph.tensors, graph2.tensors):
            self.assertEqual(tensor1._to_record(), tensor2._to_record())
        graph2.save_tensor_graph(resave_2_file)

    def test_tensor_graph_load_save3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/test_ops.csv"
        resave_1_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save3_resave1.csv"
        resave_2_file = "/tmp/TestTensorGraph_test_tensor_graph_load_save3_resave2.csv"

        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.save_tensor_graph(resave_1_file)
        graph2 = TensorGraph.load_tensor_graph(resave_1_file)
        for tensor1, tensor2 in zip(graph.tensors, graph2.tensors):
            self.assertEqual(tensor1._to_record(), tensor2._to_record())
        graph2.save_tensor_graph(resave_2_file)

    def test_tensor_graph_get_tensor_child_to_parent_link1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        child_to_parent = graph.get_tensor_child_to_parent_link()
        logger.debug(child_to_parent)

    def test_tensor_graph_get_tensor_child_to_parent_link2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        child_to_parent = graph.get_tensor_child_to_parent_link()
        logger.debug(child_to_parent)

    def test_tensor_graph_get_tensor_child_to_parent_link3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/test_ops.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        child_to_parent = graph.get_tensor_child_to_parent_link()
        logger.debug(child_to_parent)

    def test_tensor_graph_get_tensor_parent_to_child_link1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        parent_to_child = graph.get_tensor_parent_to_child_link()
        logger.debug(parent_to_child)

    def test_tensor_graph_get_tensor_parent_to_child_link2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        parent_to_child = graph.get_tensor_parent_to_child_link()
        logger.debug(parent_to_child)

    def test_tensor_graph_get_tensor_parent_to_child_link3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/test_ops.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        parent_to_child = graph.get_tensor_parent_to_child_link()
        logger.debug(parent_to_child)

    def test_tensor_graph_get_dimensions_1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        dims = graph.get_dimensions()
        logger.debug(dims)

    def test_tensor_graph_get_dimensions_2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        dims = graph.get_dimensions()
        logger.debug(dims)

    def test_tensor_graph_get_dimensions_3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/test_ops.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        dims = graph.get_dimensions()
        logger.debug(dims)

    def test_tensor_graph_get_symbols_1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        syms = graph.get_symbols()
        logger.debug(syms)

    def test_tensor_graph_get_symbols_2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        syms = graph.get_symbols()
        logger.debug(syms)

    def test_tensor_graph_get_symbols_3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/test_ops.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        syms = graph.get_symbols()
        logger.debug(syms)

    def test_tensor_graph_sanity_check_1(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/module/linear.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.sanity_check()

    def test_tensor_graph_sanity_check_2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/mlp2.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.sanity_check()

    def test_tensor_graph_sanity_check_3(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph

        validation_file = "./sharding_spreadsheets/test/test_ops.csv"
        graph = TensorGraph.load_tensor_graph(validation_file)
        graph.sanity_check()
