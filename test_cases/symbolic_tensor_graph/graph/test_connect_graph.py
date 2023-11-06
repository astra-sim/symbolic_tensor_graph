import unittest
import logging

logger = logging.getLogger("test/graph/test_connect_graph")

linear_filepath = "./sharding_spreadsheets/module/linear.csv"


class TestConnectGraph(unittest.TestCase):
    def test_connect_mlp2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph
        from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
        from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph

        save_path = "/tmp/TestConnectGraph_test_connect_mlp2.csv"
        pdf_path = "/tmp/TestConnectGraph_test_connect_mlp2.pdf"

        linear = TensorGraph.load_tensor_graph(linear_filepath)
        linear1 = ReplicateGraph.apply(
            graph=linear,
            tensor_name_template="%s1",
            old_symbol_map_new_symbol={"InFeat": "DModel", "OutFeat": "DFF"},
        )

        linear2 = ReplicateGraph.apply(
            graph=linear,
            tensor_name_template="%s2",
            old_symbol_map_new_symbol={"InFeat": "DFF", "OutFeat": "DModel"},
        )

        mlp2 = ConnectGraph.apply(
            graphs=[linear1, linear2],
            links={"y1": "x2", "dx2": "dy1"},
        )

        mlp2.save_tensor_graph(save_path)
        mlp2.visualize(pdf_path)
