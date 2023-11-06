import unittest
import logging

logger = logging.getLogger("test/graph/test_connect_graph")


class TestConnectGraph(unittest.TestCase):
    def test_connect_mlp2(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph
        from symbolic_tensor_graph.graph.connect_graph import ConnectGraph
        from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph

        self.assertTrue(False)
