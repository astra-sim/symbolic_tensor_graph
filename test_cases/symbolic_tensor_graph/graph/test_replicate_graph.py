import unittest
import logging

logger = logging.getLogger("test/graph/test_replicate_graph")

linear_filepath = "./sharding_spreadsheets/module/linear.csv"
mlp2_filepath = "./sharding_spreadsheets/test/mlp2.csv"
test_ops_filepath = "./sharding_spreadsheets/test/test_ops.csv"


class TestReplicateGraph(unittest.TestCase):
    def test_replicate_linear(self):
        from symbolic_tensor_graph.graph.graph import TensorGraph
        from symbolic_tensor_graph.graph.replicate_graph import ReplicateGraph

        save_path = "/tmp/TestReplicateGraph_test_replicate_linear.csv"
        pdf_path = "/tmp/TestReplicateGraph_test_replicate_linear.pdf"

        linear_ori = TensorGraph.load_tensor_graph(linear_filepath)
        new_name_template = "stack_1_1_%s"
        new_replica = "114514"
        new_dims = {"InFeat": "DModel", "OutFeat": "DFF"}
        linear_new = ReplicateGraph.apply(
            linear_ori,
            tensor_name_template=new_name_template,
            new_revision=new_replica,
            old_symbol_map_new_symbol=new_dims,
        )

        linear_new.save_tensor_graph(save_path)
        linear_new.visualize(pdf_path)
