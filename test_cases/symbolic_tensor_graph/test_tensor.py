import unittest


class TestTensor(unittest.TestCase):
    def test_parse_to_records1(self):
        validation_file = "./sharding_spreadsheets/module/linear_layer.csv"
        resave_1_file = "/tmp/TestTensor_test_parse_to_records1_resave1.csv"
        resave_2_file = "/tmp/TestTensor_test_parse_to_records1_resave2.csv"
        from symbolic_tensor_graph.tensor import Tensor

        tensors1 = Tensor.parse_records(validation_file)
        Tensor.to_records(tensors1, resave_1_file)
        tensors2 = Tensor.parse_records(resave_1_file)
        self.assertEqual(tensors1, tensors2)
        Tensor.to_records(tensors2, resave_2_file)

    def test_op_handler1(self):
        validation_file = "./sharding_spreadsheets/module/linear_layer.csv"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        for tensor in tensors:
            print(
                f"{tensor.name} shape={tensor.y_shape} hidden={tensor.y_hidden} ops={tensor.ops}"
            )

    def test_parse_to_records2(self):
        validation_file = "./sharding_spreadsheets/module/mlp2.csv"
        resave_1_file = "/tmp/TestTensor_test_parse_to_records2_resave1.csv"
        resave_2_file = "/tmp/TestTensor_test_parse_to_records2_resave2.csv"
        from symbolic_tensor_graph.tensor import Tensor

        tensors1 = Tensor.parse_records(validation_file)
        Tensor.to_records(tensors1, resave_1_file)
        tensors2 = Tensor.parse_records(resave_1_file)
        self.assertEqual(tensors1, tensors2)
        Tensor.to_records(tensors2, resave_2_file)

    def test_op_handler2(self):
        validation_file = "./sharding_spreadsheets/module/mlp2.csv"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        for tensor in tensors:
            print(
                f"{tensor.name} shape={tensor.y_shape} hidden={tensor.y_hidden} ops={tensor.ops}"
            )

    def test_op_handler3(self):
        validation_file = "./sharding_spreadsheets/module/test_ops.csv"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        for tensor in tensors:
            print(
                f"{tensor.name} shape={tensor.y_shape} hidden={tensor.y_hidden} ops={tensor.ops}"
            )

    def test_visualize1(self):
        validation_file = "./sharding_spreadsheets/module/linear_layer.csv"
        visualize_file = "/tmp/TestTensor_test_visualize1.pdf"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        Tensor.visualize(tensors, visualize_file)

    def test_visualize2(self):
        validation_file = "./sharding_spreadsheets/module/mlp2.csv"
        visualize_file = "/tmp/TestTensor_test_visualize2.pdf"
        from symbolic_tensor_graph.tensor import Tensor

        tensors = Tensor.parse_records(validation_file)
        Tensor.visualize(tensors, visualize_file)


if __name__ == "__main__":
    unittest.main()
