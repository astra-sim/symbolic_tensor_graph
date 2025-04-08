import copy
from symbolic_tensor_graph.tensor import Tensor
from symbolic_tensor_graph.ops import Element2


def reduce_chain(inputs, name, amp=None):
    if amp is None:
        amp = 1
    if not "%d" in name:
        name = name + "_%d"
    last = inputs[0]
    new_nodes = list()
    shape = inputs[0].y_shape
    hidden = inputs[0].y_hidden
    for i, input in enumerate(inputs):
        if i == 0:
            continue
        new_node = Tensor(True)
        new_node.name = name % (i,)
        new_node.require_grads = False
        new_node.x1 = last
        new_node.x2 = input
        new_node.op_type = Element2.type_name
        new_node.op_attr = str(amp)
        new_node.x1_shape = copy.deepcopy(shape)
        new_node.x1_hidden = copy.deepcopy(hidden)
        new_node.x2_shape = copy.deepcopy(shape)
        new_node.x2_hidden = copy.deepcopy(hidden)
        new_node.revision = last.revision

        Element2._sanity_check(new_node)

        last = new_node
        new_nodes.append(new_node)

    return new_nodes
