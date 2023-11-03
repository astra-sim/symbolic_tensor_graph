from .add import Add
from .einsum import Einsum
from .element import Element
from .identical import Identical
from .place_holder import PlaceHolder
from .reshape import Reshape


class OPHandler:
    ops = [Add, Einsum, Element, Identical, PlaceHolder, Reshape]

    @classmethod
    def handle(cls, tensor):
        matched_op = cls.match_op(tensor)
        return matched_op.eval(tensor)

    @classmethod
    def match_op(cls, tensor):
        matched_op = None
        op_type = tensor.op_type
        for op in cls.ops:
            if op_type == op.type_name:
                matched_op = op
                break
        assert matched_op is not None
        return matched_op

    @classmethod
    def tokenrize(cls, tensor):
        matched_op = cls.match_op(tensor)
        token = matched_op.tokenrize(tensor)
        return token
