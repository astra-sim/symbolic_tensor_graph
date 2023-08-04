import sys
sys.path.append("..")

from tensor import Tensor
from .matmul import matmul, einsum
from .place_holder import place_holder