import sys
import logging
import unittest
from test_tensor import *
from graph.test_graph import *
from graph.test_replicate_graph import *
from graph.test_connect_graph import *


# class CustomizedFormatter(logging.Formatter):
#     def __init__(
#         self,
#         fmt=None,
#         datefmt=None,
#         style="%",
#         validate=True,
#         *,
#         defaults=None,
#     ) -> None:
#         if fmt is None:
#             fmt = "%(asctime)s - %(levelname)s -%(message)s"
#         super().__init__(fmt, datefmt, style, validate, defaults=defaults)

#     def format(self, record):
#         log_msg = super(CustomizedFormatter, self).format(record)
#         log_msg = f"{log_msg} - ({record.filename}, {record.lineno}, {record.funcName})"
#         return log_msg


logging.basicConfig(
    level=logging.DEBUG,
    filename="test.log",
    format="[%(name)s] %(asctime)s - %(levelname)s - %(message)s - (%(filename)s, %(lineno)d, %(funcName)s)",
)

if __name__ == "__main__":
    unittest.main()
