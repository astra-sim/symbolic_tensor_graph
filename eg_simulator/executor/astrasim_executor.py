import os, sys, tempfile, json, copy, shutil

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "../../chakra/et_def/"))
sys.path.append(os.path.join(file_dir, "../../chakra/third_party/utils/"))

from et_def_pb2 import *
from protolib import *


class AstraSIMExecutor:
    def __init__(
        self, system, network, memory, workloads, astrasim_bin, tmp_dir_root=None
    ):
        self.tmp_dir_root = tmp_dir_root

        self.system = os.path.abspath(system)
        assert os.path.exists(self.system)
        self.network = os.path.abspath(network)
        assert os.path.exists(self.network)
        self.memory = os.path.abspath(memory)
        assert os.path.exists(self.memory)
        self.update_workload(workloads)
        self.astrasim_bin = os.path.abspath(astrasim_bin)
        assert os.path.exists(self.astrasim_bin)

    def update_workload(self, workloads):
        self.tmp_dir = tempfile.TemporaryDirectory(
            "", "astrasim_runner", self.tmp_dir_root
        )
        self.workloads = list()
        if workloads is None:
            self.workloads = None
            self.workload = None
            return

        for i, workload in enumerate(workloads):
            if isinstance(workload, str):
                npu_id = i
                assert os.path.exists(workload)
                workload_path = os.path.abspath(
                    os.path.join(self.tmp_dir.name, f"workload.{npu_id}.eg")
                )
                shutil.copy(workload, workload_path)
                self.workloads.append(workload_path)
            elif isinstance(workload, list):
                # list of nodes
                npu_id = i
                workload_path = os.path.abspath(
                    os.path.join(self.tmp_dir.name, f"workload.{npu_id}.eg")
                )
                f = open(workload_path, "wb")
                for node in workload:
                    encodeMessage(f, node)
                f.close()
                self.workloads.append(workload_path)
            elif isinstance(workload, Node):
                # once detect the workloads is nodes for a single npu,
                # then finished the whole workloads,
                # thus should not be second iteration
                assert i == 0
                network_f = open(self.network, "r")
                network_cfg = json.load(network_f)
                network_f.close()
                num_npu = 1
                if "units-count" in network_cfg:
                    for units_this_dim in network_cfg["units-count"]:
                        num_npu *= int(units_this_dim)
                elif "physical-dims" in network_cfg:
                    for units_this_dim in network_cfg["physical-dims"]:
                        num_npu *= int(units_this_dim)
                else:
                    assert False
                for npu_id in range(num_npu):
                    workload_path = os.path.abspath(
                        os.path.join(self.tmp_dir.name, f"workload.{npu_id}.eg")
                    )
                    f = open(workload_path, "wb")
                    for node in workloads:
                        assert isinstance(node, Node)
                        encodeMessage(f, node)
                    f.close()
                    self.workloads.append(workload_path)
                break
            else:
                assert False

        self.workload = os.path.abspath(os.path.join(self.tmp_dir.name, f"workload"))

    def run(self):
        assert not self.workload is None
        cmd = f"{self.astrasim_bin} \
                --workload-configuration={self.workload} \
                --system-configuration={self.system} \
                --network-configuration={self.network} \
                --remote-memory-configuration={self.memory}"
        pipe_out = os.popen(cmd, "r")
        lines = pipe_out.readlines()

        cycles = -1
        for line in reversed(lines):
            if not "sys[" in line:
                continue
            cycles = int(line[line.find(",") + 1 : line.rfind("cycles")].strip())
            break
        if cycles == -1:
            print(lines, file=sys.stderr)
        return cycles
