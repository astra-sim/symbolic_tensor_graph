import os
import time
import re
import subprocess
import multiprocessing
from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.et_def.et_def_pb2 import (
    Node,
    AttributeProto as ChakraAttr,
    NodeType,
    CollectiveCommType,
    GlobalMetadata
)
from symbolic_tensor_graph.chakra.backends.chakra_00_4_backend.protolib import *

gpt_5b = {
    "seq": 2048,
    "num_stacks": 24,
    "dmodel": 4096,
    "dff": 16384,
    "head": 32,
    "kvhead": 32,
    "model_type": "gpt",
    "dvocal": 8192,
}

gpt_175b = {
    "seq": 2048,
    "num_stacks": 96,
    "dmodel": 12288,
    "dff": 49152,
    "head": 96,
    "kvhead": 96,
    "model_type": "gpt",
    "dvocal": 8192,
}

gpt_5b_tpsp = {
    "output_name": "gpt_5b_tpsp",
    "micro_batch": 1,
    "batch": 128,
    "dp": 1,
    "cp": 1,
    "pp": 1,
    "tp": 8,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **gpt_5b,
}

gpt_5b_fsdp = {
    "output_name": "gpt_5b_fsdp",
    "micro_batch": 1,
    "batch": 128,
    "dp": 8,
    "cp": 1,
    "pp": 1,
    "tp": 1,
    "ep": 1,
    "weight_sharded": True,
    "activation_recompute": False,
    **gpt_5b,
}

gpt_5b_cp = {
    "output_name": "gpt_5b_cp",
    "micro_batch": 1,
    "batch": 128,
    "dp": 1,
    "cp": 8,
    "pp": 1,
    "tp": 1,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **gpt_5b,
}

gpt_5b_pp = {
    "output_name": "gpt_5b_pp",
    "micro_batch": 1,
    "batch": 128,
    "dp": 1,
    "cp": 1,
    "pp": 8,
    "tp": 1,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **gpt_5b,
}

gpt_175b_32tp = {
    "output_name": "gpt_175b_32tp",
    "micro_batch": 1,
    "batch": 32,
    "dp": 1,
    "cp": 1,
    "pp": 1,
    "tp": 32,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **gpt_175b,
}

gpt_175b_4tp2dp8pp = {
    "output_name": "gpt_175b_4tp2dp8pp",
    "micro_batch": 1,
    "batch": 128,
    "dp": 2,
    "cp": 1,
    "pp": 8,
    "tp": 4,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **gpt_175b,
}

llama3 = {
    "seq": 8192,
    "num_stacks": 32,
    "dmodel": 4096,
    "dff": 14336,
    "head": 32,
    "kvhead": 8,
    "model_type": "llama",
    "dvocal": 32000,
}

llama3_4tp2pp = {
    "output_name": "llama3_4tp2pp",
    "tp": 4,
    "pp": 2,
    "micro_batch": 1,
    "batch": 128,
    "dp": 1,
    "cp": 1,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **llama3,
}

llama3_8tp2pp = {
    "output_name": "llama3_8tp2pp",
    "batch": 128,
    "micro_batch": 1,
    "dp": 1,
    "cp": 1,
    "tp": 8,
    "pp": 1,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **llama3,
}

llama3_8tp2dp = {
    "output_name": "llama3_8tp2dp",
    "batch": 128,
    "micro_batch": 1,
    "dp": 2,
    "cp": 1,
    "tp": 8,
    "pp": 1,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **llama3,
}

validation_workloads = "./validation"


def run_command(command):
    start_time = time.time()

    shell_process = subprocess.Popen(
        f"/usr/bin/time -v {command}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = shell_process.communicate()
    runtime = time.time() - start_time

    peak_memory = 0
    match_ = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)

    if match_:
        peak_memory = int(match_.group(1)) * 1024

    with open(os.path.join(validation_workloads, "stage_gen_results.txt"), "a") as f:
        f.write(f"Command: {command}\n")
        f.write(f"Runtime: {runtime:.2f} seconds\n")
        f.write(f"Peak Memory: {peak_memory / (1024 * 1024):.2f} MB\n")
        f.write("\n")

    filename = (
        command[command.find("--output_name") + len("--output_name") :].split()[0]
        + ".stdout"
    )
    with open(os.path.join(validation_workloads, filename), "w") as f:
        f.write(f"Command: {command}\n")
        f.write(stdout)
        f.write(stderr)

    return command, runtime, peak_memory


def generate_commands():
    configs = [
        gpt_5b_cp,
        gpt_5b_pp,
        gpt_5b_fsdp,
        gpt_5b_tpsp,
        gpt_175b_32tp,
        gpt_175b_4tp2dp8pp,
        llama3_4tp2pp,
        llama3_8tp2pp,
        llama3_8tp2dp,
    ]
    commands = list()
    for config in configs:
        commands.append(
            f"python stage1.py --output_dir {validation_workloads} --output_name {config['output_name']} --num_stacks {config['num_stacks']} "
            f"--dp {config['dp']} --tp {config['tp']} --sp {config['cp']} --ep {config['ep']} --pp {config['pp']} "
            f"--weight_sharded {config['weight_sharded']} --activation_recompute {config['activation_recompute']} "
            f"--micro_batch {config['micro_batch']} --seq {config['seq']} --dmodel {config['dmodel']} --dff {config['dff']} "
            f"--head {config['head']} --kvhead {config['kvhead']} --dvocal {config['dvocal']} "
            f"--model_type {config['model_type']} --batch {config['batch']}"
        )
    return commands


def generate_stage_validation_workloads():
    if not os.path.exists(validation_workloads):
        os.makedirs(validation_workloads)

    commands = generate_commands()
    with multiprocessing.Pool(processes=24) as pool:
        results = pool.map(run_command, commands)

    with open(
        os.path.join(validation_workloads, "stage_gen_results_full.txt"), "w"
    ) as f:
        for command, runtime, peak_memory in results:
            f.write(f"Command: {command}\n")
            f.write(f"Runtime: {runtime:.2f} seconds\n")
            f.write(f"Peak Memory: {peak_memory / (1024 * 1024):.2f} MB\n")
            f.write("\n")

def extract_nodes_from_chakra(chakra_trace_filename):
    freq = {
        "gemm": 0,
        "attn": 0,
        "elementWise": 0,
        "others": 0,
        "p2p": 0,
        "AR": 0,
        "A2A": 0,
        "AG": 0,
        "RS": 0
    }
    f = open(chakra_trace_filename, "rb")
    global_metadata = GlobalMetadata()
    decodeMessage(f, global_metadata)
    node = Node()
    while decodeMessage(f, node):
        if node.type == NodeType.COMM_COLL_NODE:
            comm_type = None
            for attr in node.attr:
                if attr.name == "comm_type":
                    comm_type = attr.int64_val
                    break
            if comm_type is None:
                assert False
            elif comm_type == CollectiveCommType.ALL_REDUCE:
                freq['AR'] += 1
            elif comm_type == CollectiveCommType.ALL_GATHER:
                freq['AG'] += 1
            elif comm_type == CollectiveCommType.REDUCE_SCATTER:
                freq['RS'] += 1
            elif comm_type == CollectiveCommType.ALL_TO_ALL:
                freq['A2A'] += 1
            else:
                assert False
        elif node.type == NodeType.COMM_SEND_NODE:
            freq['p2p'] += 1
        elif node.type == NodeType.COMM_RECV_NODE:
            freq['p2p'] += 1
        elif node.type == NodeType.COMP_NODE:
            if "mha.attn_kernel" in node.name:
                freq["attn"] += 1
                continue
            op_type = None
            for attr in node.attr:
                if attr.name == "op_type":
                    op_type = attr.string_val
            if op_type == 'M':
                freq['gemm'] += 1
            elif op_type == 'A':
                freq['elementWise'] += 1
            elif op_type == 'E':
                freq['elementWise'] += 1
            elif op_type in {"SLICE", "B", "CUSTOM"}:
                continue
            else:
                print(op_type)
                assert False
    return freq


def extract_chakras_freqs():
    files = [
        "./validation/gpt_5b_fsdp.0.et",
        "./validation/gpt_5b_cp.0.et",
        "./validation/gpt_5b_pp.0.et",
        "./validation/gpt_5b_tpsp.0.et",
        "./validation/llama3_4tp2pp.0.et",
        "./validation/llama3_8tp2pp.0.et",
    ]

    freqs = dict()
    for file in files:
        freqs[file] = extract_nodes_from_chakra(file)
    print(freqs)
    import json
    with open("freqs.04131705.json", "w") as f:
        json.dump(freqs, f)

if __name__ == "__main__":
    generate_stage_validation_workloads()
    # extract_chakras_freqs()
