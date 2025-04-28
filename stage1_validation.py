import os
import time
import re
import subprocess
import multiprocessing

gpt_5b = {
    "seq": 2048,
    "num_stacks": 24,
    "dmodel": 4096,
    "dff": 16384,
    "head": 32,
    "kvhead": 32,
    "model_type": "gpt",
    "dvocal": 8192,
    "mixed_precision": True
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
    "mixed_precision": True
}

gpt_5b_tpsp = {
    "output_name": "gpt_5b_tpsp",
    "micro_batch": 1,
    "batch": 1,
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
    "batch": 1,
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
    "batch": 1,
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
    "batch": 1,
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
    "batch": 1,
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
    "num_stacks": 80,
    "dmodel": 8192,
    "dff": 28672,
    "head": 64,
    "kvhead": 64,
    "model_type": "llama",
    "dvocal": 32000,
    "mixed_precision": True
}

llama3_16tp = {
    "output_name": "llama3_16tp",
    "tp": 16,
    "pp": 1,
    "micro_batch": 1,
    "batch": 1,
    "dp": 1,
    "cp": 1,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **llama3,
}

llama3_32tp = {
    "output_name": "llama3_32tp",
    "tp": 32,
    "pp": 1,
    "micro_batch": 1,
    "batch": 1,
    "dp": 1,
    "cp": 1,
    "ep": 1,
    "weight_sharded": False,
    "activation_recompute": False,
    **llama3,
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
    "batch": 1,
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
        #gpt_5b_cp,
        # gpt_5b_pp,
        #gpt_5b_fsdp,
        # gpt_5b_tpsp,
        gpt_175b_32tp,
        #gpt_175b_4tp2dp8pp,
        #llama3_4tp2pp,
        #llama3_8tp2pp,
        #llama3_8tp2dp,
        # llama3_16tp,
        # llama3_32tp,
    ]
    commands = list()
    for config in configs:
        commands.append(
            f"python3 stage1.py --output_dir {validation_workloads} --output_name {config['output_name']} --num_stacks {config['num_stacks']} "
            f"--dp {config['dp']} --tp {config['tp']} --sp {config['cp']} --ep {config['ep']} --pp {config['pp']} "
            f"--weight_sharded {config['weight_sharded']} --activation_recompute {config['activation_recompute']} "
            f"--micro_batch {config['micro_batch']} --seq {config['seq']} --dmodel {config['dmodel']} --dff {config['dff']} "
            f"--head {config['head']} --kvhead {config['kvhead']} --dvocal {config['dvocal']} "
            f"--model_type {config['model_type']} --batch {config['batch']} "
            f"--mixed_precision {config['mixed_precision'] if 'mixed_precision' in config else False}"
        )
        print(commands[-1])
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


if __name__ == "__main__":
    generate_stage_validation_workloads()
