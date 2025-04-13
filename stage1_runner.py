import subprocess
import multiprocessing
import os
import time
import psutil

commands = [
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type dense --batch 64",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type moe --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type moe --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type moe --batch 64",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type moe --batch 128",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type moe --batch 256",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type moe --batch 512",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type dense --batch 128",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type dense --batch 256",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type dense --batch 512",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type dense --batch 1024",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type dense --batch 2048",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type dense --batch 4096",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type dense --batch 8192",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16384",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32768",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type moe --batch 1024",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type moe --batch 2048",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type moe --batch 4096",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type moe --batch 8192",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16384",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32768",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
]


def run_command(command):
    start_time = time.time()

    # shell_process = psutil.Popen(
    #     command, shell=True
    # )  # This tracks the shell, not Python
    
    shell_process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Wait a moment for the shell to spawn the Python process
    time.sleep(0.1)

    # Find the Python child process
    python_process = None
    for child in psutil.Process(shell_process.pid).children(recursive=True):
        if "python" in child.name().lower():
            python_process = child
            break

    if not python_process:
        raise RuntimeError("Could not find Python child process!")

    # Now monitor the Python process instead of the shell
    peak_memory = 0
    try:
        while True:
            try:
                mem = python_process.memory_info().rss
                peak_memory = max(peak_memory, mem)
            except psutil.NoSuchProcess:
                break  # Python process exited

            if not python_process.is_running():
                break  # Process finished

            time.sleep(0.1)  # Reduce CPU usage
    finally:
        # Clean up
        if shell_process.poll() is None:
            shell_process.terminate()
        if python_process.is_running():
            python_process.terminate()

    runtime = time.time() - start_time

    with open("results.txt", "a") as f:
        f.write(f"Command: {command}\n")
        f.write(f"Runtime: {runtime:.2f} seconds\n")
        f.write(f"Peak Memory: {peak_memory / (1024 * 1024):.2f} MB\n")
        f.write("\n")
        
    filename = command[command.find("--output_name")+len("--output_name"):].split()[0] + ".stdout"
    with open(filename, "w") as f:
        f.write(f"Command: {command}\n")
        f.write(shell_process.stdout.read())
        f.write(shell_process.stderr.read())

    return command, runtime, peak_memory


def main():
    with multiprocessing.Pool(processes=int(os.cpu_count() * 0.5)) as pool:
        results = pool.map(run_command, commands)

    with open("results_full.txt", "w") as f:
        for command, runtime, peak_memory in results:
            f.write(f"Command: {command}\n")
            f.write(f"Runtime: {runtime:.2f} seconds\n")
            f.write(f"Peak Memory: {peak_memory / (1024 * 1024):.2f} MB\n")
            f.write("\n")


if __name__ == "__main__":
    main()
