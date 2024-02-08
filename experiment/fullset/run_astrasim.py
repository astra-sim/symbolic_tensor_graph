#!/usr/bin/python3
import os, subprocess, multiprocessing

def run_command(command, cwd=None):
    result=subprocess.run(command, shell=True, cwd=cwd)
    return result.returncode == 0

def list_workloads(root):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".0.et"):
            filtered.append(os.path.join(root, file[:-5]))
    return filtered

def run_astrasim(workload_path):
    astrasim_root = "/home/changhai/code/astra-sim"
    if astrasim_root is None:
        raise Exception(f"please specify astrasim folder path at variable astrasim_root at {__file__}:run_astrasim()")
    file_dir = os.path.split(os.path.abspath(__file__))[0]
    astrasim_bin = os.path.join(astrasim_root, "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")
    system = os.path.join(file_dir, "system.json")
    network = os.path.join(file_dir, "network.yml")
    memory = os.path.join(file_dir, "memory.json")
    os.makedirs(os.path.join(file_dir, "outputs"), exist_ok=True)
    log = os.path.join(file_dir, "outputs", os.path.split(workload_path)[1]+".log")
    cmd = f"{astrasim_bin} --system-configuration={system} --workload-configuration={workload_path} --network-configuration={network} --remote-memory-configuration={memory} --comm-group-configuration={workload_path}.json --log-path={log}"
    print(cmd)
    success = run_command(cmd)
    if not success:
        return cmd
    return ""
    
if __name__ == '__main__':
    design_space = list_workloads("./generated")
    with multiprocessing.Pool(int(multiprocessing.cpu_count()*0.8)) as pool:
        failed_cmds=pool.map(run_astrasim, design_space)
        print("\n\nrunfails:")
        for cmd in failed_cmds:
            if not cmd == "":
                print(cmd)
        

