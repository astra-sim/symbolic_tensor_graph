import os, sys, multiprocessing, json
import numpy as np
import pandas as pd


results_dir = "/home/cman8/code/astra-sim-dev-internal/scripts/execution-graph-converter-Intel/symbolic/v9/experiment/dp_offload_random/run_file/results"
workload_dir = "/home/cman8/code/astra-sim-dev-internal/scripts/execution-graph-converter-Intel/symbolic/v9/experiment/dp_offload_random/run_file/ets"
json_out = "/home/cman8/code/astra-sim-dev-internal/scripts/execution-graph-converter-Intel/symbolic/v9/experiment/dp_offload_random/run_file/sorted_results.json"


def get_runtime_task(stdout_file):
    ticks = 1e15
    try:
        f = open(stdout_file, "r")
        lines = f.readlines()
        lines.reverse()
        for line in lines:
            line = line.strip()
            if not "tick=" in line:
                continue
            line = line[line.find("tick=") + 5 :]
            line = line[: line.find(",")]
            ticks = int(line)
            if ticks < 1e12:
                break
    except Exception as e:
        print(e)
        ticks = 1e15
    print(stdout_file, ticks)
    return ticks


def get_num_offload_task(offload_file):
    df = pd.read_csv(offload_file, encoding="utf-8", header=None)
    ndf = df.to_numpy()
    summ = ndf[:, 1].sum()
    print(offload_file, summ)
    return summ


def get_experiment_space(workload_dir_, results_dir_):
    run_ids = os.listdir(results_dir_)
    results_files = list()
    offload_files = list()
    for run_id in run_ids:
        results_files.append(os.path.join(results_dir_, run_id, "std.out"))
        offload_files.append(os.path.join(workload_dir_, run_id, "offload.csv"))
    # print(results_files)
    return run_ids, results_files, offload_files


def run_runtime_gather(num_workers=-1):
    if num_workers == -1:
        num_workers = int(multiprocessing.cpu_count() * 8)
    pool = multiprocessing.Pool(num_workers)
    run_ids, results_files, offload_files = get_experiment_space(
        workload_dir, results_dir
    )
    runtimes = pool.map(get_runtime_task, results_files)
    offloads = pool.map(get_num_offload_task, offload_files)
    results_map = dict()
    for k, r, o in zip(run_ids, runtimes, offloads):
        results_map[k] = (r, o)

    pool.close()
    pool.join()
    return results_map


if __name__ == "__main__":
    get_runtime_task(os.path.join(results_dir, "125", "std.out"))
    get_num_offload_task(os.path.join(workload_dir, "125", "offload.csv"))
    results = run_runtime_gather()
    f = open(json_out, "w")
    json.dump(results, f)
    f.close()
    # f = open(json_out, 'r')
    # results = json.load(f)
    # f.close()
    sorted_results = dict(sorted(results.items(), key=lambda kv: kv[1][0]))
    print(sorted_results)
    f = open(json_out, "w")
    json.dump(sorted_results, f)
    f.close()
