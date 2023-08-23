import os, sys, multiprocessing


results_dir = '/home/cman8/code/astra-sim-dev-internal/scripts/execution-graph-converter-Intel/symbolic/v9/experiment/dp_offload_random/run_file/results'
astrasim_bin = '/home/cman8/code/astra-sim-dev-internal/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra'
workload_dir = '/home/cman8/code/astra-sim-dev-internal/scripts/execution-graph-converter-Intel/symbolic/v9/experiment/dp_offload_random/run_file/ets'
system_file = '/home/cman8/code/astra-sim-dev-internal/scripts/execution-graph-converter-Intel/symbolic/v9/experiment/dp_offload_random/run_file/system_MOE_1T_256_Cards_256_default_2048_4096_2048_512.txt'
network_file = '/home/cman8/code/astra-sim-dev-internal/scripts/execution-graph-converter-Intel/symbolic/v9/experiment/dp_offload_random/run_file/network_MOE_1T_256_Cards_256_default_2048_4096_2048_512.txt'


def run_task(id_, workload, system, network, run_name_prefix=''):
    run_dir = os.path.join(results_dir, id_)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'std.out')
    cmd = f"{astrasim_bin} --workload-configuration={workload} --system-configuration={system} --network-configuration={network} --path={run_dir} --run_name={id_} > {log_path} 2>&1"
    print(cmd)
    os.system(cmd)
    return True


def get_experiment_space(workload_dir_, system_file_, network_file_):
    ret = list()
    ids = os.listdir(workload_dir_)
    for id_ in ids:
        ret.append((id_, os.path.join(workload_dir_, id_, "transformer"), system_file_, network_file_))
    return ret


def run_experiments(num_workers=-1):
    if num_workers == -1:
        num_workers = int(multiprocessing.cpu_count()*1)
    pool = multiprocessing.Pool(num_workers)
    experiment_space = get_experiment_space(workload_dir, system_file, network_file)
    rets = list()
    for experiment_point in experiment_space:
        pool.apply_async(run_task, experiment_point)
    pool.close()
    pool.join()
    for ret in rets:
        ret.get()
    return


if __name__ == '__main__':
    run_experiments()
    