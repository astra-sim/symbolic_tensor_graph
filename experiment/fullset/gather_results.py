#!/usr/bin/python3
import os, subprocess, multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def list_logs(root):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".log"):
            if len(file.split(".")) > 2:
                continue
            filtered.append(os.path.join(root, file))
    return filtered

def extract_runtime(log_path):
    log_filename = os.path.split(log_path)[-1]
    dp, mp, sp, pp, sharded = log_filename[:-4].split('_')
    dp, mp, sp, pp, sharded = int(dp), int(mp), int(sp), int(pp), int(sharded)
    runtime = -1
    f = open(log_path, 'r')
    log_lines = f.readlines()
    f.close()
    if len(log_lines) < 10:
        runtime = -1
    else:
        for i, line in enumerate(reversed(log_lines)):
            if i > 10:
                break
            elif "cycles" in line:
                line = line[line.rfind(',')+1:]
                line = line.replace('cycles', '').strip()
                runtime = int(line)
                break
    return dp, mp, sp, pp, sharded, runtime

def gather_runtimes(root):
    logs = list_logs(root)
    runtimes = None
    with multiprocessing.Pool() as pool:
        runtimes = pool.map(extract_runtime, logs)
        # runtimes = map(extract_runtime, logs)
    runtimes_dict = dict()
    for dp, mp, sp, pp, sharded, runtime in runtimes:
        runtimes_dict[(dp, mp, sp, pp, sharded)] = runtime
    return runtimes_dict

def get_fails(runtimes):
    fail_cases = list()
    for key in runtimes.keys():
        if runtimes[key] == -1:
            fail_cases.append(key)
            print(key)
    return fail_cases

def visualize1(runtimes, ssp, sharded):
    max_runtimes = max(runtimes.values())
    mat = -1*np.ones((7, 7))
    # vis all data, x=(dp, mp) y=(sp, pp)
    for ddp in range(7):
        x_value = ddp
        for mmp in range(7):
            y_value = mmp
            for ssp in {ssp}:
                ppp = 6-ddp-mmp-ssp
                rddp, rmmp = int(2**ddp), int(2**mmp)
                rssp, rppp = int(2**ssp), int(2**ppp)
                key = (rddp, rmmp, rssp, rppp, sharded)
                if not key in runtimes:
                    mat[x_value, y_value] = -1
                elif runtimes[key]==-1:
                    mat[x_value, y_value] = -1
                else:
                    mat[x_value, y_value] = runtimes[key]/max_runtimes
    plt.figure(dpi=120)
    sns.heatmap(mat)
    plt.title(f"dp vs mp, x=dp y=mp, sp={ssp}, sharded={sharded}")
    return plt

def visualize2(runtimes, sharded):
    max_runtimes = max(runtimes.values())
    mat = -1*np.ones((7*7, 7))
    # vis all data, x=(dp, mp) y=(sp, pp)
    for ddp in range(7):
        for mmp in range(7):
            x_value = ddp*7+mmp
            for ssp in range(7):
                y_value = ssp
                ppp = 6-ddp-mmp-ssp
                rddp, rmmp = int(2**ddp), int(2**mmp)
                rssp, rppp = int(2**ssp), int(2**ppp)
                key = (rddp, rmmp, rssp, rppp, sharded)
                if not key in runtimes:
                    mat[x_value, y_value] = -1
                elif runtimes[key]==-1:
                    mat[x_value, y_value] = -1
                else:
                    mat[x_value, y_value] = 1-runtimes[key]/max_runtimes
    plt.figure(dpi=120)
    sns.heatmap(mat)
    plt.title(f"all results, x=(dp, mp) y=sp, sharded={sharded}")
    return plt

def serialize_results(runtimes, json_filename):
    def get_jsonable_dict(dict_):
        ret = dict()
        for key in dict_.keys():
            ret[str(key)] = dict_[key]
        return ret
    import json
    f = open(json_filename, 'w')
    json.dump(get_jsonable_dict(runtimes), f, indent=4)
    f.close()
    
def topk(runtimes, k=10):
    if k == 0:
        top_k_items = sorted(runtimes.items(), key=lambda x: -x[1] if x[1]!=-1 else -1e100, reverse=True)
    else:
        top_k_items = sorted(runtimes.items(), key=lambda x: -x[1] if x[1]!=-1 else -1e100, reverse=True)[:k]
    for key, value in top_k_items:
        print(f"{key}: {value}")
    return top_k_items

    
if __name__ == '__main__':
    runtimes = gather_runtimes("./outputs")
    hook = 0
    # for sp in range(7):
    #     plt1 = visualize1(runtimes, sp, 0)
    #     plt.savefig(f"dpvsmp_{sp}_ns.png")
    #     plt1 = visualize1(runtimes, sp, 1)
    #     plt.savefig(f"dpvsmp_{sp}_s.png")
    # plt1 = visualize2(runtimes, 0)
    # plt.savefig(f"all_ns.png")
    # plt1 = visualize2(runtimes, 1)
    # plt.savefig(f"all_s.png")

    serialize_results(runtimes, "./results.json")
    
    print("top:")
    topk(runtimes, k=0)
    print("\n\n\nfail cases")
    fails = get_fails(runtimes)

    print(f"\n\n\n fail/total={len(fails)/len(runtimes)}")
