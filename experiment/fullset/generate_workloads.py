#!/usr/bin/python3
import os, subprocess, multiprocessing

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"run fail! {command}")
    return True

def get_design_space():
    num_npus = 1024
    dp = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
    mp = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
    pp = {1, 2, 4, 8, 16, 32}
    sharded = {True, False}
    
    design_space = list()

    for ddp in dp:
        for mmp in mp:
            for ssharded in sharded:
                for ppp in pp:
                    ssp = num_npus // (ddp * mmp * ppp)
                    if ssp < 1:
                        continue
                    design_space.append((ddp, mmp, ssp, ppp, ssharded))
    return design_space

def generate_instance(design_point):
    root = os.path.join(
            os.path.split(
                os.path.abspath(__file__)
            )[0], "generated"
        )
    dp, mp, ssp, pp, sharded = design_point
    cmd = (
        f"python main_adv.py "
        f"--output_dir {root} "
        f"--output_name {dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.%d.et "
        f"--comm_group {dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.json "
        # f"--num_stacks 4 "
        f"--dp {dp} "
        f"--mp {mp} "
        f"--sp {ssp} "
        f"--pp {pp} "
        f"--weight_sharded {sharded} "
        f"--chakra_schema_version v0.0.4 "
        f"--din 12288 "
        f"--dout 12288 "
        f"--dff 49152 "
        f"--dmodel 12288 "
        f"--batch 8192 "
        f"--seq 2048 "
        f"--head 96 "
        f"--num_stacks 96 "
        f"--generate_io_info 1 "
    )
    cwd = os.path.join(
            os.path.split(
                os.path.abspath(__file__)
            )[0], "..", ".."
            )
    run_command(cmd, cwd)

if __name__ == '__main__':
    design_space = get_design_space()
    # with multiprocessing.Pool(int(multiprocessing.cpu_count()*0.8)) as pool:
    with multiprocessing.Pool(int(multiprocessing.cpu_count()*0.3)) as pool:
        pool.map(generate_instance, design_space)
    
