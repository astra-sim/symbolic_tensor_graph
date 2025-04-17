import subprocess
import multiprocessing
import os
import time
import psutil
import re

# palm540b
# python stage1.py --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48 --num_stacks 2 --output_name palm540b --batch 16

commands = [
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_8_4_4 --dp 4 --tp 8 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_16_4_4 --dp 4 --tp 16 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_32_4_4 --dp 4 --tp 32 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_64_4_4 --dp 4 --tp 64 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_128_4_4 --dp 4 --tp 128 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_256_4_4 --dp 4 --tp 256 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_512_4_4 --dp 4 --tp 512 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 118 --output_name palm_4_1024_4_4 --dp 4 --tp 1024 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type dense --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type dense --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type dense --batch 64",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type dense --batch 128",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type dense --batch 256",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type dense --batch 512",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type dense --batch 1024",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type dense --batch 2048",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type dense --batch 4096",
    "python stage1.py --output_dir generated_l/ --num_stacks 80 --output_name dense_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type dense --batch 8192",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16384",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32768",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type moe --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type moe --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type moe --batch 64",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type moe --batch 128",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type moe --batch 256",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type moe --batch 512",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type moe --batch 1024",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type moe --batch 2048",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type moe --batch 4096",
    "python stage1.py --output_dir generated_l/ --num_stacks 32 --output_name moe_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type moe --batch 8192",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16384",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32768",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
]

commands2 = [
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_8_4_4 --dp 4 --tp 8 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_16_4_4 --dp 4 --tp 16 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_32_4_4 --dp 4 --tp 32 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_64_4_4 --dp 4 --tp 64 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_128_4_4 --dp 4 --tp 128 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_256_4_4 --dp 4 --tp 256 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_512_4_4 --dp 4 --tp 512 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_1024_4_4 --dp 4 --tp 1024 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type dense --batch 64",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type dense --batch 128",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type dense --batch 256",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type dense --batch 512",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type dense --batch 1024",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type dense --batch 2048",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type dense --batch 4096",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type dense --batch 8192",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16384",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32768",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
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
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type moe --batch 1024",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type moe --batch 2048",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type moe --batch 4096",
    "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type moe --batch 8192",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16384",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32768",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
]

commands3 = [
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_8_4_4 --dp 4 --tp 8 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_16_4_4 --dp 4 --tp 16 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_32_4_4 --dp 4 --tp 32 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_64_4_4 --dp 4 --tp 64 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_128_4_4 --dp 4 --tp 128 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_256_4_4 --dp 4 --tp 256 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_512_4_4 --dp 4 --tp 512 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name palm_4_1024_4_4 --dp 4 --tp 1024 --sp 4 --ep 4 --model_type dense --batch 1024 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type dense --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type dense --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type dense --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 512",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type dense --batch 1024",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type dense --batch 2048",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type dense --batch 4096",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type dense --batch 8192",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16384",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32768",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name dense_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type dense --batch 131072",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16384",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32768",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type dense --batch 65536",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type moe --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type moe --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type moe --batch 256",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type moe --batch 512",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type moe --batch 1024",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type moe --batch 2048",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type moe --batch 4096",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type moe --batch 8192",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16384",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32768",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
    "python stage1.py --output_dir generated_lb/ --num_stacks 2 --output_name moe_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type moe --batch 131072",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2048_4_4_4 --dp 2048 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16384",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_4096_4_4_4 --dp 4096 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32768",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_8192_4_4_4 --dp 8192 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
    # # "python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_16384_4_4_4 --dp 16384 --tp 4 --sp 4 --ep 4 --model_type moe --batch 65536",
]


commands = commands3

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

    with open("results.txt", "a") as f:
        f.write(f"Command: {command}\n")
        f.write(f"Runtime: {runtime:.2f} seconds\n")
        f.write(f"Peak Memory: {peak_memory / (1024 * 1024):.2f} MB\n")
        f.write("\n")

    filename = (
        command[command.find("--output_name") + len("--output_name") :].split()[0]
        + ".stdout"
    )
    with open(os.path.join("./generated", filename), "w") as f:
        f.write(f"Command: {command}\n")
        f.write(stdout)
        f.write(stderr)

    return command, runtime, peak_memory


def main():
    # with multiprocessing.Pool(processes=int(os.cpu_count() * 0.8)) as pool:
    with multiprocessing.Pool(processes=12) as pool:
        results = pool.map(run_command, commands)
        # results = map(run_command, commands)

    with open("results_full.txt", "w") as f:
        for command, runtime, peak_memory in results:
            f.write(f"Command: {command}\n")
            f.write(f"Runtime: {runtime:.2f} seconds\n")
            f.write(f"Peak Memory: {peak_memory / (1024 * 1024):.2f} MB\n")
            f.write("\n")


if __name__ == "__main__":
    main()
