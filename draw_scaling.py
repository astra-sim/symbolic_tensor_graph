import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.ticker import MultipleLocator

data = """
Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type moe --batch 16
Runtime: 50.06 seconds
Peak Memory: 175.95 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type dense --batch 16
Runtime: 2.90 seconds
Peak Memory: 141.43 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 4.19 seconds
Peak Memory: 141.50 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16
Runtime: 3.06 seconds
Peak Memory: 142.00 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type dense --batch 16
Runtime: 3.13 seconds
Peak Memory: 142.07 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_16_4_4 --dp 4 --tp 16 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 9.45 seconds
Peak Memory: 143.30 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_8_4_4 --dp 4 --tp 8 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 5.84 seconds
Peak Memory: 142.61 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type dense --batch 16
Runtime: 3.56 seconds
Peak Memory: 141.68 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type dense --batch 32
Runtime: 4.18 seconds
Peak Memory: 142.30 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type dense --batch 64
Runtime: 5.72 seconds
Peak Memory: 142.65 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type dense --batch 128
Runtime: 9.53 seconds
Peak Memory: 143.45 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_32_4_4 --dp 4 --tp 32 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 18.67 seconds
Peak Memory: 145.74 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type dense --batch 256
Runtime: 17.87 seconds
Peak Memory: 144.49 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_64_4_4 --dp 4 --tp 64 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 43.33 seconds
Peak Memory: 149.34 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type dense --batch 512
Runtime: 39.76 seconds
Peak Memory: 149.18 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_128_4_4 --dp 4 --tp 128 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 121.26 seconds
Peak Memory: 159.52 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type dense --batch 1024
Runtime: 106.80 seconds
Peak Memory: 157.95 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_2_2 --dp 2 --tp 4 --sp 2 --ep 2 --model_type moe --batch 16
Runtime: 6.46 seconds
Peak Memory: 145.66 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 16
Runtime: 5.53 seconds
Peak Memory: 143.19 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_2_4 --dp 2 --tp 4 --sp 2 --ep 4 --model_type moe --batch 16
Runtime: 5.44 seconds
Peak Memory: 143.87 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_4_4_4 --dp 2 --tp 4 --sp 4 --ep 4 --model_type moe --batch 16
Runtime: 6.42 seconds
Peak Memory: 142.63 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_4_4_4_4 --dp 4 --tp 4 --sp 4 --ep 4 --model_type moe --batch 32
Runtime: 7.77 seconds
Peak Memory: 144.03 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_8_4_4_4 --dp 8 --tp 4 --sp 4 --ep 4 --model_type moe --batch 64
Runtime: 10.95 seconds
Peak Memory: 144.80 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_16_4_4_4 --dp 16 --tp 4 --sp 4 --ep 4 --model_type moe --batch 128
Runtime: 19.11 seconds
Peak Memory: 145.55 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_32_4_4_4 --dp 32 --tp 4 --sp 4 --ep 4 --model_type moe --batch 256
Runtime: 37.16 seconds
Peak Memory: 148.86 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_64_4_4_4 --dp 64 --tp 4 --sp 4 --ep 4 --model_type moe --batch 512
Runtime: 83.18 seconds
Peak Memory: 155.51 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type dense --batch 2048
Runtime: 333.49 seconds
Peak Memory: 180.64 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_256_4_4 --dp 4 --tp 256 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 391.46 seconds
Peak Memory: 183.82 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_128_4_4_4 --dp 128 --tp 4 --sp 4 --ep 4 --model_type moe --batch 1024
Runtime: 222.25 seconds
Peak Memory: 167.51 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_256_4_4_4 --dp 256 --tp 4 --sp 4 --ep 4 --model_type moe --batch 2048
Runtime: 794.13 seconds
Peak Memory: 201.41 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type dense --batch 4096
Runtime: 1355.06 seconds
Peak Memory: 232.49 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name palm_4_512_4_4 --dp 4 --tp 512 --sp 4 --ep 4 --model_type dense --batch 16 --dvocal 8192 --dmodel 18432 --dff 73728 --seq 2048 --head 48 --kvhead 48
Runtime: 1684.55 seconds
Peak Memory: 237.98 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_512_4_4_4 --dp 512 --tp 4 --sp 4 --ep 4 --model_type moe --batch 4096
Runtime: 3047.66 seconds
Peak Memory: 271.46 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name dense_1024_4_4_4 --dp 1024 --tp 4 --sp 4 --ep 4 --model_type dense --batch 8192
Runtime: 5600.71 seconds
Peak Memory: 334.04 MB

Command: python stage1.py --output_dir generated/ --num_stacks 2 --output_name moe_2_2_2_2 --dp 2 --tp 2 --sp 2 --ep 2 --model_type moe --batch 16
Runtime: 5.61 seconds
Peak Memory: 145.43 MB


"""

lines = data.strip().split('\n')
commands = []
runtimes = []
memories = []
models = []

MODEL_TYPE_MAP = {"dense": "llama-70b", "moe": "mixtral 8x7b", "palm": "palm-540b"}

# Process each group of 3 lines, separated by a line space
for i in range(0, len(lines), 4):
    command_line = lines[i]
    runtime_line = lines[i + 1]
    memory_line = lines[i + 2]

    # Parse the command line
    parts = command_line.split(' --')
    command = {x.split(' ')[0]: x.split(' ')[1] for x in parts}
    commands.append(command)

    # Extract runtime
    runtime = float(runtime_line.split(': ')[1].strip().split(' ')[0])
    runtimes.append(runtime)

    # Extract peak memory
    memory = float(memory_line.split(': ')[1].strip().split(' ')[0])
    memories.append(memory)

    # Extract and map model type from output_name
    output_name = command['output_name']
    model_type = output_name.split('_')[0]
    model_type = MODEL_TYPE_MAP.get(model_type, model_type)
    models.append(model_type)

# Calculate system scale
scales = [int(cmd['dp']) * int(cmd['tp']) * int(cmd['sp']) * int(cmd['ep']) for cmd in commands]

# Aggregate data
runtime_data = defaultdict(list)
memory_data = defaultdict(list)

for scale, model, runtime, memory in zip(scales, models, runtimes, memories):
    runtime_data[(model, scale)].append(runtime)
    memory_data[(model, scale)].append(memory)

# Calculate averages
average_runtimes = {key: np.mean(values) for key, values in runtime_data.items()}
average_memories = {key: np.mean(values) for key, values in memory_data.items()}

# Prepare data for plotting
model_types = sorted(set(models))
unique_scales = sorted(set(scales))
ind = np.arange(len(unique_scales))  # x locations for the groups

plt.figure(figsize=(10, 7.5))
bar_width = 0.25  # Fixed width for each bar

# Plot runtime bar chart
ax1 = plt.subplot(2, 1, 1)  # Change to (2, 1, 1) to stack vertically
for idx, model_type in enumerate(model_types):
    bar_positions = ind + idx * bar_width
    ax1.bar(bar_positions,
            [average_runtimes.get((model_type, scale), 0) for scale in unique_scales],
            bar_width, label=model_type)

ax1.set_yscale('log')
ax1.set_xlabel('Num NPUs', fontsize=16)
ax1.set_ylabel('Average Runtime (s)', fontsize=16)
ax1.set_title('Average Runtime vs. System Scale', fontsize=16)
ax1.set_xticks(ind + bar_width * (len(model_types) - 1) / 2)
ax1.set_xticklabels(unique_scales, rotation=45)
ax1.legend(fontsize=12)
ax1.grid(True, which='both', linestyle='--', linewidth=0.2)
ax1.tick_params(axis="y", labelsize=12)

# Plot peak memory bar chart
ax2 = plt.subplot(2, 1, 2)  # Change to (2, 1, 2) to stack vertically
for idx, model_type in enumerate(model_types):
    bar_positions = ind + idx * bar_width
    ax2.bar(bar_positions,
            [average_memories.get((model_type, scale), 0) for scale in unique_scales],
            bar_width, label=model_type)

ax2.set_yscale('log')
ax2.set_xlabel('Num NPUs', fontsize=16)
ax2.set_ylabel('Average Peak Memory (MB)', fontsize=16)
ax2.set_title('Average Peak Memory vs. System Scale', fontsize=16)
ax2.set_xticks(ind + bar_width * (len(model_types) - 1) / 2)
ax2.set_xticklabels(unique_scales, rotation=45)
ax2.legend(fontsize=12)
ax2.yaxis.set_major_locator(MultipleLocator(100))
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.tick_params(axis="y", labelsize=12)

plt.tight_layout()
plt.savefig("scaling.pdf")
plt.show()
