import matplotlib.pyplot as plt
import numpy as np

dense_lb = {
    "32768": {
        "wall": 313344918,
        "comm": 307340040,
        "compute": 14105756
    },
    "16384": {
        "wall": 312823838,
        "comm": 306818960,
        "compute": 14105756
    },
    "1024": {
        "wall": 193457918,
        "comm": 187453040,
        "compute": 14105756
    },
    "256": {
        "wall": 188810750,
        "comm": 182805872,
        "compute": 14105756
    },
    "2048": {
        "wall": 194510894,
        "comm": 188506016,
        "compute": 14105756
    },
    "4096": {
        "wall": 195515198,
        "comm": 189510320,
        "compute": 14105756
    },
    "512": {
        "wall": 166768070,
        "comm": 160763192,
        "compute": 14105756
    },
    "8192": {
        "wall": 196969694,
        "comm": 190964816,
        "compute": 14105756
    },
}

palm_lb = {
    "1024": {
        "wall": 1009413254,
        "comm": 985903736,
        "compute": 58450160
    },
    "256": {
        "wall": 1206440506,
        "comm": 1077096632,
        "compute": 233534272
    },
    "2048": {
        "wall": 983440610,
        "comm": 971015616,
        "compute": 29917518
    },
    "4096": {
        "wall": 971056140,
        "comm": 964116440,
        "compute": 15975198
    },
    "512": {
        "wall": 1028206894,
        "comm": 966443248,
        "compute": 116811532
    },
    "8192": {
        "wall": 966258447,
        "comm": 961728904,
        "compute": 10752106
    },
    "16384": {
        "wall": 966696985,
        "comm": 962752016,
        "compute": 8307104
    },
}

datasets = {"Dense": dense_lb, "Palm": palm_lb}

plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 1.8})
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for ax, (name, data) in zip(axes, datasets.items()):
    labels = sorted([int(k) for k in data.keys()])
    wall = np.array([data[str(x)]["wall"] / 1e6 for x in labels])
    compute = np.array([data[str(x)]["compute"] / 1e6 for x in labels])
    comm = np.array([data[str(x)]["comm"] / 1e6 for x in labels])

    overlap = (compute + comm) - wall
    compute_only = compute - overlap
    comm_only = comm - overlap

    x = np.arange(len(labels))
    width = 0.5

    ax.bar(x, compute_only, width, label="Compute", color='orange')
    ax.bar(x, overlap, width, bottom=compute_only, label="Overlap", color='darkgray', alpha=0.7, hatch='//')
    ax.bar(x, comm_only, width, bottom=compute, label="Comm only", color='skyblue')

    ax.scatter(x, wall, marker="_", color='black', s=500, linewidth=2, label='Wall-time')

    if name == "Palm":
        ideal_curve = compute[0] * labels[0] / np.array(labels)
        ax.plot(x, ideal_curve, 'r--o', label='Ideal Scaling')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("# GPUs")
    ax.set_ylabel("Time (ms)")
    if name == "Dense":
        ax.set_title(f"a. Compute & Comms Breakdown: Data Parallel")
    elif name == "Palm":
        ax.set_title(f"b. Compute & Comms Breakdown: Tensor Parallel")
    else:
        assert False
    ax.grid(axis='y')

axes[0].legend(fontsize='small')
plt.tight_layout()
plt.savefig("combined_scaling.pdf")

