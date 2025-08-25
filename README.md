
---

# Symbolic Tensor Graph (STG) Generator

**Author:** Changhai Man @ [SynergyLab](https://synergy.ece.gatech.edu/)    
**Contact:** cman8@gatech.edu

## Overview

The Symbolic Tensor Graph is a generator for [Chakra Execution Trace (ET)](https://github.com/mlcommons/chakra) files. This tool is designed to generate synthetic workload traces for use in parallel strategy exploration without gathering data from a real system or implementing actual workload codes. It supports various parallelization strategies like Data Parallelism (DP), Tensor Parallelism (TP), Pipeline Parallelism (PP) and Sequence Parallelism (SP).

### Key Features
- Generate synthetic transformer workloads in Chakra ET format.
- Supports multiple parallelism strategies (DP, TP, PP, SP).
- Support customized model dimensions for Transformer Models (batch, seq, dmodel, dff, n_head)

## Installation

To set up the environment and install the required dependencies, follow these steps:

```bash
# Clone the repository
git clone git@github.com:astra-sim/symbolic_tensor_graph.git

# Navigate to the project directory
cd symbolic_tensor_graph

# Install dependencies via conda
conda create -n <env_name>
conda activate <env_name>
conda install numpy sympy python-graphviz protobuf pandas -c conda-forge
```

## Usage

To generate symbolic workloads, use the following command:

```bash
python main.py –h
```

This will show all available options and their descriptions. Example of running the generator:

```bash
python main.py --output_dir generated/ \
               --output_name workload.%d.et \
               --comm_group_file comm_group.json \
               --dp 2 --tp 2 --pp 2 \
               --weight_sharded 0 
```

### Example Output:

```bash
$ ls generated/
comm_group.json  workload.0.et  workload.1.et  workload.2.et  workload.3.et
```

## Parameters

    | Argument               | Type    | Required | Default    | Description                                                                 |
    |------------------------|---------|----------|------------|-----------------------------------------------------------------------------|
    | --output_dir           | str     | Yes      | -          | Directory to store output traces.                                           |
    | --output_name          | str     | Yes      | -          | Name of the output traces.                                                  |
    | --dp                   | int     | No       | 1          | Data parallel degree.                                                       |
    | --tp                   | int     | No       | 1          | Tensor parallel degree.                                                     |
    | --sp                   | int     | No       | 1          | Sequence parallel degree.                                                   |
    | --ep                   | int     | No       | 1          | Expert parallel degree.                                                     |
    | --pp                   | int     | No       | 1          | Pipeline parallel degree.                                                   |
    | --weight_sharded       | bool    | No       | False      | Whether weights are sharded.                                                |
    | --activation_recompute | bool    | No       | False      | Whether to recompute activations.                                           |
    | --tpsp                 | bool    | No       | True       | Use tensor parallel + sequence parallel or tensor parallel only.            |
    | --dvocal               | int     | No       | 32000      | Vocabulary size.                                                            |
    | --dmodel               | int     | No       | 8192       | Model dimension.                                                            |
    | --dff                  | int     | No       | 28672      | Feed-forward dimension.                                                     |
    | --batch                | int     | No       | 64         | Batch size.                                                                 |
    | --micro_batch          | int     | No       | -1         | Micro-batch size. Default is -1 (same as batch size).                       |
    | --seq                  | int     | No       | 1024       | Sequence length.                                                            |
    | --head                 | int     | No       | 64         | Number of attention heads.                                                  |
    | --kvhead               | int     | No       | 8          | Number of key-value heads.                                                  |
    | --num_stacks           | int     | No       | 80         | Number of transformer layers.                                               |
    | --experts              | int     | No       | 8          | Number of experts in MoE.                                                   |
    | --kexperts             | int     | No       | 2          | Number of selected experts per token.                                       |
    | --chakra_schema_version| str     | No       | "v0.0.4"   | Chakra schema version.                                                      |
    | --model_type           | str     | No       | "llama"    | Type of model to assemble ("llama", "gpt", "moe", or "debug").              |
    | --mixed_precision      | bool    | No       | False      | Whether to use mixed precision.                                             |
    | --print_gpu_vram       | bool    | No       | False      | Whether to print per-GPU VRAM footprint.                                    |

\*: We do not specify number of total NPUs, which will be infered from the parallel degree as: ```num_NPUs=DP*TP*PP*SP```
## Example Commands

- **Generate with DP=8, TP=4, PP=4, no FSDP:**
  ```bash
  python main.py --output_dir generated/ --output_name workload_1.%d.et --comm_group_file comm_group_1.json --dp 8 --tp 4 --pp 4 --sp 1 --weight_sharded 0 --chakra_schema_version v0.0.4
  ```

- **Generate with DP=64, TP=1, PP=1, FSDP:**
  ```bash
  python main.py --output_dir generated/ --output_name workload_2.%d.et --comm_group_file comm_group_2.json --dp 64 --tp 1 --pp 1 --sp 1 --weight_sharded 1 --chakra_schema_version v0.0.4
  ```

- **Generate with DP=4, TP=4, PP=2, SP=2, FSDP, output in JSON format:**
  ```bash
  python main.py --output_dir generated/ --output_name workload_3.%d.json --comm_group_file comm_group_3.json --dp 4 --tp 4 --pp 2 --sp 2 --weight_sharded 1 --chakra_schema_version json
  ```


## Tool workflow
Here is a breif workflow about how stg generate traces step by step.
![alt text](./docs/images/stg_workflow.png)

## Chakra Schema Version

The schema version used determines compatibility with different tools and repositories:
- **v0.0.4**: Current latest chakra version (by Oct.6 2024).
- **v0.0.1**: Supported for lagacy, not fully tested.

## License

MIT

---
