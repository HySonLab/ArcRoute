# Hierarchical Directed Capacitated Arc Routing Problem (HDCARP)

## Table of Contents

- [Overview](#overview)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Problem Instances](#generate-problem-instances)
  - [Run Heuristic Algorithms](run-heuristic-algorithms)
  - [Run Exact Method](#train-reinforcement-learning-model)
  - [Run RL method](#evaluate-performance)
  - [RL Training](#rl-training)
- [Results](#results)
- [Weight and Data](#weight_and_data)
- [License](#license)
- [Contact](#contact)

## Overview

This project provides a solution to the **Hierarchical Directed Capacitated Arc Routing Problem (HDCARP)**.

The project implements multiple approaches:

- **Exact methods** for solving smaller instances.
- **Meta Heuristic algorithms** (ea, aco, ils).
- **Hybrid algorithm combining Reinforcement Learning (RL) and heuristics**.

## Key Components

The project is organized into several directories:

```
hdcarp/
├── baseline/
│   ├── aco.py                 # aco algorithm
│   ├── ea.py                  # ea algorithm
│   ├── ils.py                 # ea algorithm
│   ├── rl_hyb.py              # HRDA algorithm
│   ├── lp.py                  # exact method
│   ├── meta.py                # implemented code of Meta Heuristic algorithms
├── common/
├── env/
│   ├── env.py                # Environment setup for the routing problem
│   ├── generator.py          # Problem instance generator
├── policy/
│   ├── context.py            # Contextual features for the RL model
│   ├── encoder.py            # Encoding components
│   ├── decoder.py            # Decoding components
│   ├── init.py               # Model initialization functions
│   ├── policy.py             # Policy network for RL
├── rl/
│   ├── critic.py
│   ├── ppo.py                # Proximal Policy Optimization (PPO) algorithm
│   ├── policy.py             # Policy network for RL
│   ├── trainer.py
├── .gitignore                # Git ignore file
├── requirements.txt          # Python dependencies
├── train.py                  # Main script to start training the models
├── README.md                 # Project documentation
├── LICENSE                   # License information
```


## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Generate Problem Instances
```python
    python3 data/gen.py
```
### Run Meta Heuristic Algorithms
```python
    python3 baseline/ils.py --data_path "data/instances/30/61_20.npz"
    python3 baseline/ea.py --data_path "data/instances/30/61_20.npz"
    python3 baseline/aco.py --data_path "data/instances/30/61_20.npz"
```

### Run Exact Method
```python
    python3 baseline/lp.py
```
### Run RL Method
```python
    python3 baseline/rl_infer.py \
    --checkpoint_path "best.ckpt" \
    --data_path "data/30/61_20.npz"
```

### RL Training
```python
    python3 train.py \
    --seed 6868 \
    --max_epoch 1000 \
    --batch_size 4096 \
    --mini_batch_size 512 \
    --train_data_size 100000 \
    --val_data_size 10000 \
    --embed_dim 128 \
    --num_encoder_layers 12 \
    --num_heads 8 \
    --num_loc 20 \
    --num_arc 20 \
    --variant P \
    --checkpoint_dir /home/project/checkpoints/cl123 \
    --accelerator gpu \
    --devices 1
```

## Results

## Weight_and_Data
[Weight of HDRA](https://drive.google.com/drive/folders/16R3gudbrDpMo9sZn-iLxHVXkXTHUCVQi)

[Data](https://drive.google.com/drive/folders/1JAgkaH1TnPz7zls_oCbsjJP2nGpkTRPm)

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Please cite our work!

```bibtex
@misc{nguyen2025hybridisingreinforcementlearningheuristics,
      title={Hybridising Reinforcement Learning and Heuristics for Hierarchical Directed Arc Routing Problems}, 
      author={Van Quang Nguyen and Quoc Chuong Nguyen and Thu Huong Dang and Truong-Son Hy},
      year={2025},
      eprint={2501.00852},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.00852}, 
}
```

## Contact
For questions or collaboration inquiries, please reach out to Truong-Son Hy at thy@uab.edu
