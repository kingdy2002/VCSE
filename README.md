# Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration (VCSE)

This repository contains the implementation of various algorithms used in our paper, which can be found at [ArXiv](https://arxiv.org/abs/2305.19476).

## Algorithms

The repository is organized into several folders, each containing the implementation of a specific algorithm:

- `VCSE_A2C/`: Contains the implementation of the A2C algorithm.
- `VCSE_DrQv2/`: Contains the implementation of the DrQv2 algorithm.
- `VCSE_MWM/`: Contains the implementation of the MWM algorithm.
- `VCSE_SAC/`: Contains the implementation of the SAC algorithm.

If you wand to refer this research, please cite:
"""
@misc{kim2023accelerating,
      title={Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration}, 
      author={Dongyoung Kim and Jinwoo Shin and Pieter Abbeel and Younggyo Seo},
      year={2023},
      eprint={2305.19476},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

## VCSE + A2C
Our code is built on top of the RE3 + A2C implementation [RE3](https://github.com/younggyoseo/RE3). That trainning code can be found in rl-starter-files directory. Which is fork from [rl-starter-files](https://github.com/lcswillems/rl-starter-files) and A2C implementation is fork from [torch-ac](https://github.com/lcswillems/torch-ac)

Refer to the individual README files in VCSE_A2C for details of installation and instructions.

## VCSE + DrQv2
Our code is built on top of the [drqv2](https://github.com/facebookresearch/drqv2) repository.

Refer to the individual README files in VCSE_DrQv2 for details of installation and instructions.

## VCSE + MWM
Our code is built on top of the [MWM](https://github.com/younggyoseo/MWM) repository.

Refer to the individual README files in VCSE_MWM for details of installation and instructions.

## VCSE + SAC
Our code is built on top of the [pytorch_sac](https://github.com/denisyarats/pytorch_sac) repository.

Refer to the individual README files in VCSE_SAC for details of installation and instructions.


Please refer to the individual README files in each folder for more detailed instructions.
