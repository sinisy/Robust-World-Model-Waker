
# Robust World Models Training without Rewards

![](https://github.com/sinisy/Robust-World-Model-Waker/blob/main/terrain_walker.gif)

This is the official revised source code to reproduce experiments from the ICLR 2024 paper [Reward-Free Curricula for Training Robust World Models](https://openreview.net/forum?id=eCGpNGDeNu). It implements the WAKER: Weighted Acquisition of Knowledge across Environments for Robustness algorithm, as well as the baselines presented in the paper.

## Setup

Install dependencies via pip:

```
cd Robust-World-Model-Waker
pip3 install -r requirements.txt
```

To use the SafetyGym environments, you must also install MuJoCo 210:
```
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C ~/.mujoco
```

## Running the code

To reproduce the experiments presented in the paper, run the code using the following format:
```
python3 Robust-World-Model-Waker/train.py --logdir ~/log_dir --configs domain alg expl_policy 
```

The variables domain, alg, and expl_policy should be replaced with:
- The domain of choice (Either TerrainWalker, TerrainHopper, CleanUp, or CarCleanUp).
- The algorithm to use (Either WAKER-M, WAKER-R, DR, HardestEnvOracle, ReweightingOracle, or GradualExpansion).
- The exploration policy to implement (Plan2Explore or RandomExploration).

An example run statement is illustrated below:
```
python3 Robust-World-Model-Waker/train.py --logdir ~/log_dir --configs TerrainWalker WAKER-M Plan2Explore
```

## Citing Robust-World-Model-Waker

```
@article{rigter2024waker,
  title={Reward-Free Curricula for Training Robust World Models},
  author={Rigter, Marc and Jiang, Minqi and Posner, Ingmar},
  journal={International Conference on Learning Representations},
  year={2024}
}
```