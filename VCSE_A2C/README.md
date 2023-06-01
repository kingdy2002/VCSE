# Maximum Value-Conditional State Entropy Exploration for Sample-Efficient Deep Reinforcement Learning

Thank you for reviewing our source code! We provide the instruction for reproducting our main results below:

## Installation
Please install below packages in a following order:

### rl-starter-files

```
cd rl-starter-files/rl-starter-files
pip3 install -r requirements.txt
```

### gym_minigrid

```
git clone https://github.com/Farama-Foundation/Minigrid.git
cd Minigrid
git checkout 116fa65bf9584149f9a23c2b61c95fd84c25e467
pip3 install -e .
```

### torch-ac

```
cd torch-ac
pip3 install -e .
```

### pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.2 -c pytorch -c nvidia
```

## Reproducing our main results

### A2C
```
cd rl-starter-files/rl-stater-files
source run_original.sh
```

### A2C+SE
```
cd rl-starter-files/rl-stater-files
source run_sent.sh
```

### A2C+VCSE
```
cd rl-starter-files/rl-stater-files
source run_vcse.sh
```
