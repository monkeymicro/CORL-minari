# CORL (Clean Offline Reinforcement Learning) -minari

Fork from [CORL](https://github.com/tinkoff-ai/CORL), and add [minari](https://github.com/Farama-Foundation/Minari) support.

Tested on:

-Ubuntu 24.04,
-Python 3.10
-PyTorch 2.7.1+cu118
-Minari 0.5.3-
-Gymnasium 1.2.0.

For example, to run the offline reinforcement learning algorithm, execute the following script:

1. Download minari dataset:
```python
# change dataset_name
dataset_name = 'mujoco/hopper/expert-v0'
```

```
python minari_download.py
```

2. Run offline rl train script:
```
python algorithms/offline/san_n.py
```