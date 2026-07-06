# Assignment 6: Deep Q-Network from Scratch

**Author:** Jan Matthias
**Environment:** CartPole-v1 (OpenAI Gymnasium)

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to balance a pole on a cart. The agent is trained entirely from scratch using PyTorch.

CartPole-v1 provides a 4-dimensional continuous observation (cart position, cart velocity, pole angle, pole angular velocity) and 2 discrete actions (push left, push right). An episode terminates when the pole angle exceeds ±12° or the cart moves beyond ±2.4 units. The maximum reward per episode is 500 (one reward unit per surviving timestep, truncated at 500).

## Environment Setup

The setup installs a minimal stack: Python 3.10, PyTorch 2.0+ with CUDA 11.8, and Gymnasium (the maintained fork of OpenAI Gym). Gymnasium is required rather than `gym` because the latter is deprecated and the step API changed (5-tuple return: `state, reward, terminated, truncated, info`). CUDA is optional — CartPole runs comfortably on CPU, but the cluster run used a V100S GPU.

**Requirements:** Python 3.10, PyTorch 2.0+, Gymnasium, NumPy, Weights & Biases.

### Quick setup (JHPCE cluster)

```bash
bash setup_env.sh
```

This creates a conda environment at `/users/jmatthia/deep_learning/env` with all dependencies and then runs `verify_env.py` to confirm the environment steps correctly. Change the `ENV` path in `setup_env.sh` if installing elsewhere.

The setup script also runs `verify_env.py` at the end, which creates the CartPole-v1 environment, prints the observation/action spaces, and steps through up to 20 random actions — confirming that the 4-dim `Box` observation and `Discrete(2)` action space are wired correctly.

To activate manually:

```bash
conda activate /users/jmatthia/deep_learning/env
```

## Architecture

### Q-Network

A small MLP with two hidden layers maps states to Q-values:

```
Input (4) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2)
```

**Design rationale:**
- 128 hidden units provide sufficient capacity for CartPole's 4-dimensional state without overfitting. Larger networks offer no measurable benefit on this task.
- ReLU avoids vanishing gradients and is the standard choice for value networks.
- No output activation — Q-values are unbounded real numbers.

### Target Network

A second copy of the Q-network provides stable targets for the Bellman update. Its weights are hard-copied from the online Q-network every 3 episodes. This decouples the target from the online network's rapid updates, preventing the moving-target instability where the network chases its own shifting predictions.

### Replay Buffer

A fixed-size circular buffer (capacity 10,000) stores `(state, action, reward, next_state, done, info)` transitions. Uniform random sampling breaks temporal correlations in the training data, which is essential for stable convergence.

## Hyperparameters and Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 1e-4 | Conservative enough to avoid Q-value oscillation while the target network lags between hard updates. Higher rates (1e-3) destabilize the Bellman target. |
| Discount factor (γ) | 0.99 | Effective horizon of ~100 steps, long enough that the agent values sustained balance over immediate survival on a 500-step task.  |
| Batch size | 64 | Lower gradient variance than smaller batches with negligible wall-clock cost on GPU. Small enough that the 10k buffer provides many non-overlapping batches, preserving sample diversity. |
| Buffer capacity | 10,000 | Covers ~20 full-length episodes. Large enough to decorrelate samples; small enough that stale transitions from the early near-random policy get overwritten as the policy improves. |
| Epsilon start | 1.0 | Full exploration before the Q-network has learned anything. |
| Epsilon end | 0.01 | Near-greedy after decay, retaining minimal exploration to avoid premature convergence to a suboptimal greedy policy. |
| Epsilon decay | 500 steps | By ~1500 environment steps, epsilon ≈ 0.05, so exploitation dominates early and gradient updates refine a real policy rather than random actions. |
| Target update | Every 3 episodes | At ~100–500 steps/episode, this gives 300–1500 steps between hard updates — enough lag for target stability without the online network drifting too far ahead. |
| Warmup | 1,000 steps | Fills the buffer with experience before gradient updates begin, so the first mini-batches aren't dominated by a handful of highly correlated early transitions. |
| Gradient clip | max_norm=1.0 | Prevents occasional gradient explosions when the target network lags significantly behind the online network. |

## Training

### Run on JHPCE (SLURM)

```bash
sbatch run_training.sh
```

### Run locally

```bash
python dqn/dqn.py --episodes 1500 --wandb-project "DQN-CartPole"
```

### All CLI arguments

```
--episodes       Number of training episodes (default: 600)
--lr             Learning rate (default: 1e-4)
--batch-size     Mini-batch size (default: 64)
--buffer-size    Replay buffer capacity (default: 10000)
--gamma          Discount factor (default: 0.99)
--eps-start      Initial epsilon (default: 1.0)
--eps-end        Final epsilon (default: 0.01)
--eps-decay      Epsilon decay rate in steps (default: 500)
--target-update  Target network update interval in episodes (default: 3)
--warmup         Warmup steps before training (default: 1000)
--eval-interval  Evaluate every N episodes (default: 20)
--eval-episodes  Number of evaluation episodes (default: 10)
--wandb-project  W&B project name (default: DQN-CartPole)
```

## Training Procedure

1. **Warmup phase:** The agent acts via epsilon-greedy for 1,000 steps to populate the replay buffer before any gradient updates.
2. **Training phase:** Each timestep, the agent selects an action via epsilon-greedy, stores the transition, and performs one gradient update by sampling a mini-batch from the buffer.
3. **DQN loss:** `L = (r + γ · max_a' Q_target(s', a') − Q_online(s, a))²`. For terminal states, the target reduces to `r`.
4. **Target updates:** Every 3 episodes, the target network weights are hard-copied from the online Q-network.
5. **Epsilon schedule:** Exponential decay from 1.0 to 0.01 with a decay constant of 500 steps.

## Evaluation

Every 20 training episodes, the agent is evaluated for 10 episodes using a pure greedy policy (ε = 0). The best model (by evaluation reward) is saved to `dqn/best_model.pt`. A final 20-episode evaluation runs at the end of training.

## Results

The trained agent achieves the maximum possible reward on CartPole-v1.

- **Final evaluation (20 episodes, greedy):** 500.0
- **Best evaluation reward:** 500.0
- **First episode reaching avg-100 ≥ 400:** ~episode 1080
- **Stable convergence to 500.0 avg-100:** ~episode 1200

This exceeds the assignment target of 400.

### Training dynamics

Training shows four phases: a stall for the first ~200 episodes while the buffer fills with short random episodes, a breakthrough around episode 220–350 where reward climbs from ~20 to ~100+, an intermediate plateau with loss spikes (~500–750) as the target network lags a rapidly improving online network, and final convergence to reward 500 by episode ~1200. Brief single-episode regressions appear occasionally (e.g. ep 1270, 1460) but recover within a few episodes — characteristic vanilla-DQN instability without the fixes from later variants (Double DQN, prioritized replay).

## Monitoring

All metrics are logged to Weights & Biases:

- `episode_reward`: reward per training episode
- `avg_reward_100`: rolling average over the last 100 episodes
- `avg_loss`: mean DQN loss per episode
- `epsilon`: current exploration rate
- `eval_avg_reward`: greedy evaluation reward (every 20 episodes)
- `best_eval_reward`: running best evaluation reward

## File Structure

```
assignment_6/
├── setup_env.sh          # Environment setup script
├── run_training.sh       # SLURM job submission script
├── dqn/
│   ├── dqn.py            # Complete DQN implementation
│   ├── verify_env.py     # Environment verification script
│   ├── requirements.txt  # Python dependencies
│   └── best_model.pt     # Saved best model (created during training)
├── logs/                 # SLURM output logs (created during training)
└── README.md             # This file
```

## Code Quality

The implementation is object-oriented (`QNetwork`, `ReplayBuffer`, `DQNAgent`, `EpisodeStats`, `HyperParams`) and linted to 10/10 with pylint.