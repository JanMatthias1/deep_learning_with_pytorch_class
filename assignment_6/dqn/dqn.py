"""
Deep Q-Network (DQN) for CartPole-v1.

Implements the core DQN algorithm from Mnih et al. (2015) with:
- Experience replay buffer
- Separate target network with periodic hard updates
- Epsilon-greedy exploration with exponential decay
- W&B logging for training and evaluation metrics

"""

import argparse
import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
import wandb


# ============================================================================
# Data structures
# ============================================================================
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "info")
)


@dataclass
class EpsilonSchedule:
    """Epsilon-greedy decay parameters."""

    start: float
    end: float
    decay: float


@dataclass
class HyperParams:
    """Groups all DQN hyperparameters to keep agent attributes minimal."""

    lr: float
    gamma: float
    batch_size: int
    buffer_size: int
    eps: EpsilonSchedule
    target_update: int
    action_dim: int = 2

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "HyperParams":
        """Construct from parsed CLI arguments."""
        return cls(
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            eps=EpsilonSchedule(args.eps_start, args.eps_end, args.eps_decay),
            target_update=args.target_update,
        )


# ============================================================================
# Part 2a: Neural Network Architecture
# ============================================================================
class QNetwork(nn.Module):
    """
    Multi-layer perceptron Q-network.

    Architecture: 3-layer MLP with ReLU activations.
      Input (4) -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(2)

    Design decisions:
      - 128 hidden units: sufficient capacity for CartPole's low-dimensional
        state space (4 features). Larger networks offer no benefit here and
        risk overfitting.
      - ReLU activation: standard for value networks; avoids vanishing
        gradient issues.
      - No output activation: Q-values are unbounded real numbers.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize the Q-network.

        Args:
            state_dim: Dimension of the observation space.
            action_dim: Number of discrete actions.
            hidden_dim: Number of units in each hidden layer.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state -> Q-values for each action.

        Args:
            state: Batch of states, shape (batch_size, state_dim).

        Returns:
            Q-values for each action, shape (batch_size, action_dim).
        """
        return self.network(state)

    def copy_weights_from(self, source: "QNetwork"):
        """Hard-copy weights from another QNetwork (for target updates)."""
        self.load_state_dict(source.state_dict())


# ============================================================================
# Part 2c: Replay Buffer
# ============================================================================
class ReplayBuffer:
    """
    Fixed-size experience replay buffer.

    Stores Transition namedtuples in a circular buffer. Sampling uniformly
    at random breaks temporal correlations in training data, which
    stabilizes learning (Mnih et al., 2015, Section 4).
    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        """
        Store a transition in the buffer.

        Args:
            transition: A Transition namedtuple.
        """
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list:
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of Transition namedtuples.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


# ============================================================================
# Part 2b + Part 3: DQN Agent
# ============================================================================
class DQNAgent:
    """
    DQN Agent with Q-network, target network, and epsilon-greedy policy.

    """

    def __init__(self, hp: HyperParams, device: torch.device):
        """
        Initialize the DQN agent.

        Args:
            hp: Grouped hyperparameters.
            device: Torch device (cpu or cuda).
        """
        self.device = device
        self.hp = hp
        self.steps_done = 0
        state_dim = 4  # CartPole observation space

        # Q-Network (online) and Target Network
        self.q_network = QNetwork(state_dim, hp.action_dim).to(device)
        self.target_network = QNetwork(state_dim, hp.action_dim).to(device)
        self.target_network.copy_weights_from(self.q_network)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=hp.lr)
        self.replay_buffer = ReplayBuffer(hp.buffer_size)

    def get_epsilon(self) -> float:
        """Return the current epsilon value."""
        return self.hp.eps.end + (
            self.hp.eps.start - self.hp.eps.end
        ) * math.exp(-1.0 * self.steps_done / self.hp.eps.decay)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.

        During training, epsilon decays exponentially from eps_start to
        eps_end over eps_decay steps. During evaluation, greedy=True
        bypasses exploration entirely.

        Args:
            state: Current observation from the environment.
            greedy: If True, always select the greedy action.

        Returns:
            Selected action (0 or 1 for CartPole).
        """
        if not greedy:
            epsilon = self.get_epsilon()
            self.steps_done += 1
            if random.random() < epsilon:
                return random.randrange(self.hp.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()

    def store_transition(self, transition: Transition):
        """
        Store a transition in the replay buffer.

        Args:
            transition: A Transition namedtuple.
        """
        self.replay_buffer.push(transition)

    def update(self) -> float:
        """
        Sample a mini-batch and perform one gradient step on the DQN loss.

        DQN Loss (Mnih et al., 2015):
          L = (r + gamma * max_a' Q_target(s', a') - Q_online(s, a))^2

        For terminal states, the target is simply r (no future reward).

        Returns:
            The scalar loss value for logging.
        """
        if len(self.replay_buffer) < self.hp.batch_size:
            return 0.0

        transitions = self.replay_buffer.sample(self.hp.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Q_online(s, a) for the actions actually taken
        current_q = self.q_network(states).gather(1, actions).squeeze(1)

        # r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_max = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.hp.gamma * next_q_max * (1.0 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Hard update: copy Q-network weights to target network."""
        self.target_network.copy_weights_from(self.q_network)


# ============================================================================
# Part 5: Evaluation
# ============================================================================
def evaluate(agent: DQNAgent, num_episodes: int = 10) -> float:
    """
    Evaluate the agent using a pure greedy policy (no exploration).

    Args:
        agent: Trained DQN agent.
        num_episodes: Number of evaluation episodes.

    Returns:
        Average reward across evaluation episodes.
    """
    env = gym.make("CartPole-v1")
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state, greedy=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards)


# ============================================================================
# Part 3: Episode collection
# ============================================================================
def run_episode(agent: DQNAgent, env: gym.Env, warmup_done: bool):
    """
    Run one episode: collect transitions and optionally train.

    Args:
        agent: DQN agent.
        env: Gymnasium environment.
        warmup_done: If True, perform gradient updates each step.

    Returns:
        Tuple of (episode_reward, average_loss, steps).
    """
    state, _ = env.reset()
    episode_reward = 0.0
    total_loss = 0.0
    num_updates = 0
    steps = 0
    terminated = False

    while not terminated:
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        terminated = done or truncated

        agent.store_transition(
            Transition(state, action, reward, next_state, terminated, info)
        )

        if warmup_done:
            total_loss += agent.update()
            num_updates += 1

        state = next_state
        episode_reward += reward
        steps += 1

    return episode_reward, total_loss / max(num_updates, 1), steps


# ============================================================================
# Part 4: Logging and training
# ============================================================================
@dataclass
class EpisodeStats:
    """Bundles per-episode metrics to avoid passing many loose arguments."""

    episode: int
    reward: float
    avg100: float
    avg_loss: float
    epsilon: float
    total_steps: int
    buffer_size: int

    def to_log_dict(self) -> dict:
        """Build the W&B log dictionary."""
        return {
            "episode": self.episode,
            "episode_reward": self.reward,
            "avg_reward_100": self.avg100,
            "avg_loss": self.avg_loss,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "buffer_size": self.buffer_size,
        }

    def print_train(self):
        """Print a formatted training log line."""
        print(
            f"[Ep {self.episode:4d}] "
            f"Reward: {self.reward:6.1f} | "
            f"Avg100: {self.avg100:6.1f} | "
            f"Loss: {self.avg_loss:.4f} | "
            f"Eps: {self.epsilon:.3f}"
        )

    def print_eval(self, eval_reward: float):
        """Print a formatted evaluation log line."""
        print(
            f"[Ep {self.episode:4d}] "
            f"Reward: {self.reward:6.1f} | "
            f"Avg100: {self.avg100:6.1f} | "
            f"Eval: {eval_reward:6.1f} | "
            f"Loss: {self.avg_loss:.4f} | "
            f"Eps: {self.epsilon:.3f} | "
            f"Steps: {self.total_steps}"
        )


def print_training_header(args: argparse.Namespace):
    """Print a summary of training configuration."""
    print("=" * 60)
    print("DQN Training on CartPole-v1")
    print(f"Episodes: {args.episodes}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Buffer: {args.buffer_size}, Gamma: {args.gamma}")
    print(f"Eps: {args.eps_start} -> {args.eps_end} (decay={args.eps_decay})")
    print(f"Target update every {args.target_update} episodes")
    print(f"Warmup steps: {args.warmup}")
    print("=" * 60)


def run_eval_step(agent, args, stats, log_dict, best_eval_reward):
    """Run evaluation if due; return updated best reward."""
    if stats.episode % args.eval_interval != 0:
        return best_eval_reward

    eval_reward = evaluate(agent, num_episodes=args.eval_episodes)
    log_dict["eval_avg_reward"] = eval_reward

    if eval_reward > best_eval_reward:
        best_eval_reward = eval_reward
        torch.save(agent.q_network.state_dict(), "dqn/best_model.pt")
        log_dict["best_eval_reward"] = best_eval_reward

    stats.print_eval(eval_reward)
    return best_eval_reward


def init_training(args):
    """Set up device, W&B, environment, and agent. Returns a tuple."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"DQN_lr{args.lr}_bs{args.batch_size}_gamma{args.gamma}",
    )

    env = gym.make("CartPole-v1")
    hp = HyperParams.from_args(args)
    agent = DQNAgent(hp, device)
    print_training_header(args)
    return env, agent, hp


def train(args: argparse.Namespace):
    """
    Main training loop.

    Workflow:
      1. Initialize environment, agent, and W&B logging.
      2. For each episode:
         a. Reset environment, collect transitions via epsilon-greedy.
         b. After warmup, sample mini-batches and update Q-network.
         c. Every target_update episodes, hard-update target network.
         d. Every eval_interval episodes, evaluate with greedy policy.
      3. Log all metrics to W&B.

    Args:
        args: Parsed command-line arguments.
    """
    env, agent, hp = init_training(args)

    episode_rewards = []
    best_eval_reward = 0.0
    total_steps = 0

    for episode in range(1, args.episodes + 1):
        result = run_episode(agent, env, total_steps >= args.warmup)
        total_steps += result[2]
        episode_rewards.append(result[0])

        stats = EpisodeStats(
            episode=episode, reward=result[0],
            avg100=np.mean(episode_rewards[-100:]),
            avg_loss=result[1], epsilon=agent.get_epsilon(),
            total_steps=total_steps, buffer_size=len(agent.replay_buffer),
        )

        if episode % hp.target_update == 0:
            agent.update_target_network()

        log_dict = stats.to_log_dict()
        best_eval_reward = run_eval_step(
            agent, args, stats, log_dict, best_eval_reward
        )

        if episode % 10 == 0 and episode % args.eval_interval != 0:
            stats.print_train()

        wandb.log(log_dict)

    final_eval = evaluate(agent, num_episodes=20)
    print("=" * 60)
    print(f"Final evaluation (20 episodes): {final_eval:.1f}")
    print(f"Best evaluation reward: {best_eval_reward:.1f}")
    print("=" * 60)

    wandb.log({"final_eval_reward": final_eval})
    wandb.finish()
    env.close()


# ============================================================================
# CLI argument parser
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DQN hyperparameters."""
    parser = argparse.ArgumentParser(description="DQN for CartPole-v1")

    parser.add_argument(
        "--episodes", type=int, default=600, help="Number of training episodes"
    )
    parser.add_argument(
        "--warmup", type=int, default=1000,
        help="Steps before training begins (fill replay buffer)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for training"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000, help="Replay buffer capacity"
    )
    parser.add_argument(
        "--eps-start", type=float, default=1.0, help="Initial epsilon"
    )
    parser.add_argument(
        "--eps-end", type=float, default=0.01, help="Final epsilon"
    )
    parser.add_argument(
        "--eps-decay", type=float, default=500,
        help="Epsilon exponential decay rate (in steps)",
    )
    parser.add_argument(
        "--target-update", type=int, default=3,
        help="Update target network every N episodes",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=20,
        help="Evaluate every N episodes",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="DQN-CartPole",
        help="W&B project name",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
