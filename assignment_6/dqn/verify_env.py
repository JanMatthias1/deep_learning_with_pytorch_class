"""
Environment Setup Verification.

Demonstrates creating the CartPole-v1 environment, stepping through it,
and inspecting the observation/action spaces.
"""

import gymnasium as gym


def verify_environment():
    """Create CartPole-v1, step through one episode, and print observations."""
    env = gym.make("CartPole-v1")

    # Inspect spaces
    print(f"Observation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Low:  {env.observation_space.low}")
    print(f"  High: {env.observation_space.high}")
    print(f"Action space: {env.action_space}")
    print(f"  Number of actions: {env.action_space.n}")
    print()

    # Reset and step through a few timesteps
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    print("  [cart_pos, cart_vel, pole_angle, pole_angular_vel]")
    print()

    total_reward = 0.0
    for step in range(1, 21):
        action = env.action_space.sample()  # random action
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        print(
            f"Step {step:3d} | Action: {action} | "
            f"Obs: [{obs[0]:+.3f}, {obs[1]:+.3f}, "
            f"{obs[2]:+.3f}, {obs[3]:+.3f}] | "
            f"Reward: {reward} | Done: {done}"
        )
        if done or truncated:
            print(f"\nEpisode ended at step {step}.")
            break

    print(f"\nTotal reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    verify_environment()
