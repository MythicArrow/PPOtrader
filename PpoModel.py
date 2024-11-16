import gymnasium as gym
import gym_anytrading
from stable_baselines3 import PPO

# Define the policy and value function networks
policy_kwargs = dict(
    net_arch=[dict(pi=[64, 64], vf=[64, 64])]
)

# Create the environment
env = gym.make("stocks-v0")

# Create the PPO model
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1
)

# Train the PPO model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_trading_model")