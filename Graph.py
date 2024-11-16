import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from main import data
from main import env

# Load the trained model
model = PPO.load("ppo_trading_model")

# Reset the environment and get the initial observation
obs = env.reset()

# Initialize variables to store results
balance_history = []
actions = []
prices = data["Close"].values

# Run the model in the environment
for _ in range(len(prices)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    balance_history.append(env.env_method('get_balance')[0])
    actions.append(action)
    if done:
        obs = env.reset()  # Reset the environment if done

# Convert actions to numpy array for easier plotting
actions = np.array(actions)

# Chart 1, a plot showing trading actions
fig, ax = plt.subplots()
ax.plot(prices, label='Price')
buy_signals = np.where(actions == 1)[0]
sell_signals = np.where(actions == 2)[0]
ax.plot(buy_signals, prices[buy_signals], 'g^', label='Buy')
ax.plot(sell_signals, prices[sell_signals], 'rv', label='Sell')
ax.set_title('Trading Actions')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()

# Chart 2, a plot of the balance_history over time
fig2, ax2 = plt.subplots()
ax2.plot(balance_history, label='Balance')
ax2.set_title('Balance History')
ax2.set_xlabel('Time')
ax2.set_ylabel('Balance')
ax2.legend()