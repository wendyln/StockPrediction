import numpy as np
from env import TradingEnv
from model import MLP
from agent import DQNAgent

def train_agent(prices, episodes=10, batch_size=32, initial_cash=10000):
    state_size = 3
    action_size = 3

    # Initialize environment and agent
    model = MLP(state_size=3, action_size=3)
    env = TradingEnv(prices, initial_cash=initial_cash)
    agent = DQNAgent(state_size, action_size, model)

    all_rewards = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [state_size])
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [state_size])
            agent.save(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"Episode {e+1}/{episodes} — Total Reward: {total_reward:.2f}")
                all_rewards.append(total_reward)
                break

            if len(agent.memory_buffer) > batch_size:
                agent.replay(batch_size)

    return agent, all_rewards
