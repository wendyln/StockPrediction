import numpy as np
from env import TradingEnv
from model import MLP
from agent import DQNAgent

def test_agent(agent, prices, initial_cash=10000):
    env = TradingEnv(prices, initial_cash=initial_cash)

    state = env.reset()
    state = np.reshape(state, [agent.state_size])

    prices_log = []
    actions_log = []
    pnl_log = []
    stock_owned_log = []

    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        prices_log.append(env.prices[env.current_step])
        actions_log.append(action)
        pnl_log.append(env._get_portfolio_value())
        stock_owned_log.append(env.stock_owned)

        state = np.reshape(next_state, [agent.state_size])

        if done:
            break

    return prices_log, actions_log, pnl_log, stock_owned_log
