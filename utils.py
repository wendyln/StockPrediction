import matplotlib.pyplot as plt

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Trading Agent Performance")
    plt.show()


def plot_trade_path(prices, actions, pnl):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(prices, label='Stock Price', color='blue')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Mark actions on the price curve
    for i, action in enumerate(actions):
        if action == 1:  # BUY
            ax1.scatter(i, prices[i], marker='^', color='green', label='Buy' if i == 0 else "")
        elif action == 2:  # SELL
            ax1.scatter(i, prices[i], marker='v', color='red', label='Sell' if i == 0 else "")

    ax2 = ax1.twinx()
    ax2.plot(pnl, label='Portfolio Value', color='orange', alpha=0.6)
    ax2.set_ylabel('Portfolio Value', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.legend(loc='upper left')
    plt.title('Agent Trading Path & PnL')
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.show()
