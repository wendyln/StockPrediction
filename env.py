import numpy as np

class TradingEnv:
    def __init__(self, prices, initial_cash=10000):
        self.prices = prices
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.current_step = 0
        self.stock_owned = 0
        self.cash_in_hand = self.initial_cash
        return self._get_state()

    def _get_state(self):
        return np.array([[self.stock_owned, self.prices[self.current_step], self.cash_in_hand]])

    def _get_portfolio_value(self):
        return self.stock_owned * self.prices[self.current_step] + self.cash_in_hand

    def step(self, action):
        prev_value = self._get_portfolio_value()
        price = self.prices[self.current_step]

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1 and self.cash_in_hand >= price:
            self.stock_owned += 1
            self.cash_in_hand -= price
        elif action == 2 and self.stock_owned > 0:
            self.stock_owned -= 1
            self.cash_in_hand += price
        # if action in [1, 2]:  # BUY or SELL
        #     reward -= 0.001 * current_price  # transaction cost

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        new_value = self._get_portfolio_value()
        # reward = new_value - prev_value 
        # reward = (new_value - prev_value)*10 ##
        reward =(new_value - prev_value) * 10
        return self._get_state(), reward, done
