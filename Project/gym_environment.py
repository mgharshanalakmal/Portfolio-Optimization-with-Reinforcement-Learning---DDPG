import config
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from gymnasium.utils import seeding


class StockPortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.stock_dimension = len(config.stocks)
        self.observation_dimensions = len(config.observation_columns)
        self.episode_length = config.EPISODE_LENGTH
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stock_dimension,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dimensions,)
        )

        self.dataframe = config.scaled_df
        self.price_columns = [
            location
            for location, column in enumerate(self.dataframe.columns)
            if column[-5:] == "Price"
        ]

        self.initial_amount = config.INITIAL_AMOUNT
        self.terminal = False
        self.step_counter = 0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.action_memory = [[1 / self.stock_dimension] * self.stock_dimension]
        self.data = self.get_random_dataset()
        self.state = self.data.values[self.step_counter, :].flatten()

    # ..................................................................................................................

    def reset(self):
        self.initial_amount = config.INITIAL_AMOUNT
        self.terminal = False
        self.step_counter = 0
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.action_memory = [[1 / self.stock_dimension] * self.stock_dimension]
        self.data = self.get_random_dataset()
        self.state = self.data.values[0, :].flatten()

        return self.state

    # ..................................................................................................................

    def step(self, actions):
        self.terminal = self.step_counter >= len(self.data.index.unique()) - 1

        if self.terminal:
            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            if np.std(self.portfolio_return_memory) != 0:
                sharpe_ratio = (
                    (252**0.5)
                    * np.mean(self.portfolio_return_memory)
                    / np.std(self.portfolio_return_memory)
                )
                print("Sharpe: ", sharpe_ratio)

            print("=================================")
            return self.state, self.reward, self.terminal, {}

        else:
            weights = self.softmax_normalization(actions)
            self.action_memory.append(weights)
            last_day_memory = self.state

            self.step_counter += 1
            self.state = self.data.values[self.step_counter, :].flatten()

            portfolio_return = np.sum(
                (
                    (
                        self.state[self.price_columns]
                        / last_day_memory[self.price_columns]
                    )
                    - 1
                )
                * weights
            )

            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value

            self.portfolio_return_memory.append(portfolio_return)
            self.asset_memory.append(new_portfolio_value)

            self.reward = portfolio_return

            return self.state, self.reward, self.terminal, {}

    # ..................................................................................................................

    def render(self, mode="human"):
        return self.state

    # ..................................................................................................................

    def get_random_dataset(self):
        max_index = self.dataframe.shape[0]
        start_point = np.random.choice(
            np.arange(3, max_index - (self.episode_length + 2))
        )
        end_point = start_point + self.episode_length
        random_selection = self.dataframe.loc[start_point : end_point + 2]

        return random_selection

    @staticmethod
    def softmax_normalization(actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator

        return softmax_output


def main():
    env = StockPortfolioEnv()

    for i in range(520):
        sap = env.action_space.sample()
        env.step(sap)
        print(env.step_counter >= len(env.data.index.unique()) - 1)


if __name__ == "__main__":
    main()
