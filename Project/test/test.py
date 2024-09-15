import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class PortfolioEnv(gym.Env):
    def __init__(
        self,
        df,
        stock_dim,
        initial_amount,
        reward_scaling,
        state_space,
        action_space,
        lookback=252,
        day=0,
    ):
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space

        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.state = self.df.values
        self.terminal = False

        self.portfolio_value = self.initial_amount

        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.action_memory = [[1 / self.stock_dim] * self.stock_dim]
