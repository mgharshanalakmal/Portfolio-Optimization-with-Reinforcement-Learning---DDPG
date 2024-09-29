from abc import ABC

import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

import config


class PortfolioEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            (len(config.stocks),),
            np.float64,
            minimum=0,
            maximum=1,
            name="action",
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            (len(config.observation_columns),),
            dtype=np.float64,
            minimum=-np.inf,
            maximum=np.inf,
            name="observation",
        )

        self.df = config.scaled_df
        self.reset()
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self) -> ts.TimeStep:
        self._episode_ended = False
        self.index = 0

        self.max_index = self.df.shape[0]
        start_point = np.random.choice(np.arange(3, self.max_index - (config.EPISODE_LENGTH + 2)))
        end_point = start_point + config.EPISODE_LENGTH
        self.df = self.df.loc[start_point : end_point + 2]
        self.step_reward = 0
        self._state = self.get_observations(self.index)
        self._episode_ended = True if self.index == config.EPISODE_LENGTH else False

        return ts.restart(self._state)

    def get_observations(self, index):
        return self.df[index].values.flatten()

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self.index += 1
        self.step_time()

    def step_time(self):
        pass


ff = PortfolioEnv()
