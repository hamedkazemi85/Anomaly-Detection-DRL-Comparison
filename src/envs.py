import numpy as np
from gym import Env
from gym.spaces import Discrete
from src.utils import observ_index

class DetectionEnv_DRL(Env):
    """
    DRL-style environment (observations = [pre2, pre1, prev, current] as in your notebook).
    step(df, action, tau) -> (state, reward, done, t, next_observation, info)
    next_observation is a 4-element list of thr-lev values.
    """
    def __init__(self):
        self.action_space = Discrete(2)  # {0: continue, 1: stop}
        self.state = 0
        self.Detection_length = 200
        self.t = 0

    def step(self, df, action, tau):
        self.t += 1
        observation_memory = np.ones(4, dtype=int)
        if action == 1:
            self.state = 2
            reward = -10 if self.t <= tau else 1
            if self.t >= 3:
                # gather a 4-element next observation [t-3, t-2, t-1, t]
                next_obs = [
                    int(df['thr-lev'].iloc[self.t - 3]),
                    int(df['thr-lev'].iloc[self.t - 2]),
                    int(df['thr-lev'].iloc[self.t - 1]),
                    int(df['thr-lev'].iloc[self.t])
                ]
            else:
                next_obs = [1, 1, 1, 1]
        else:
            # continue
            if self.t >= tau:
                reward = -5
                self.state = 2
            else:
                reward = 1
                self.state = 0
            if self.t >= 2:
                # observation memory uses t, t-1, t-2
                observation_memory[0] = int(df['thr-lev'].iloc[self.t])
                observation_memory[1] = int(df['thr-lev'].iloc[self.t - 1])
                observation_memory[2] = int(df['thr-lev'].iloc[self.t - 2])
                # observation_memory[3] stays 1 (unused)
            else:
                observation_memory[:] = 1
            next_obs = [
                int(observation_memory[2]),
                int(observation_memory[1]),
                int(observation_memory[0]),
                int(observation_memory[0])  # duplicate of current for fixed-size
            ]

        done = self.t > self.Detection_length - 2
        return self.state, reward, done, self.t, next_obs, {}

    def reset(self, df):
        self.state = 0
        self.Detection_length = len(df) - 1
        self.t = 0
        return [1, 1, 1, 1]


class DetectionEnv_RL(Env):
    """
    RL/Q-table environment: uses 3-values sliding window to compute observation index (1..64)
    step(df, action, tau, observ_num) -> (state, reward, done, t, next_observ_num, info)
    """
    def __init__(self):
        self.action_space = Discrete(2)
        self.state = 0
        self.Detection_length = 200
        self.t = 0

    def step(self, df, action, tau, observ_num):
        self.t += 1
        if action == 1:
            self.state = 2
            reward = 1 if self.t <= tau else 0
            if self.t >= 12:
                # next observation built from t-1, t, t+1 (keeps same logic as your notebook)
                next_observ_num = observ_index(
                    int(df['thr-lev'].iloc[self.t - 1]),
                    int(df['thr-lev'].iloc[self.t]),
                    int(df['thr-lev'].iloc[self.t + 1])
                )
            else:
                next_observ_num = 1
        else:
            if self.t >= tau:
                reward = 0.02
                self.state = 1
            else:
                reward = 0
                self.state = 0
            if self.t >= 12:
                next_observ_num = observ_index(
                    int(df['thr-lev'].iloc[self.t - 2]),
                    int(df['thr-lev'].iloc[self.t - 1]),
                    int(df['thr-lev'].iloc[self.t])
                )
            else:
                next_observ_num = 1

        done = self.t > self.Detection_length - 2
        return self.state, reward, done, self.t, next_observ_num, {}

    def reset(self, df):
        self.state = 0
        self.Detection_length = len(df) - 1
        self.t = 0
        return 1
