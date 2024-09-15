import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

file = "../Data/data.csv"

df = pd.read_csv(file).drop("Date", axis=1)

stocks = ["ACL", "BROW", "HHL", "JKH", "SOFT"]
observation_columns = df.columns

scaler = StandardScaler()
scaler.fit(df.values)

scaled_df = pd.DataFrame(scaler.transform(df.values), columns=observation_columns)


observation_columns_max = []
observation_columns_min = []

for column in observation_columns:
    observation_columns_max.append(np.max(df[column]))

for column in observation_columns:
    observation_columns_min.append(np.min(df[column]))

EPISODE_LENGTH = 500
NUM_ITERATIONS = 1000
COLLECT_STEPS_PER_ITERATION = 100
LOG_INTERVAL = 10
EVAL_INTERVAL = 4
MODEL_SAVE_FREQ = 12

REPLAY_BUFFER_MAX_LENGTH = 1000
BATCH_SIZE = 100
NUM_EVAL_EPISODES = 4

actor_fc_layers = (400, 300)
critic_obs_fc_layers = (400,)
critic_action_fc_layers = None
critic_joint_fc_layers = (300,)
ou_stddev = 0.2
ou_damping = 0.15
target_update_tau = 0.05
target_update_period = 5
dqda_clipping = None
gamma = 0.05
reward_scale_factor = 1.0
gradient_clipping = None

actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
debug_summaries = False
summarize_grads_and_vars = False


if __name__ == "__main__":
    data = df.values
    print(data.shape)
