import numpy as np
import pandas as pd
from src.utils import observ_index

def rl_detect(q_table, df):
    """
    Run RL/Q-table detection on df using q_table.
    - q_table expected shape: (n_period, 2). The code uses indices 1..64.
    Returns:
      detection_vector (0/1 per time step),
      detection_time (int), final_action, final_state
    """
    packet_length = len(df)
    detection_vector = np.zeros(packet_length, dtype=int)
    t = 2
    done = False
    observ_num = 1

    # q_table shape check
    if q_table.ndim != 2 or q_table.shape[1] != 2:
        raise ValueError("q_table must be an (N x 2) numpy array.")

    while not done and t < packet_length:
        obs3 = int(df['thr-lev'].iloc[t])
        obs2 = int(df['thr-lev'].iloc[t - 1])
        obs1 = int(df['thr-lev'].iloc[t - 2])
        observ_num = observ_index(obs1, obs2, obs3)

        if observ_num < 0 or observ_num >= q_table.shape[0]:
            # If observ_num is out-of-bounds -> raise helpful message
            raise IndexError(f"observ_num={observ_num} out of bounds for q_table with shape {q_table.shape}")

        av = q_table[observ_num]
        # choose action with minimum Q (remember original used idxmin)
        action = int(np.argmin(av))

        if action == 1 and t >= 12:
            detection_vector[t:] = 1
            final_action = 1
            final_state = 2
        else:
            final_action = 0
            final_state = 0

        if t >= packet_length - 2 or final_action == 1:
            done = True
            detection_time = t
        else:
            t += 1

    return detection_vector, detection_time, final_action, final_state
