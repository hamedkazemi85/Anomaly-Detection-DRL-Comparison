import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

NUM_INPUTS = 4

def load_tf_model(model_path):
    """
    Load a Keras model (.h5 or .keras).
    Raises informative error if not found.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DRL model not found at: {model_path}")
    model = load_model(model_path)
    return model

def drl_detect(model, df, threshold=0.98, min_detect_time=18):
    """
    Run the DRL detector (same logic as your notebook's q_learning_detection_reward_action).
    Returns:
      - action_probs_over_time: numpy array of length len(df) with probability of 'detect' (action 1)
      - actions_taken: list of 0/1 decisions per time step (stops appended)
      - observations_history: list of 4-element observations
      - rewards: list of rewards encountered
      - action_values_list: raw action probability vectors from the model
    """
    packet_length = len(df)
    action_probs_over_time = np.zeros(packet_length)
    actions_taken = []
    observations_history = []
    rewards = []
    action_values_list = []

    t = 2
    done = False

    while not done and t < packet_length:
        # build 4-element observation: [t-3, t-2, t-1, t]
        # stay consistent with original code ordering
        obs0 = int(df['thr-lev'].iloc[t])          # current
        obs1 = int(df['thr-lev'].iloc[t - 1])
        obs2 = int(df['thr-lev'].iloc[t - 2])
        obs3 = int(df['thr-lev'].iloc[t - 3]) if t - 3 >= 0 else 1
        observations = [obs3, obs2, obs1, obs0]

        # model expects shape (1, NUM_INPUTS)
        probs = model.predict(np.array(observations).reshape(1, NUM_INPUTS), verbose=0).flatten()
        p_detect = float(probs[1])  # probability assigned to action 1
        action_val = 1 if (p_detect > threshold and t >= min_detect_time) else 0

        # keep 'p_detect' value for t..end until changed (same logic as notebook)
        action_probs_over_time[t:] = p_detect

        actions_taken.append(action_val)
        observations_history.append(observations)
        action_values_list.append(probs)

        # reward heuristic as in notebook
        if action_val == 1:
            reward = 10 if p_detect > threshold else -10
            state = 2
        else:
            reward = -1
            state = 0

        rewards.append(reward)

        if t >= packet_length - 2 or action_val == 1:
            done = True
        else:
            t += 1

    return action_probs_over_time, actions_taken, observations_history, rewards, action_values_list


def train_policy_gradient(model, env, data_frames, optimizer,
                          num_iterations=400, num_game_rounds=20, max_game_steps=1000,
                          discount_rate=0.98):
    """
    Optional training function (keeps the same algorithmic structure you used).
    Not run by default; use for (re)training if you want.
    - model: Keras model
    - env: instance of DetectionEnv_DRL
    - data_frames: list of (df, tau)
    - optimizer: tf.keras.optimizer instance
    """
    def helper_discount_rewards(rewards, discount_rate):
        discounted_rewards = np.zeros(len(rewards))
        cumulative = 0
        for step in reversed(range(len(rewards))):
            cumulative = rewards[step] + cumulative * discount_rate
            discounted_rewards[step] = cumulative
        return discounted_rewards

    def discount_and_normalize(all_rewards):
        all_disc = [helper_discount_rewards(r, discount_rate) for r in all_rewards]
        flat = np.concatenate(all_disc)
        mean = flat.mean()
        std = flat.std() + 1e-8
        return [(r - mean) / std for r in all_disc]

    for iteration in range(num_iterations):
        all_rewards = []
        all_gradients = []

        for df, tau in data_frames:
            for game in range(num_game_rounds):
                current_rewards = []
                current_gradients = []

                observations = env.reset(df)
                for step in range(max_game_steps):
                    obs = np.array(observations).reshape(1, NUM_INPUTS)
                    probs = model(obs).numpy().flatten()
                    if np.random.rand() < 0.7:
                        action = np.random.choice(2, p=probs)
                    else:
                        action = np.random.choice(2)

                    state, reward, done, t, observations, info = env.step(df, action, tau)
                    current_rewards.append(reward)

                    with tf.GradientTape() as tape:
                        logits = model(np.array(observations).reshape(1, NUM_INPUTS), training=True)
                        action_one_hot = tf.one_hot([action], 2)
                        loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=action_one_hot, logits=logits))
                    grads = tape.gradient(loss_value, model.trainable_variables)
                    current_gradients.append(grads)

                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

        # Normalize rewards and update parameters
        all_rewards = discount_and_normalize(all_rewards)
        # compute mean gradients (matching your notebook)
        mean_grads = []
        for var_idx in range(len(model.trainable_variables)):
            # build array of reward * grad for each var
            arr = []
            for game_i, rewards in enumerate(all_rewards):
                for step, reward in enumerate(rewards):
                    grad = all_gradients[game_i][step][var_idx]
                    arr.append(reward * grad)
            mean = np.mean(arr, axis=0)
            mean_grads.append(mean)
        optimizer.apply_gradients(zip(mean_grads, model.trainable_variables))
