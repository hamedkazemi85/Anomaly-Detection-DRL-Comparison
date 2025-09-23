DQN-Based Anomaly Detector for Cyber-Physical Systems

This repository implements a Deep Q-Network (DQN)–based anomaly detector for Cyber-Physical Systems (CPS). The framework is designed for resilient and timely detection of anomalies (e.g., sensor faults, cyberattacks) from streaming thresholded data.

Key Features

Detection Environment

Encodes CPS time-series data into discrete observation states.

Supports both training (with known anomaly occurrence time tau) and testing (without ground truth).

Flexible parameters for thresholds (I) and memory length (M).

Deep Reinforcement Learning Agent

Implements a DQN agent with an experience replay buffer and target network.

Learns a detection policy by approximating the Q-function with a neural network.

Optimizes trade-offs between early detection, false alarms, and late detection using a tailored reward function.

Online Detection

Once trained, the DQN policy can be deployed to perform real-time detection without requiring prior knowledge of the anomaly location.

Returns detection time, action, and system state.

Why DQN for Detection?

Traditional anomaly detection baselines include:

Threshold-based rules → simple but prone to high false alarms under noise.

Statistical change detection (CUSUM, GLR, etc.) → effective under well-modeled distributions, but less robust when system dynamics are nonlinear or under adversarial attacks.

Classical RL (tabular Q-learning, SARSA) → limited to small state spaces and cannot generalize to unseen patterns.

By contrast, DQN-based detection:

Scales to high-dimensional state spaces using neural networks.

Learns directly from raw/sequential CPS data without manual feature engineering.

Provides adaptive policies that improve with training experience.

Outperforms traditional baselines in nonlinear, noisy, or adversarial settings.

Repository Structure

environment/ → CPS detection environment definition.

agent/ → DQN agent implementation (neural network, replay buffer, optimizer).

training/ → training loop, hyperparameters, and model checkpoint saving/loading.

detection/ → online detection script using trained DQN models.

plotting/ → utilities for visualizing training curves and detection timelines.

Getting Started

Place your dataset (e.g., QtableAttack_Onlyphi15.csv) in the project root.

Train a detector:

Q, history = train_dqn(env, agent, episodes=1000)


Save and load the model:

torch.save(agent.q_network.state_dict(), "trained_dqn.pth")
agent.q_network.load_state_dict(torch.load("trained_dqn.pth"))


Run online detection:

t, action, state = run_detection(agent, df_attack)
print(f"Detection finished at t={t}, action={action}, state={state}")
