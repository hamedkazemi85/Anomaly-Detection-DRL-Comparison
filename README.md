# DRL-Anomaly-Detection

Reinforcement Learning and Deep Reinforcement Learning detectors for anomaly (FDI/fault) detection in cyber-physical systems.

This repository contains:
- DRL detector (Deep Q-network based;Keras/TensorFlow model) — detection & optional training code
- RL detector (Q-table / SARSA) — detection using saved Q-table
- Baseline detectors (Chi-square, Euclidean) — comparison support
- Data loader and plotting utilities

## Project layout

DRL-Anomaly-Detection/
├── data/ # put your .csv datasets here
├── models/ # Train the model based on your data or put your pretrained model here e.g. modelAttackTheta.h5
├── src/ # python modules
├── main.py # entry point (detection)
├── README.md
└── requirements.txt





