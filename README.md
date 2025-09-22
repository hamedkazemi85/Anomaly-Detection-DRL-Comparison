# DRL-Anomaly-Detection

Reinforcement Learning and Deep Reinforcement Learning detectors for anomaly (FDI/fault) detection in cyber-physical systems.

This repository contains:
- DRL detector (Keras/TensorFlow model) — detection & optional training code
- RL detector (Q-table / SARSA) — detection using saved Q-table
- Baseline detectors (Chi-square, Euclidean) — comparison support
- Data loader and plotting utilities

## Project layout

DRL-Anomaly-Detection/
├── data/ # put your .csv datasets here
├── models/ # put your pretrained model here e.g. modelAttackTheta.h5
├── src/ # python modules
├── main.py # entry point (detection)
├── README.md
└── requirements.txt


## Quick start (detection-only)
1. Place your CSV files inside `data/`. Examples from the paper/notebook:
   - `QtableAttack_phi35.csv`
   - `QtableAttack_theta46.csv`
   - `QtableAttack_actTheta55.csv`
   - `QtableAttack_actPsi66.csv`
   - `action_values_Attack_theta.csv` (Q-table)
2. Put your pretrained model file under `models/` (e.g. `models/modelAttackTheta.h5`).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run detection:
 python main.py --data-dir data --model-path models/modelAttackTheta.h5 --qtable-path data/action_values_Attack_theta.csv


to run and only save results
  !python main.py

to run, show, and save results

from main import main

# Call main() directly
main(data_dir="data", model_path="models/modelAttackTheta.h5", qtable_path=None)





