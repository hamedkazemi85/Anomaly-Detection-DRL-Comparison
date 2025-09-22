# main.py  (root)
import os
import argparse
import importlib
import inspect

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.utils import extend_dataframe
from src.plot_results import (
    plot_fdi_attack_detection,
    plot_covert_attack_detection,
    plot_performance_comparison,
)

# --------------------------
# DEFAULT FILENAMES (edit if your filenames differ)
# --------------------------
DEFAULT_FILES = {
    "phi35": "QtableAttack_phi35.csv",
    "theta46": "QtableAttack_theta46.csv",
    "theta46_sysnoise": "QtableAttack_theta46Measnoise.csv",
    "theta46_attacknoise": "QtableAttack_theta46AttackNoise.csv",
    "theta46_sysnoiseD": "QtableAttack_theta46NoiseD.csv",
    "theta43": "QtableAttack_theta43.csv",
    "theta43_attacknoise": "QtableAttack_theta43AttackNoise.csv",
    "theta43_sysnoise": "QtableAttack_theta43Measnoise.csv",
    "act_Theta55": "QtableAttack_actTheta55.csv",
    "act_Psi66": "QtableAttack_actPsi66.csv",
    "act_Theta25_Covert": "QtableAttack_actTheta25Covert.csv",
    "act_Psi45_Covert": "QtableAttack_actPsi45Covert.csv",
    # chi / euc comparison
    "chi_theta43": "Chi_attack_theta43_time.csv",
    "euc_theta43": "Euc_attack_theta43_time.csv",
    "chi_theta43_attacknoise": "Chi_attack_theta43_timeAttackNoise.csv",
    "euc_theta43_attacknoise": "Euc_attack_theta43_timeAttackNoise.csv",
    "chi_theta43_sysnoise": "Chi_attack_theta43_timeMeasnoise.csv",
    "euc_theta43_sysnoise": "Euc_attack_theta43_timeMeasnoise.csv",
    "qtable": "action_values_Attack_theta.csv",
}

# --------------------------
# Helpers to flexibly find your detection functions inside src
# --------------------------
def find_detection_function(possible_modules, possible_names):
    """
    Search for a function with any name in possible_names inside the first importable module
    from possible_modules that contains it. Returns function or None.
    """
    for m in possible_modules:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        for name in possible_names:
            if hasattr(mod, name):
                return getattr(mod, name), f"{m}.{name}"
    return None, None

# possible module candidates and function names (keeps backward compatibility)
DRL_MODULE_CANDIDATES = [
    "src.drl_detector",
    "src.drl",
    "src.drl_detection",
    "src.drl_agent",
    "src.dqn",
    "src.drl_detector_py",
    "src"
]
DRL_FUNC_NAMES = [
    "q_learning_detection_reward_action",  # your notebook name
    "drl_detect",
    "q_learning_detection_reward_action_from_model",
    "q_learning_detection_reward_action_model_df"
]

RL_MODULE_CANDIDATES = [
    "src.rl_detector",
    "src.rl",
    "src.sarsa",
    "src.q_learning",
    "src.rl_detector_py",
    "src"
]
RL_FUNC_NAMES = [
    "q_learning_detection_RL",  # your notebook name
    "q_learning_detection_RL_from_qtable",
    "rl_detect",
    "q_learning_detection_rl",
    "q_learning_detection_RL"
]

def safe_call(fn, args_list):
    """
    Attempt to call fn with a subset of args_list that matches fn's signature.
    args_list is a list/tuple of candidate args in priority order (e.g. [model, df, qtable, tau])
    """
    sig = inspect.signature(fn)
    n = len(sig.parameters)
    try:
        return fn(*args_list[:n])
    except TypeError:
        # try to call with no args or single arg (best-effort)
        try:
            return fn()
        except Exception as e:
            raise

# --------------------------
# CSV loader helper (normalizes headerless CSVs)
# --------------------------
def load_csv_norm(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=None) if pd.read_csv(path, nrows=1, header=None).shape[1] <= 2 else pd.read_csv(path)
    # If it has >=2 cols, ensure they are named 'thr-lev','time'
    if df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["thr-lev", "time"]
    elif df.shape[1] == 1:
        df.columns = ["thr-lev"]
        df["time"] = range(len(df))
    return df

def load_qtable(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=None)
    if df.shape[1] >= 3:
        arr = df.iloc[:, 1:3].to_numpy()
    elif df.shape[1] == 2:
        arr = df.to_numpy()
    else:
        raise ValueError("Unexpected Q-table format.")
    return arr

# --------------------------
# Main
# --------------------------
def main(data_dir="data", model_path="models/modelAttackTheta.h5", qtable_path=None):
    # Resolve file paths for datasets using DEFAULT_FILES keys
    files = {k: os.path.join(data_dir, v) for k, v in DEFAULT_FILES.items()}
    if qtable_path:
        files['qtable'] = qtable_path

    # Load csv datasets that are needed
    def try_load(name):
        p = files.get(name)
        if p and os.path.exists(p):
            try:
                return load_csv_norm(p)
            except Exception as e:
                print(f"Failed to read {p}: {e}")
                return None
        else:
            print(f"File for '{name}' not found at expected path: {p}")
            return None

    df_attack_phi35 = try_load("phi35")
    df_attack_theta46 = try_load("theta46")
    df_attack_act_Theta55 = try_load("act_Theta55")
    df_attack_act_Psi66 = try_load("act_Psi66")

    # sysnoise / attacknoise / theta43 variants
    df_attack_theta46sysnoise = try_load("theta46_sysnoise")
    df_attack_theta46attacknoise = try_load("theta46_attacknoise")
    df_attack_theta46sysnoiseD = try_load("theta46_sysnoiseD")

    df_attack_theta43 = try_load("theta43")
    df_attack_theta43attacknoise = try_load("theta43_attacknoise")
    df_attack_theta43sysnoise = try_load("theta43_sysnoise")

    # Covert
    df_attack_act_Theta25_Covert = try_load("act_Theta25_Covert")
    df_attack_act_Psi45_Covert = try_load("act_Psi45_Covert")

    # chi / euc comparison csvs
    df_chi43 = try_load("chi_theta43")
    df_euc43 = try_load("euc_theta43")
    df_chi43attacknoise = try_load("chi_theta43_attacknoise")
    df_euc43attacknoise = try_load("euc_theta43_attacknoise")
    df_chi43sysnoise = try_load("chi_theta43_sysnoise")
    df_euc43sysnoise = try_load("euc_theta43_sysnoise")

    # Load model (optional)
    model = None
    if model_path and os.path.exists(model_path):
        print("Loading DRL model from:", model_path)
        try:
            model = load_model(model_path)
        except Exception as e:
            print("Warning: failed to load model via keras.load_model:", e)
            model = None
    else:
        print("No DRL model file found (skipping DRL detection).")

    # Load Q-table (optional)
    q_table = None
    qtable_file = files.get('qtable')
    if qtable_file and os.path.exists(qtable_file):
        q_table = load_qtable(qtable_file)
        print("Q-table loaded, shape:", None if q_table is None else q_table.shape)
    else:
        print("No Q-table found (skipping RL detection).")

    # Discover the DRL detection function in src (flexible)
    drl_fn, drl_loc = find_detection_function(DRL_MODULE_CANDIDATES, DRL_FUNC_NAMES)
    if drl_fn:
        print("Found DRL detection function at:", drl_loc)
    else:
        print("DRL detection function not found in candidate modules. Looking for function name 'q_learning_detection_reward_action' in global scope.")
        # Attempt direct import if the function is in src top-level
        try:
            import src as s
            if hasattr(s, "q_learning_detection_reward_action"):
                drl_fn = getattr(s, "q_learning_detection_reward_action")
                print("Found drl function as src.q_learning_detection_reward_action")
        except Exception:
            pass

    # Discover the RL detection function
    rl_fn, rl_loc = find_detection_function(RL_MODULE_CANDIDATES, RL_FUNC_NAMES)
    if rl_fn:
        print("Found RL detection function at:", rl_loc)
    else:
        print("RL detection function not found in candidate modules. Looking for 'q_learning_detection_RL' in src.")
        try:
            import src as s
            if hasattr(s, "q_learning_detection_RL"):
                rl_fn = getattr(s, "q_learning_detection_RL")
                print("Found rl function as src.q_learning_detection_RL")
        except Exception:
            pass

    # === RUN DRL detection calls (use drl_fn) ===
    # We try to call drl_fn with the correct arguments depending on signature:
    def run_drl_call(drl_fn, model, df):
        if drl_fn is None or df is None:
            return None
        import inspect
        sig = inspect.signature(drl_fn)
        n = len(sig.parameters)
        try:
            if n == 1:
                # expects (df,) - your original notebook function form
                return drl_fn(df)
            elif n == 2:
                # maybe expects (model, df)
                return drl_fn(model, df)
            elif n == 0:
                return drl_fn()
            else:
                # last resort try df only:
                return drl_fn(df)
        except Exception as e:
            # as fallback, attempt (model, df)
            try:
                return drl_fn(model, df)
            except Exception as e2:
                print("Failed to call DRL detection function:", e, e2)
                return None

    print("\nRunning DRL detection (if DRL function + model available)...")
    def safe_unpack(result):
        # drl_fn in notebook returns (action_probs_test, _, _, _, _).
        # We'll try to safely unpack the first returned element (action_probs).
        if result is None:
            return None
        if isinstance(result, tuple) or isinstance(result, list):
            if len(result) >= 1:
                return result[0]
            else:
                return result
        return result

    action_probs_test_phi35 = safe_unpack(run_drl_call(drl_fn, model, df_attack_phi35))
    action_probs_test_theta46 = safe_unpack(run_drl_call(drl_fn, model, df_attack_theta46))
    action_probs_test_act_Theta55 = safe_unpack(run_drl_call(drl_fn, model, df_attack_act_Theta55))
    action_probs_test_act_Psi66 = safe_unpack(run_drl_call(drl_fn, model, df_attack_act_Psi66))

    # more DRL variants
    action_probs_test_theta46_sysnoise = safe_unpack(run_drl_call(drl_fn, model, df_attack_theta46sysnoise))
    action_probs_test_theta46_attacknoise = safe_unpack(run_drl_call(drl_fn, model, df_attack_theta46attacknoise))
    action_probs_test_theta46_sysnoiseD = safe_unpack(run_drl_call(drl_fn, model, df_attack_theta46sysnoiseD))

    action_probs_test_theta43 = safe_unpack(run_drl_call(drl_fn, model, df_attack_theta43))
    action_probs_test_theta43_attacknoise = safe_unpack(run_drl_call(drl_fn, model, df_attack_theta43attacknoise))
    action_probs_test_theta43_sysnoise = safe_unpack(run_drl_call(drl_fn, model, df_attack_theta43sysnoise))

    action_probs_test_act_Theta25_Covert = safe_unpack(run_drl_call(drl_fn, model, df_attack_act_Theta25_Covert))
    action_probs_test_act_Psi45_Covert = safe_unpack(run_drl_call(drl_fn, model, df_attack_act_Psi45_Covert))

    # === RUN RL detection (q-table based) ===
    print("\nRunning RL detection (if RL function + q-table available)...")
    detection_vector_theta46RL = None
    detection_vector_theta46_attacknoiseRL = None
    detection_vector_theta46_sysnoiseRL = None
    detection_vector_theta46_sysnoiseDRL = None
    detection_vector_theta43RL = None
    detection_vector_theta43_attacknoiseRL = None
    detection_vector_theta43_sysnoiseRL = None

    if rl_fn is not None and q_table is not None:
        try:
            res = rl_fn(q_table, df_attack_theta46)
            # rl_fn usually returns (detection_vector, t, action, state)
            if isinstance(res, (tuple, list)):
                detection_vector_theta46RL = res[0]
            else:
                detection_vector_theta46RL = res
        except Exception:
            # maybe signature is (df,)
            try:
                res = rl_fn(df_attack_theta46)
                detection_vector_theta46RL = res[0] if isinstance(res, (tuple, list)) else res
            except Exception as e:
                print("RL detection call failed for theta46:", e)

        # theta43 variants
        try:
            res = rl_fn(q_table, df_attack_theta43)
            detection_vector_theta43RL = res[0] if isinstance(res, (tuple, list)) else res
        except Exception:
            try:
                res = rl_fn(df_attack_theta43)
                detection_vector_theta43RL = res[0] if isinstance(res, (tuple, list)) else res
            except Exception as e:
                print("RL detection call failed for theta43:", e)
    else:
        print("Skipping RL detection (function or q-table missing).")

    # === Prepare chi/euc dfs (extend if necessary) ===
    if df_chi43 is not None and df_euc43 is not None:
        df_chi43_ext = extend_dataframe(df_chi43, 90)
        df_euc43_ext = extend_dataframe(df_euc43, 90)
    else:
        df_chi43_ext = df_chi43
        df_euc43_ext = df_euc43

    # For sysnoise/attacknoise variants, if files missing, try to reuse base
    df_chi43sysnoise_ext = extend_dataframe(df_chi43attacknoise, 90) if (df_chi43attacknoise is not None) else (df_chi43_ext)
    df_euc43sysnoise_ext = extend_dataframe(df_euc43attacknoise, 90) if (df_euc43attacknoise is not None) else (df_euc43_ext)
    df_chi43attacknoise_ext = extend_dataframe(df_chi43attacknoise, 90) if (df_chi43attacknoise is not None) else (df_chi43_ext)
    df_euc43attacknoise_ext = extend_dataframe(df_euc43attacknoise, 90) if (df_euc43attacknoise is not None) else (df_euc43_ext)

    # === NOW PLOT the exact figures you requested (use defaults if missing) ===
    print("\nPlotting FDIAttackDetection (this will save FDIAttackDetection.pdf)...")
    plot_fdi_attack_detection(
        action_probs_test_phi35 if action_probs_test_phi35 is not None else (np.zeros(100)),
        action_probs_test_theta46 if action_probs_test_theta46 is not None else (np.zeros(100)),
        action_probs_test_act_Theta55 if action_probs_test_act_Theta55 is not None else (np.zeros(100)),
        action_probs_test_act_Psi66 if action_probs_test_act_Psi66 is not None else (np.zeros(100)),
        save_path="FDIAttackDetection.pdf"
    )

    print("Plotting CovertAttackDetection (this will save CovertAttackDetection.pdf)...")
    plot_covert_attack_detection(
        action_probs_test_act_Theta25_Covert if action_probs_test_act_Theta25_Covert is not None else (np.zeros(100)),
        action_probs_test_act_Psi45_Covert if action_probs_test_act_Psi45_Covert is not None else (np.zeros(100)),
        save_path="CovertAttackDetection.pdf"
    )

    print("Plotting DetectionComparison (this will save DetectionComparison.pdf)...")
    plot_performance_comparison(
        action_probs_test_theta43 if action_probs_test_theta43 is not None else (np.zeros(100)),
        detection_vector_theta43RL if detection_vector_theta43RL is not None else (np.zeros(100)),
        df_chi43_ext if df_chi43_ext is not None else pd.DataFrame({"thr-lev":[0],"time":[0]}),
        df_euc43_ext if df_euc43_ext is not None else pd.DataFrame({"thr-lev":[0],"time":[0]}),
        action_probs_test_theta43_sysnoise if action_probs_test_theta43_sysnoise is not None else (np.zeros(100)),
        detection_vector_theta43_sysnoiseRL if detection_vector_theta43_sysnoiseRL is not None else (np.zeros(100)),
        df_chi43sysnoise_ext,
        df_euc43sysnoise_ext,
        action_probs_test_theta43_attacknoise if action_probs_test_theta43_attacknoise is not None else (np.zeros(100)),
        detection_vector_theta43_attacknoiseRL if detection_vector_theta43_attacknoiseRL is not None else (np.zeros(100)),
        df_chi43attacknoise_ext,
        df_euc43attacknoise_ext,
        save_path="DetectionComparison.pdf"
    )

    print("\nAll plots saved (FDIAttackDetection.pdf, CovertAttackDetection.pdf, DetectionComparison.pdf).")
    print("If any plot contains placeholders (all zeros), it means an expected dataset or function was not found; adjust file names or ensure modules are in src/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="folder with your CSVs")
    parser.add_argument("--model-path", default="models/modelAttackTheta.h5", help="pretrained keras model")
    parser.add_argument("--qtable-path", default=None, help="optional q-table CSV path (default in data/action_values_Attack_theta.csv)")
    args = parser.parse_args()

    main(data_dir=args.data_dir, model_path=args.model_path, qtable_path=args.qtable_path)
