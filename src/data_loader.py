import os
import pandas as pd

def load_all_csvs(data_dir="data"):
    """
    Load every .csv in data_dir into a dict mapping base-filename -> DataFrame.
    Each dataframe will have columns renamed to ['thr-lev','time'] when possible.
    """
    dfs = {}
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, header=None)
        except Exception:
            # fallback with pandas auto-detect
            df = pd.read_csv(path)

        # Normalize columns: prefer to name first two columns 'thr-lev' and 'time'
        if df.shape[1] >= 2:
            df = df.iloc[:, :2].copy()
            df.columns = ["thr-lev", "time"]
        elif df.shape[1] == 1:
            df = df.rename(columns={0: "thr-lev"})
            df["time"] = df.index
        else:
            # if empty or weird, just keep as-is
            pass

        # Ensure integer values if possible (thr-lev in your datasets are integers 1..4)
        try:
            df["thr-lev"] = df["thr-lev"].astype(int)
        except Exception:
            pass

        dfs[fname] = df.reset_index(drop=True)

    return dfs
