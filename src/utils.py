# src/utils.py
import pandas as pd

def extend_dataframe(df, desired_time):
    """
    Extend dataframe in time if shorter than desired_time, keeping last thr-lev value.
    Input df must have columns ['thr-lev','time'] (or first two columns will be treated as such).
    """
    df = df.copy()
    # If headerless CSV produced integer column names, normalize:
    if df.shape[1] >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["thr-lev", "time"]

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    if "time" not in df.columns or "thr-lev" not in df.columns:
        raise ValueError("DataFrame must contain 'thr-lev' and 'time' columns (or first two columns will be used).")

    df = df.sort_values("time").reset_index(drop=True)
    last_time = int(df["time"].iloc[-1])

    if desired_time <= last_time:
        return df.copy()

    last_thr_lev = df["thr-lev"].iloc[-1]
    new_times = list(range(last_time + 1, desired_time + 1))
    new_df = pd.DataFrame({"thr-lev": [last_thr_lev] * len(new_times), "time": new_times})

    df_extended = pd.concat([df, new_df], axis=0, ignore_index=True)
    return df_extended
