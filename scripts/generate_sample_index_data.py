"""Generate data/sample/sample_index_data.csv, a synthetic index file for testing.

The values are invented (fixed seed) and are not real ASPI or S&P SL20 levels;
the names are only used so the Market Regime section can be tried out.

    python scripts/generate_sample_index_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_index_data.csv"
SEED = 20260101
START = "2025-01-01"

# (trading days, mean daily return, daily standard deviation)
PHASES = {
    "ASPI": [(200, 0.0004, 0.005), (15, 0.0, 0.016), (10, -0.018, 0.015),
             (45, 0.004, 0.006), (50, 0.0004, 0.005)],
    "S&P SL20": [(180, 0.0005, 0.006), (140, -0.0012, 0.006)],
}
BASE_LEVEL = {"ASPI": 1000.0, "S&P SL20": 500.0}


def generate():
    rng = np.random.default_rng(SEED)
    frames = []
    for name, phases in PHASES.items():
        returns = np.concatenate([rng.normal(mean, sd, days) for days, mean, sd in phases])
        closes = BASE_LEVEL[name] * np.cumprod(np.concatenate([[1.0], 1 + returns]))
        frames.append(pd.DataFrame({
            "date": pd.bdate_range(START, periods=len(closes)).strftime("%Y-%m-%d"),
            "index_name": name,
            "close": np.round(closes, 2),
        }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    generate().to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")
