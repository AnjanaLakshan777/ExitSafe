"""Generate tests/fixtures/synthetic_backtest_data.csv: SYNTHETIC stocks for backtests.

TEST DATA ONLY. The four symbols (ALPHA, BRAVO, CHARLIE, DELTA) are invented,
and the prices, volumes and turnover come from a fixed random seed. The
results of backtests on this file are not evidence about any real market or
strategy.

Each stock's daily return is  beta * market return + mean + noise,  where the
market return is the SYNTHETIC "ASPI" series in data/sample/sample_index_data.csv
(same 321 weekdays from 2025-01-01), so the stocks have differing returns,
volatility and correlation, and the sample index can serve as the benchmark.

  ALPHA    beta 1.0, moderate noise, liquid, reported turnover
  BRAVO    beta 0.6, low noise (defensive), liquid, reported turnover
  CHARLIE  beta 1.4, high noise, higher drift, mid liquidity, NO turnover column
           values (the importer estimates traded value as close x volume)
  DELTA    beta 0.3, high noise, illiquid with zero-volume days (price unchanged)

    python scripts/generate_synthetic_backtest_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "data" / "sample" / "sample_index_data.csv"
OUTPUT = ROOT / "tests" / "fixtures" / "synthetic_backtest_data.csv"
SEED = 20260313

# symbol: (beta, mean daily return, noise sd, start price, mean volume, has turnover)
STOCKS = {
    "ALPHA": (1.0, 0.0002, 0.006, 100.0, 400_000, True),
    "BRAVO": (0.6, 0.0001, 0.003, 50.0, 600_000, True),
    "CHARLIE": (1.4, 0.0006, 0.012, 25.0, 150_000, False),
    "DELTA": (0.3, 0.0003, 0.015, 200.0, 8_000, True),
}
ZERO_VOLUME_SHARE = 0.08          # DELTA only


def generate():
    index = pd.read_csv(INDEX_FILE)
    aspi = index[index["index_name"] == "ASPI"].sort_values("date")
    dates = aspi["date"].tolist()
    market = aspi["close"].pct_change().iloc[1:].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for symbol, (beta, mean, sd, start, volume, has_turnover) in STOCKS.items():
        returns = beta * market + mean + rng.normal(0, sd, len(market))
        volumes = np.round(volume * rng.lognormal(0, 0.4, len(dates))).astype(int)
        if symbol == "DELTA":
            zero = rng.random(len(dates)) < ZERO_VOLUME_SHARE
            zero[0] = False
            volumes[zero] = 0
            returns[zero[1:]] = 0.0          # no trade: the close stays unchanged
        closes = start * np.cumprod(np.concatenate([[1.0], 1 + returns]))
        for i, day in enumerate(dates):
            close = closes[i]
            previous = closes[i - 1] if i else close
            if volumes[i] == 0:
                o = h = l = previous if i else close
                close = o
            else:
                o = previous * (1 + rng.normal(0, sd / 3))
                h = max(o, close) * (1 + abs(rng.normal(0, sd / 2)))
                l = min(o, close) * (1 - abs(rng.normal(0, sd / 2)))
            rows.append({"Symbol": symbol, "Date": day, "Open": round(o, 2), "High": round(h, 2),
                         "Low": round(l, 2), "Close": round(close, 2), "Volume": int(volumes[i]),
                         "Turnover": round(close * volumes[i], 2) if has_turnover else ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    generate().to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")
