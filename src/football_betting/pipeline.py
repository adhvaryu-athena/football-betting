"""CLI pipeline to process EPL odds data and run naive backtests without notebooks.

Usage:
    python -m football_betting.pipeline --data-dir data/raw --out-dir data/processed

Steps:
1. Load all CSVs under data_dir.
2. Select best-available odds with closing preference.
3. Compute no-vig probabilities.
4. Run flat-stake Home/Draw/Away backtests (stake=100 by default).
5. Persist processed parquet and a summary CSV of backtests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from football_betting.odds import compute_no_vig_probs

ODDS_PRIORITY: List[Tuple[str, str, str]] = [
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("MaxCH", "MaxCD", "MaxCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("MaxH", "MaxD", "MaxA"),
    ("AvgH", "AvgD", "AvgA"),
]

RESULT_COL = "FTR"
PICK_MAP = {
    "H": ("odds_home", "pH"),
    "D": ("odds_draw", "pD"),
    "A": ("odds_away", "pA"),
}


def load_epl_data(data_dir: Path) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    date_cols = [c for c in ["Date", "DateTime", "Kickoff"] if c in data.columns]
    data["match_date"] = pd.to_datetime(data[date_cols[0]], errors="coerce") if date_cols else pd.NaT

    def infer_season(file_name: str) -> str:
        stem = Path(file_name).stem
        for token in stem.replace("-", "_").split("_"):
            if token and token[0].isdigit() and len(token) >= 4:
                return token
        return "unknown"

    data["season"] = data["source_file"].apply(infer_season)
    return data


def select_market_odds(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Tuple[float, float, float, Optional[str]]] = []
    for _, row in df.iterrows():
        selected: Tuple[float, float, float, Optional[str]] = (np.nan, np.nan, np.nan, None)
        for h_col, d_col, a_col in ODDS_PRIORITY:
            if (
                h_col in row
                and d_col in row
                and a_col in row
                and pd.notna(row[h_col])
                and pd.notna(row[d_col])
                and pd.notna(row[a_col])
            ):
                selected = (row[h_col], row[d_col], row[a_col], h_col)
                break
        records.append(selected)

    out = pd.DataFrame(records, columns=["odds_home", "odds_draw", "odds_away", "odds_source"])
    return pd.concat([df.reset_index(drop=True), out], axis=1)


def add_no_vig_probs(df: pd.DataFrame) -> pd.DataFrame:
    probs = df.apply(
        lambda r: compute_no_vig_probs(r["odds_home"], r["odds_draw"], r["odds_away"])
        if pd.notna(r["odds_home"]) and pd.notna(r["odds_draw"]) and pd.notna(r["odds_away"])
        else {"pH": np.nan, "pD": np.nan, "pA": np.nan},
        axis=1,
        result_type="expand",
    )
    df[["pH", "pD", "pA"]] = probs
    return df


def evaluate_flat_strategy(df: pd.DataFrame, pick: str, stake: float = 100.0) -> Dict[str, float]:
    odds_col, prob_col = PICK_MAP[pick]
    subset = df.dropna(subset=[odds_col, prob_col, RESULT_COL])
    if subset.empty:
        return {"pick": pick, "n_bets": 0, "pnl": np.nan, "roi": np.nan, "ev": np.nan}

    returns = np.where(subset[RESULT_COL] == pick, subset[odds_col] - 1, -1) * stake
    pnl = returns.sum()
    n_bets = len(subset)
    roi = pnl / (n_bets * stake) if n_bets else np.nan
    ev = (subset[prob_col] * (subset[odds_col] - 1) - (1 - subset[prob_col])).mean() * stake
    return {"pick": pick, "n_bets": n_bets, "pnl": pnl, "roi": roi, "ev": ev}


def summarize_flat_strategies(df: pd.DataFrame, stake: float = 100.0) -> pd.DataFrame:
    rows = [evaluate_flat_strategy(df, pick, stake) for pick in ("H", "D", "A")]
    return pd.DataFrame(rows).sort_values("pick")


def run_pipeline(data_dir: Path, out_dir: Path, stake: float = 100.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_epl_data(data_dir)
    market = add_no_vig_probs(select_market_odds(data))
    flat = summarize_flat_strategies(market, stake=stake)

    out_dir.mkdir(parents=True, exist_ok=True)
    market_path = out_dir / "market_epl.parquet"
    flat_path = out_dir / "flat_backtest.csv"
    market.to_parquet(market_path, index=False)
    flat.to_csv(flat_path, index=False)
    return market, flat


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process EPL odds and run naive backtests.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing EPL CSV files.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Directory to save outputs.")
    parser.add_argument("--stake", type=float, default=100.0, help="Flat stake size for backtests.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    market, flat = run_pipeline(args.data_dir, args.out_dir, stake=args.stake)
    print("Processed rows:", len(market))
    print("Backtest results:")
    print(flat.to_string(index=False))
    print(f"Saved processed market to {args.out_dir / 'market_epl.parquet'}")
    print(f"Saved flat backtest summary to {args.out_dir / 'flat_backtest.csv'}")


if __name__ == "__main__":
    main()
