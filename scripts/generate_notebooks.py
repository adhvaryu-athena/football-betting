"""Utility script to generate project notebooks with starter cells without extra deps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path.cwd().resolve()
NB_DIR = PROJECT_ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def markdown_cell(text: str) -> Dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(code: str) -> Dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def write_notebook(name: str, cells: List[Dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=2))
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


# 01 - Naive backtest
cells_01 = [
    markdown_cell(
        """# 01 - Naive Flat-Stake Backtest

Replicates flat Home/Draw/Away strategies on EPL using closing odds (or best available) and reports ROI and EV with no-vig probabilities."""
    ),
    markdown_cell(
        """## Data inputs
- Source: Football-Data.co.uk CSVs for EPL (2016/17 onward).
- Closing odds preferred: `PS*` or `*C` columns. Fallback: Bet365 then Avg/Max, flagged in `odds_source`.
- Results column: `FTR` (H/D/A). Update `RESULT_COL` below if your files differ."""
    ),
    code_cell(
        """import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
sys.path.append(str(PROJECT_ROOT / "src"))
from football_betting.odds import compute_no_vig_probs

pd.options.display.float_format = "{:.4f}".format"""
    ),
    markdown_cell(
        """## Load EPL data
Adjust the glob or parsing logic if your filenames differ. CSVs are expected in `data/raw/`."""
    ),
    code_cell(
        """def load_epl_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError("Add Football-Data.co.uk CSVs to data/raw (e.g., E0_2025-26.csv).")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    date_cols = [c for c in ["Date", "DateTime", "Kickoff"] if c in data.columns]
    if date_cols:
        data["match_date"] = pd.to_datetime(data[date_cols[0]], errors="coerce")
    else:
        data["match_date"] = pd.NaT

    def infer_season(file_name: str) -> str:
        stem = Path(file_name).stem
        for token in stem.replace("-", "_").split("_"):
            if token and token[0].isdigit() and len(token) >= 4:
                return token
        return "unknown"

    data["season"] = data["source_file"].apply(infer_season)
    return data


raw = load_epl_data()
raw.head()"""
    ),
    markdown_cell(
        """## Build market odds with closing preference and compute no-vig probabilities
Priority: Pinnacle closing (`PS*` or `PS* C`), then Bet365 closing, Max/Avg closing, then non-closing. The chosen source is recorded in `odds_source`."""
    ),
    code_cell(
        """ODDS_PRIORITY = [
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("MaxCH", "MaxCD", "MaxCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
    ("MaxH", "MaxD", "MaxA"),
    ("AvgH", "AvgD", "AvgA"),
]


def select_market_odds(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        selected = (np.nan, np.nan, np.nan, None)
        for cols in ODDS_PRIORITY:
            h_col, d_col, a_col = cols
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


market = add_no_vig_probs(select_market_odds(raw))
market[["source_file", "odds_source", "odds_home", "odds_draw", "odds_away", "pH", "pD", "pA"]].head()"""
    ),
    markdown_cell(
        """## Backtest flat Home/Draw/Away strategies
Uses stake = 100 by default to align with EV definition. ROI is P&L divided by total stakes."""
    ),
    code_cell(
        """RESULT_COL = "FTR"

PICK_MAP = {
    "H": ("odds_home", "pH"),
    "D": ("odds_draw", "pD"),
    "A": ("odds_away", "pA"),
}


def evaluate_flat_strategy(df: pd.DataFrame, pick: str, stake: float = 1.0) -> dict:
    odds_col, prob_col = PICK_MAP[pick]
    subset = df.dropna(subset=[odds_col, prob_col, RESULT_COL])
    if subset.empty:
        return {"pick": pick, "n_bets": 0, "pnl": np.nan, "roi": np.nan, "ev": np.nan}

    returns = np.where(subset[RESULT_COL] == pick, subset[odds_col] - 1, -1) * stake
    pnl = returns.sum()
    n_bets = len(subset)
    roi = pnl / (n_bets * stake)
    ev = (subset[prob_col] * (subset[odds_col] - 1) - (1 - subset[prob_col])).mean() * stake
    return {"pick": pick, "n_bets": n_bets, "pnl": pnl, "roi": roi, "ev": ev}


def summarize_flat_strategies(df: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:
    rows = [evaluate_flat_strategy(df, pick, stake) for pick in ("H", "D", "A")]
    return pd.DataFrame(rows).sort_values("pick")


flat_results = summarize_flat_strategies(market, stake=100.0)
flat_results"""
    ),
    markdown_cell(
        """## Persist processed market dataset
Saved for downstream notebooks to avoid repeating the same parsing."""
    ),
    code_cell(
        """processed_path = PROJECT_ROOT / "data" / "processed" / "market_epl.parquet"
processed_path.parent.mkdir(parents=True, exist_ok=True)
market.to_parquet(processed_path, index=False)
processed_path"""
    ),
]

write_notebook("01_backtest_naive.ipynb", cells_01)


# 02 - Baseline model
cells_02 = [
    markdown_cell(
        """# 02 - Calibrated Logistic Baseline

One-vs-rest logistic regression with calibration. Train on older seasons, validate on 2024/25, test on 2025/26 to date."""
    ),
    code_cell(
        """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "market_epl.parquet"
sys.path.append(str(PROJECT_ROOT / "src"))

sns.set_style("whitegrid")
pd.options.display.float_format = "{:.4f}".format"""
    ),
    markdown_cell(
        """## Load processed market dataset
Run `01_backtest_naive.ipynb` first to build `data/processed/market_epl.parquet`."""
    ),
    code_cell(
        """RESULT_COL = "FTR"

from typing import Optional


def load_market_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Run 01_backtest_naive.ipynb to build data/processed/market_epl.parquet.")
    return pd.read_parquet(path)


def season_start_year(season: str) -> Optional[int]:
    if isinstance(season, str) and "-" in season:
        try:
            return int(season.split("-")[0])
        except ValueError:
            return None
    return None


data = load_market_dataset()
data["season_start"] = data.get("season", pd.Series(dtype=str)).apply(season_start_year)

data.head()"""
    ),
    markdown_cell(
        """## Feature set and splits
Add pre-match features (form, rolling goal/shots differentials, league position deltas) to `FEATURE_COLS` once engineered. Only use information available before kickoff to avoid leakage."""
    ),
    code_cell(
        """FEATURE_COLS = [
    "odds_home",
    "odds_draw",
    "odds_away",
    # TODO: add pre-match stats (rolling goal diff, shots, form, league position deltas)
]

TRAIN_START = 2016
TRAIN_END = 2023
VALID_SEASON = 2024
TEST_SEASON = 2025

model_df = data.dropna(subset=FEATURE_COLS + [RESULT_COL]).copy()
model_df = model_df[model_df["season_start"].notna()]

train_df = model_df[(model_df["season_start"] >= TRAIN_START) & (model_df["season_start"] <= TRAIN_END)]
valid_df = model_df[model_df["season_start"] == VALID_SEASON]
test_df = model_df[model_df["season_start"] == TEST_SEASON]

print({
    "train_rows": len(train_df),
    "valid_rows": len(valid_df),
    "test_rows": len(test_df),
})

X_train = train_df[FEATURE_COLS]
y_train = train_df[RESULT_COL]
X_valid = valid_df[FEATURE_COLS]
y_valid = valid_df[RESULT_COL]
X_test = test_df[FEATURE_COLS]
y_test = test_df[RESULT_COL]

if X_train.empty:
    raise ValueError("Training set is empty. Check season labels and FEATURE_COLS.")"""
    ),
    markdown_cell("## Fit one-vs-rest logistic regression with calibration"),
    code_cell(
        """base_model = LogisticRegression(
    multi_class="ovr",
    C=1.0,
    penalty="l2",
    max_iter=500,
)

calibrated_model = CalibratedClassifierCV(
    base_estimator=base_model,
    method="isotonic",
    cv=5,
)

calibrated_model.fit(X_train, y_train)"""
    ),
    markdown_cell("## Evaluate log loss, Brier score, and calibration curves"),
    code_cell(
        """def evaluate_split(name: str, X: pd.DataFrame, y: pd.Series):
    if X.empty:
        print(f"[warn] {name} split is empty.")
        return None
    probs = calibrated_model.predict_proba(X)
    classes = calibrated_model.classes_
    prob_df = pd.DataFrame(probs, columns=[f"model_p{c}" for c in classes])

    ll = log_loss(y, probs, labels=classes)
    brier_components = []
    for cls in classes:
        brier_components.append(brier_score_loss((y == cls).astype(int), prob_df[f"model_p{cls}"]))
    brier = float(np.mean(brier_components))

    print(f"{name} log loss: {ll:.4f} | Brier (macro): {brier:.4f}")
    return prob_df, classes, ll, brier


eval_valid = evaluate_split("valid", X_valid, y_valid)
eval_test = evaluate_split("test", X_test, y_test)"""
    ),
    code_cell(
        """if eval_valid:
    prob_df, classes, _, _ = eval_valid
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4), sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        true_binary = (y_valid == cls).astype(int)
        prob_pos = prob_df[f"model_p{cls}"]
        frac_pos, mean_pred = calibration_curve(true_binary, prob_pos, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label="Observed")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        ax.set_title(f"Calibration for {cls} (valid)")
        ax.set_xlabel("Predicted prob")
        ax.set_ylabel("Observed freq")
        ax.legend()
    plt.tight_layout()
    plt.show()"""
    ),
    markdown_cell(
        """## Save test predictions for downstream comparisons
Stores market and model probabilities for each outcome."""
    ),
    code_cell(
        """preds_test = test_df.reset_index(drop=True).copy()
if eval_test:
    prob_df, classes, _, _ = eval_test
    for cls in classes:
        preds_test[f"model_p{cls}"] = prob_df[f"model_p{cls}"]

preds_path = PROJECT_ROOT / "reports" / "predictions_baseline.csv"
preds_path.parent.mkdir(parents=True, exist_ok=True)
preds_test.to_csv(preds_path, index=False)
preds_path"""
    ),
]

write_notebook("02_model_baseline.ipynb", cells_02)


# 03 - Poisson/DC skeleton
cells_03 = [
    markdown_cell(
        """# 03 - Poisson / Dixon-Coles (Optional)

Skeleton notebook for fitting team attack/defence parameters and mapping score matrices to 1X2 probabilities."""
    ),
    code_cell(
        """import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import optimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "market_epl.parquet"
sys.path.append(str(PROJECT_ROOT / "src"))"""
    ),
    markdown_cell(
        """## Outline
1. Load processed match data with goals (`FTHG`, `FTAG`) and market probabilities.
2. Fit attack/defence ratings with Dixon-Coles (or plain Poisson) likelihood and ridge regularization.
3. Generate scoreline probability matrix per match.
4. Aggregate to 1X2 probabilities and compare with market.

Add the full implementation as time permits; keep all features strictly pre-match."""
    ),
    code_cell(
        """def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Run 01_backtest_naive.ipynb first to build processed data.")
    df = pd.read_parquet(path)
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Poisson model: {missing}")
    return df


def fit_dixon_coles(df: pd.DataFrame):
    \"\"\"TODO: implement DC likelihood with time-decay and ridge. This stub is a placeholder.\"\"\"
    raise NotImplementedError("Implement Dixon-Coles estimation here")


def score_matrix_to_1x2(score_matrix: np.ndarray) -> dict:
    home_win = np.tril(score_matrix, -1).sum()
    draw = np.trace(score_matrix)
    away_win = np.triu(score_matrix, 1).sum()
    return {"pH": home_win, "pD": draw, "pA": away_win}"""
    ),
]

write_notebook("03_model_poisson.ipynb", cells_03)


# 04 - Market comparison
cells_04 = [
    markdown_cell(
        """# 04 - Market Comparison & Favourite–Longshot

Compares model vs no-vig market probabilities, builds favourite–longshot curves, and inspects deltas."""
    ),
    code_cell(
        """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
preds_path = PROJECT_ROOT / "reports" / "predictions_baseline.csv"
sys.path.append(str(PROJECT_ROOT / "src"))

sns.set_style("whitegrid")
pd.options.display.float_format = "{:.4f}".format"""
    ),
    markdown_cell("## Load predictions and reshape to long format"),
    code_cell(
        """if not preds_path.exists():
    raise FileNotFoundError("Run 02_model_baseline.ipynb to generate reports/predictions_baseline.csv")

df = pd.read_csv(preds_path)

long_rows = []
for _, row in df.iterrows():
    for outcome, market_col, odds_col in [
        ("H", "pH", "odds_home"),
        ("D", "pD", "odds_draw"),
        ("A", "pA", "odds_away"),
    ]:
        model_col = f"model_p{outcome}"
        if model_col not in row or pd.isna(row[market_col]):
            continue
        long_rows.append(
            {
                "outcome": outcome,
                "market_p": row[market_col],
                "model_p": row.get(model_col, np.nan),
                "odds": row.get(odds_col, np.nan),
                "result": 1 if row.get("FTR") == outcome else 0,
            }
        )

long_df = pd.DataFrame(long_rows)
long_df.head()"""
    ),
    markdown_cell(
        "## Favourite–longshot bias curve (market)\nGroups by market-probability deciles and compares observed win rates."
    ),
    code_cell(
        """if long_df.empty:
    raise ValueError("No data available to compute FLB. Check earlier notebooks.")

long_df = long_df.dropna(subset=["market_p"])
long_df["decile"] = pd.qcut(long_df["market_p"], 10, labels=False, duplicates="drop")
flb = long_df.groupby("decile").agg(
    market_p_mean=("market_p", "mean"),
    observed_win=("result", "mean"),
    count=("result", "size"),
)

plt.figure(figsize=(6, 4))
plt.plot(flb["market_p_mean"], flb["observed_win"], marker="o", label="Observed")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
plt.xlabel("Market implied probability (decile mean)")
plt.ylabel("Observed win rate")
plt.title("Favourite–Longshot Bias (market)")
plt.legend()
plt.tight_layout()
plt.show()
flb"""
    ),
    markdown_cell("## Model vs market probability deltas"),
    code_cell(
        """plt.figure(figsize=(6, 4))
plt.scatter(long_df["market_p"], long_df["model_p"], alpha=0.4)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Market no-vig p")
plt.ylabel("Model p")
plt.title("Model vs Market Probabilities")
plt.tight_layout()
plt.show()

long_df["delta"] = long_df["model_p"] - long_df["market_p"]
long_df.describe()[["market_p", "model_p", "delta"]]"""
    ),
]

write_notebook("04_market_compare.ipynb", cells_04)


# 05 - Value rule backtest
cells_05 = [
    markdown_cell(
        """# 05 - Value Backtest & CLV

Backtests a simple rule: bet when `model_p - market_p >= tau`. Reports EV, stake counts, and optional CLV if opening vs closing odds are available."""
    ),
    code_cell(
        """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
preds_path = PROJECT_ROOT / "reports" / "predictions_baseline.csv"
sys.path.append(str(PROJECT_ROOT / "src"))"""
    ),
    markdown_cell("## Load predictions"),
    code_cell(
        """if not preds_path.exists():
    raise FileNotFoundError("Run 02_model_baseline.ipynb to generate reports/predictions_baseline.csv")

df = pd.read_csv(preds_path)

required_cols = {"pH", "pD", "pA", "odds_home", "odds_draw", "odds_away", "FTR"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in predictions: {missing}")

df.head()"""
    ),
    markdown_cell("## Run threshold grid search"),
    code_cell(
        """TAUS = [round(x, 2) for x in np.arange(0.02, 0.12, 0.02)]
PICK_MAP = {
    "H": ("odds_home", "pH"),
    "D": ("odds_draw", "pD"),
    "A": ("odds_away", "pA"),
}


def value_backtest(df: pd.DataFrame, taus=TAUS, stake: float = 1.0) -> pd.DataFrame:
    rows = []
    for tau in taus:
        for outcome, (odds_col, market_col) in PICK_MAP.items():
            model_col = f"model_p{outcome}"
            if model_col not in df.columns:
                continue
            mask = (df[model_col] - df[market_col]) >= tau
            bets = df[mask].dropna(subset=[odds_col, market_col, model_col, "FTR"])
            if bets.empty:
                rows.append({"tau": tau, "outcome": outcome, "n_bets": 0, "pnl": 0.0, "roi": np.nan, "ev": np.nan})
                continue
            returns = np.where(bets["FTR"] == outcome, bets[odds_col] - 1, -1) * stake
            pnl = returns.sum()
            n_bets = len(bets)
            roi = pnl / (n_bets * stake)
            ev = (bets[model_col] * (bets[odds_col] - 1) - (1 - bets[model_col])).mean() * stake
            rows.append({"tau": tau, "outcome": outcome, "n_bets": n_bets, "pnl": pnl, "roi": roi, "ev": ev})
    return pd.DataFrame(rows)


value_results = value_backtest(df, TAUS, stake=100.0)
value_results"""
    ),
    markdown_cell("## Plot EV and ROI vs threshold"),
    code_cell(
        """if not value_results.empty:
    pivot_ev = value_results.pivot(index="tau", columns="outcome", values="ev")
    pivot_roi = value_results.pivot(index="tau", columns="outcome", values="roi")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pivot_ev.plot(ax=axes[0], marker="o")
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_title("EV vs tau")
    axes[0].set_ylabel("EV (per 100 stake)")

    pivot_roi.plot(ax=axes[1], marker="o")
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set_title("ROI vs tau")
    axes[1].set_ylabel("ROI")

    plt.tight_layout()
    plt.show()"""
    ),
    markdown_cell(
        """## Optional: Closing Line Value (CLV)
If both opening and closing odds columns exist (e.g., `PSH` vs `PSCH`), compute CLV as the difference in implied probabilities between your entry and the closing price."""
    ),
    code_cell(
        """def compute_clv(row, outcome: str) -> float:
    open_col = {"H": "PSH", "D": "PSD", "A": "PSA"}[outcome]
    close_col = {"H": "PSCH", "D": "PSCD", "A": "PSCA"}[outcome]
    if open_col not in row or close_col not in row or pd.isna(row[open_col]) or pd.isna(row[close_col]):
        return np.nan
    entry_prob = 1.0 / row[open_col]
    close_prob = 1.0 / row[close_col]
    return close_prob - entry_prob


# Example usage once columns are present:
# df[f"clv_{outcome}"] = df.apply(lambda r: compute_clv(r, outcome), axis=1)"""
    ),
]

write_notebook("05_backtest_value.ipynb", cells_05)

print("Notebook generation complete.")
