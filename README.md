# Football Odds vs Outcomes (EPL)

Academic study on whether simple, transparent pre-match models can produce calibrated 1X2 probabilities that beat no-vig bookmaker prices for EPL matches. Focus is on closing odds from Football-Data.co.uk, training on historical seasons and testing on 2025/26 (to 13-Dec-2025).

## Repo layout
- `src/football_betting/`: reusable utilities (odds transforms, data helpers).
- `notebooks/`: analysis notebooks (backtests, baseline model, market comparison, value rules).
- `tests/`: unit tests for utilities.
- `figures/` and `reports/`: output targets.
- `data/`: place raw CSVs from Football-Data.co.uk (not tracked).

## Quick start
1) Create a virtual env and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   # Editable install to make `football_betting` importable
   pip install -e . --no-build-isolation
   # If offline, you can skip the editable install and instead set:
   # export PYTHONPATH=src
   ```
3) Run the simple CLI pipeline (no notebooks):
   ```bash
   python -m football_betting.pipeline --data-dir data/raw --out-dir data/processed --stake 100
   ```
   This loads all CSVs in `data/raw`, selects best-available odds (closing preferred), computes no-vig probabilities, runs flat Home/Draw/Away backtests, and saves:
   - `data/processed/market_epl.parquet`
   - `data/processed/flat_backtest.csv`
2) Download Football-Data.co.uk CSVs for EPL seasons (2016/17 through 2025/26) into `data/raw/`. Prefer files with closing odds (`*C` columns). Keep a consistent naming scheme (e.g., `E0_2025-26.csv`).
3) Run tests:
   ```bash
   pytest
   ```
4) Open notebooks with Jupyter or VS Code and execute in order:
   - `notebooks/01_backtest_naive.ipynb`
   - `notebooks/02_model_baseline.ipynb`
   - `notebooks/03_model_poisson.ipynb` (optional)
   - `notebooks/04_market_compare.ipynb`
   - `notebooks/05_backtest_value.ipynb`

## Data notes
- Columns used: `B365H/D/A`, `PSH/PSD/PSA`, `MaxH/D/A`, `AvgH/D/A`, and closing variants (`B365CH/D/A`, `PSCH/PSCD/PSCA`, `MaxCH/D/A`, `AvgCH/D/A`).
- Always de-vig prices before comparisons; fall back to `PS*` or `B365*` if closing or Pinnacle is missing and flag the fallback in analysis.
- Use pre-kickoff features only (no leakage); time-based splits: train on older seasons, validate on 2024/25, test on 2025/26 to 13-Dec-2025.

## Deliverables checklist
- Utility function `compute_no_vig_probs` with tests.
- Five notebooks covering: naive backtests, calibrated logistic baseline, optional Poisson/DC, market comparisons (FLB, calibration), and value-rule backtest with CLV diagnostic.
- Short write-up and saved figures (calibration, FLB, EV curves, model-market scatter).
