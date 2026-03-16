# Football Odds vs Outcomes (EPL)

Academic study on whether simple, transparent pre-match models can produce calibrated 1X2 probabilities that beat no-vig bookmaker prices for EPL matches. Focus is on closing odds from Football-Data.co.uk, training on historical seasons (2003/04–2024/25) and testing on 2025/26.

## Repo layout

```
football-betting/
├── src/football_betting/   # Reusable utilities (odds transforms, data helpers)
├── notebooks/
│   ├── data/raw/           # Per-season EPL CSVs (E0_XXYY.csv)
│   ├── outputs/
│   │   ├── figs/           # Per-season cumulative profit charts
│   │   └── home_bet_summary.csv
│   ├── 01_backtest_naive.ipynb
│   ├── 02_model_baseline.ipynb
│   ├── 03_model_poisson.ipynb
│   ├── 04_market_compare.ipynb
│   └── 05_backtest_value.ipynb
├── data/
│   ├── raw/                # Raw CSVs from Football-Data.co.uk
│   └── processed/          # all_raw.csv, market_epl.parquet, flat_backtest.csv
├── tests/                  # Unit tests for utilities
├── figures/                # Saved figures
├── reports/                # Write-up outputs
├── main.py
└── requirements.txt
```

## Quick start

1. Create and activate the conda environment:
   ```bash
   conda create -n football-betting python=3.11
   conda activate football-betting
   pip install -r requirements.txt
   pip install -e . --no-build-isolation
   ```

2. Raw EPL season CSVs (2003/04 through 2025/26) are in `notebooks/data/raw/` and `data/raw/`. Sourced from [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php). Prefer files with closing odds (`*C` columns).

3. Run the CLI pipeline to generate processed data:
   ```bash
   python -m football_betting.pipeline --data-dir data/raw --out-dir data/processed --stake 100
   ```
   Outputs: `data/processed/market_epl.parquet`, `data/processed/flat_backtest.csv`

4. Run tests:
   ```bash
   pytest
   ```

5. Open notebooks in order with Jupyter or VS Code (use the `football-betting` kernel):
   - `notebooks/01_backtest_naive.ipynb` — naive home/draw/away flat-stake backtest
   - `notebooks/02_model_baseline.ipynb` — calibrated logistic baseline
   - `notebooks/03_model_poisson.ipynb` — Dixon-Coles Poisson model (optional)
   - `notebooks/04_market_compare.ipynb` — FLB and calibration vs market
   - `notebooks/05_backtest_value.ipynb` — value-rule backtest with CLV diagnostic

## Data notes
- Columns used: `B365H/D/A`, `PSH/PSD/PSA`, `MaxH/D/A`, `AvgH/D/A`, and closing variants (`B365CH/D/A`, `PSCH/PSCD/PSCA`, `MaxCH/D/A`, `AvgCH/D/A`).
- Always de-vig prices before comparisons; fall back to `PS*` or `B365*` if closing/Pinnacle is missing.
- Pre-kickoff features only (no leakage). Time-based splits: train on older seasons, validate on 2024/25, test on 2025/26.

## Deliverables checklist
- Utility function `compute_no_vig_probs` with tests.
- Five notebooks: naive backtests, calibrated logistic baseline, optional Poisson/DC, market comparisons, value-rule backtest with CLV diagnostic.
- Short write-up and saved figures (calibration, FLB, EV curves, model-market scatter).
