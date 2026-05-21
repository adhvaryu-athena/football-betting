"""
generate_figures.py — All publication figures for the EPL betting calibration paper.

Run from the repo root:
    python generate_figures.py

Outputs:
    figures/fig1_calibration_reliability.png
    figures/fig2_overround_timeseries.png
    figures/fig3_cumulative_profit.png
    figures/fig4_team_home_profitability.png

    Also prints all key numbers for the Results section to stdout.

Data: looks for per-season EPL CSVs (E0*.csv) in:
    1. data/raw/
    2. notebooks/data/raw/
"""

import glob
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
FIG_DIR  = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

STAKE        = 100
PHASE_SPLIT  = pd.Timestamp("2022-05-22")
DATA_START   = pd.Timestamp("2019-08-01")
ROLLING_N    = 50
BOOTSTRAP_N  = 2000

# Color palette (Okabe-Ito — color-blind safe)
C_BLUE   = "#0072B2"
C_RED    = "#D55E00"
C_GREEN  = "#009E73"
C_ORANGE = "#E69F00"
C_GRAY   = "#999999"
C_PURPLE = "#CC79A7"

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "figure.dpi":         150,   # screen preview
    "savefig.dpi":        300,   # publication output
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})


# ── DATA LOADING ─────────────────────────────────────────────────────────────
def load_all_seasons() -> pd.DataFrame:
    search_dirs = ["data/raw", "notebooks/data/raw"]
    files = []
    for d in search_dirs:
        found = sorted(glob.glob(f"{d}/E0*.csv"))
        if found:
            files = found
            print(f"  Found {len(files)} CSV files in {d}/")
            break
    if not files:
        raise FileNotFoundError(
            "No season CSVs found. Check data/raw/ or notebooks/data/raw/."
        )
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  Warning: skipping {f} — {e}")
    df = pd.concat(dfs, ignore_index=True)
    # Drop duplicate column names that arise from concatenating CSVs with
    # slightly different schemas (keeps first occurrence of each name)
    df = df.loc[:, ~df.columns.duplicated()]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] >= DATA_START].copy()
    print(f"  Loaded {len(df):,} matches  ({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


# ── HELPERS ───────────────────────────────────────────────────────────────────
def no_vig(h, d, a):
    """No-vig (fair) probabilities from decimal odds."""
    r = np.array([1/h, 1/d, 1/a])
    return r / r.sum()


def overround_pct(h, d, a):
    return (1/h + 1/d + 1/a - 1) * 100


def outcome_onehot(ftr):
    mapping = {"H": (1, 0, 0), "D": (0, 1, 0), "A": (0, 0, 1)}
    return mapping.get(ftr, (np.nan, np.nan, np.nan))


def brier_multi(p, y):
    """Multiclass Brier score for one match. p and y are length-3 arrays."""
    return float(np.sum((p - y) ** 2))


def log_loss_multi(p, y, eps=1e-9):
    """Multiclass log loss for one match."""
    return float(-np.sum(y * np.log(np.clip(p, eps, 1 - eps))))


def bootstrap_mean_ci(arr, n=BOOTSTRAP_N, ci=0.95):
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    lo = np.percentile(means, 100 * (1 - ci) / 2)
    hi = np.percentile(means, 100 * (1 + ci) / 2)
    return np.mean(arr), lo, hi


def reliability_curve(probs, outcomes, n_bins=10):
    """Return (mean_predicted, observed_rate, counts) for a reliability diagram."""
    edges = np.linspace(0, 1, n_bins + 1)
    mp, mo, ct = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() >= 5:
            mp.append(probs[mask].mean())
            mo.append(outcomes[mask].mean())
            ct.append(mask.sum())
    return np.array(mp), np.array(mo), np.array(ct)


# ── BUILD CORE DATAFRAME ──────────────────────────────────────────────────────
def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a row-per-match DataFrame with all computed metrics."""
    needed = ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA", "FTR"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df[["Date", "HomeTeam", "AwayTeam", "FTR"] + needed[:-1]].copy()
    for c in needed[:-1]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=needed).copy()

    # Coerce FTR to clean string (guards against mixed types after CSV concat)
    d["FTR"] = d["FTR"].astype(str).str.strip()
    d = d[d["FTR"].isin(["H", "D", "A"])].copy()

    # No-vig probs
    nv_open  = d.apply(lambda r: pd.Series(no_vig(r.B365H, r.B365D, r.B365A),
                                            index=["ph_o", "pd_o", "pa_o"]), axis=1)
    nv_close = d.apply(lambda r: pd.Series(no_vig(r.B365CH, r.B365CD, r.B365CA),
                                            index=["ph_c", "pd_c", "pa_c"]), axis=1)
    d = pd.concat([d, nv_open, nv_close], axis=1)

    # Outcomes
    oh = d["FTR"].apply(lambda x: pd.Series(outcome_onehot(x), index=["yh", "yd", "ya"]))
    d = pd.concat([d, oh], axis=1)
    d = d.dropna(subset=["yh", "yd", "ya"]).copy()

    # Brier / log-loss
    d["brier_o"] = d.apply(lambda r: brier_multi(
        np.array([r.ph_o, r.pd_o, r.pa_o]),
        np.array([r.yh,   r.yd,   r.ya])), axis=1)
    d["brier_c"] = d.apply(lambda r: brier_multi(
        np.array([r.ph_c, r.pd_c, r.pa_c]),
        np.array([r.yh,   r.yd,   r.ya])), axis=1)
    d["ll_o"] = d.apply(lambda r: log_loss_multi(
        np.array([r.ph_o, r.pd_o, r.pa_o]),
        np.array([r.yh,   r.yd,   r.ya])), axis=1)
    d["ll_c"] = d.apply(lambda r: log_loss_multi(
        np.array([r.ph_c, r.pd_c, r.pa_c]),
        np.array([r.yh,   r.yd,   r.ya])), axis=1)

    # Overround
    d["or_o"] = d.apply(lambda r: overround_pct(r.B365H, r.B365D, r.B365A),  axis=1)
    d["or_c"] = d.apply(lambda r: overround_pct(r.B365CH, r.B365CD, r.B365CA), axis=1)

    # Pinnacle closing — optional
    ps_cols = ["PSCH", "PSCD", "PSCA"]
    if all(c in df.columns for c in ps_cols):
        df_ps = df[["Date", "FTR"] + ps_cols].copy()
        for c in ps_cols:
            df_ps[c] = pd.to_numeric(df_ps[c], errors="coerce")
        df_ps = df_ps.dropna(subset=ps_cols + ["FTR"])
        nv_ps = df_ps.apply(lambda r: pd.Series(
            no_vig(r.PSCH, r.PSCD, r.PSCA), index=["ps_h", "ps_d", "ps_a"]), axis=1)
        df_ps = pd.concat([df_ps, nv_ps], axis=1)
        oh_ps = df_ps["FTR"].apply(lambda x: pd.Series(
            outcome_onehot(x), index=["yh", "yd", "ya"]))
        df_ps = pd.concat([df_ps, oh_ps], axis=1).dropna()
        d["ps_h"] = d.index.map(df_ps["ps_h"])
        d["ps_d"] = d.index.map(df_ps["ps_d"])
        d["ps_a"] = d.index.map(df_ps["ps_a"])

    return d.reset_index(drop=True)


# ── PRINT KEY NUMBERS ─────────────────────────────────────────────────────────
def print_key_numbers(d: pd.DataFrame):
    print("\n" + "="*65)
    print("  KEY NUMBERS FOR RESULTS SECTION")
    print("="*65)
    n = len(d)
    print(f"\n  N matches (Bet365, full dataset): {n:,}")

    # Brier
    m_bo, lo_bo, hi_bo = bootstrap_mean_ci(d["brier_o"].values)
    m_bc, lo_bc, hi_bc = bootstrap_mean_ci(d["brier_c"].values)
    stat_b, pval_b = stats.wilcoxon(d["brier_o"].values, d["brier_c"].values)

    print(f"\n  BRIER SCORE")
    print(f"    Opening : {m_bo:.4f}  (95% CI: {lo_bo:.4f}–{hi_bo:.4f})")
    print(f"    Closing : {m_bc:.4f}  (95% CI: {lo_bc:.4f}–{hi_bc:.4f})")
    print(f"    Delta   : {m_bo - m_bc:+.4f}")
    print(f"    Wilcoxon signed-rank: W = {stat_b:.1f}, p = {pval_b:.4f}")

    # Log loss
    m_lo, lo_lo, hi_lo = bootstrap_mean_ci(d["ll_o"].values)
    m_lc, lo_lc, hi_lc = bootstrap_mean_ci(d["ll_c"].values)
    stat_l, pval_l = stats.wilcoxon(d["ll_o"].values, d["ll_c"].values)

    print(f"\n  LOG LOSS")
    print(f"    Opening : {m_lo:.4f}  (95% CI: {lo_lo:.4f}–{hi_lo:.4f})")
    print(f"    Closing : {m_lc:.4f}  (95% CI: {lo_lc:.4f}–{hi_lc:.4f})")
    print(f"    Delta   : {m_lo - m_lc:+.4f}")
    print(f"    Wilcoxon signed-rank: W = {stat_l:.1f}, p = {pval_l:.4f}")

    # Overround
    print(f"\n  OVERROUND")
    print(f"    Opening mean : {d['or_o'].mean():.2f}%")
    print(f"    Closing mean : {d['or_c'].mean():.2f}%")
    diff = d["or_c"] - d["or_o"]
    print(f"    SD (close − open, per match): {diff.std():.4f}")

    # Phase split
    p1 = d[d["Date"] < PHASE_SPLIT]
    p2 = d[d["Date"] >= PHASE_SPLIT]
    d1 = (p1["or_c"] - p1["or_o"]).std()
    d2 = (p2["or_c"] - p2["or_o"]).std()
    print(f"\n  PHASE SPLIT  (boundary: {PHASE_SPLIT.date()})")
    print(f"    Phase 1 (Aug 2019 – May 2022): n={len(p1):,},  OR SD = {d1:.4f}")
    print(f"    Phase 2 (May 2022 – present):  n={len(p2):,},  OR SD = {d2:.4f}")

    # Strategy P&L
    print(f"\n  NAIVE STRATEGY CUMULATIVE PROFIT ($100 stake)")
    for outcome, col in [("Home win", "B365CH"), ("Draw", "B365CD"), ("Away win", "B365CA")]:
        if col not in d.columns:
            continue
        ftr_code = {"B365CH": "H", "B365CD": "D", "B365CA": "A"}[col]
        wins = (d["FTR"] == ftr_code)
        pnl  = np.where(wins, (d[col] - 1) * STAKE, -STAKE)
        cp   = np.cumsum(pnl)
        cp1  = np.cumsum(pnl[d["Date"] < PHASE_SPLIT])[-1] if (d["Date"] < PHASE_SPLIT).any() else 0
        cp2  = cp[-1] - cp1
        print(f"    {outcome:<10}: Phase 1 = ${cp1:>+9,.0f}  |  Phase 2 = ${cp2:>+9,.0f}  |  Total = ${cp[-1]:>+9,.0f}")

    print("\n" + "="*65 + "\n")


# ── FIGURE 1: RELIABILITY DIAGRAM ─────────────────────────────────────────────
def fig1_reliability(d: pd.DataFrame):
    print("Generating Figure 1: Calibration reliability diagram...")

    has_ps = all(c in d.columns for c in ["ps_h", "ps_d", "ps_a"])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    outcome_cfg = [
        ("Home Win",  "ph_c", "yh", "ps_h"),
        ("Draw",      "pd_c", "yd", "ps_d"),
        ("Away Win",  "pa_c", "ya", "ps_a"),
    ]

    for ax, (label, p_b365, y_col, p_ps) in zip(axes, outcome_cfg):
        # Bet365 closing
        mp, mo, ct = reliability_curve(d[p_b365].values, d[y_col].values)
        ax.plot(mp, mo, "o-", color=C_BLUE,  linewidth=1.8, markersize=6,
                label="Bet365 closing", zorder=3)
        # Pinnacle closing
        if has_ps and p_ps in d.columns and not d[p_ps].isna().all():
            mp_ps, mo_ps, _ = reliability_curve(
                d[p_ps].dropna().values,
                d.loc[d[p_ps].notna(), y_col].values)
            ax.plot(mp_ps, mo_ps, "s--", color=C_RED, linewidth=1.8, markersize=6,
                    label="Pinnacle closing", zorder=3)
        # Reference
        ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("No-vig predicted probability")
        ax.set_ylabel("Observed outcome frequency")
        ax.set_title(f"({chr(65 + outcome_cfg.index((label, p_b365, y_col, p_ps)))}) {label}")
        ax.legend(fontsize=9, loc="upper left")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Figure 1.  Reliability diagrams for Bet365 and Pinnacle closing no-vig probabilities "
        "(EPL 2019/20–2025/26)",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / "fig1_calibration_reliability.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ── FIGURE 2: OVERROUND TIME SERIES ───────────────────────────────────────────
def fig2_overround(d: pd.DataFrame):
    print("Generating Figure 2: Overround time series...")

    ds = d.sort_values("Date").copy()
    or_o_ma = ds["or_o"].rolling(ROLLING_N, min_periods=10).mean()
    or_c_ma = ds["or_c"].rolling(ROLLING_N, min_periods=10).mean()
    diff = ds["or_c"] - ds["or_o"]
    diff_roll_sd = diff.rolling(ROLLING_N, min_periods=10).std()

    sd1 = diff[ds["Date"] < PHASE_SPLIT].std()
    sd2 = diff[ds["Date"] >= PHASE_SPLIT].std()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 2]},
        sharex=True
    )

    # Panel A — rolling MA overround
    ax1.plot(ds["Date"], or_o_ma, color=C_BLUE, linewidth=1.8,
             label="Opening odds overround")
    ax1.plot(ds["Date"], or_c_ma, color=C_RED,  linewidth=1.8,
             label="Closing odds overround")
    ax1.axvline(PHASE_SPLIT, color=C_GRAY, linewidth=1.4, linestyle="--")
    ax1.set_ylabel("Overround (%)")
    ax1.set_ylim(4.5, 6.8)
    ax1.legend(fontsize=9)
    ax1.set_title(
        f"(A) Bet365 overround: {ROLLING_N}-match rolling mean, August 2019–2026",
        fontsize=10
    )

    # Shade phases
    xmin = ds["Date"].min(); xmax = ds["Date"].max()
    ax1.axvspan(xmin, PHASE_SPLIT, alpha=0.04, color=C_BLUE)
    ax1.axvspan(PHASE_SPLIT, xmax, alpha=0.04, color=C_RED)
    ax1.text(pd.Timestamp("2020-09-01"), 6.6, "Phase 1", fontsize=9, color=C_BLUE)
    ax1.text(pd.Timestamp("2023-09-01"), 6.6, "Phase 2", fontsize=9, color=C_RED)

    # Panel B — rolling SD of close−open difference
    ax2.fill_between(ds["Date"], diff_roll_sd, color=C_GREEN, alpha=0.4)
    ax2.plot(ds["Date"], diff_roll_sd, color=C_GREEN, linewidth=1.5)
    ax2.axvline(PHASE_SPLIT, color=C_GRAY, linewidth=1.4, linestyle="--")
    ax2.set_ylabel("Rolling SD\n(close − open, %)")
    ax2.set_xlabel("Date")
    ax2.set_title(
        "(B) Per-match volatility between opening and closing overround",
        fontsize=10
    )
    # Annotate overall SD values
    mid1 = xmin + (PHASE_SPLIT - xmin) / 2
    mid2 = PHASE_SPLIT + (xmax - PHASE_SPLIT) / 2
    ymax = diff_roll_sd.max()
    ax2.text(mid1, ymax * 0.88, f"Overall SD = {sd1:.3f}", fontsize=9,
             ha="center", color=C_BLUE,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BLUE, alpha=0.8))
    ax2.text(mid2, ymax * 0.88, f"Overall SD = {sd2:.3f}", fontsize=9,
             ha="center", color=C_RED,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_RED, alpha=0.8))

    fig.suptitle(
        "Figure 2.  Overround dynamics between opening and closing Bet365 1X2 odds "
        "(EPL 2019/20–2025/26)",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / "fig2_overround_timeseries.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ── FIGURE 3: CUMULATIVE PROFIT ────────────────────────────────────────────────
def fig3_cumulative_profit(d: pd.DataFrame):
    print("Generating Figure 3: Cumulative profit...")

    strats = [
        ("Always home win", "H", "B365CH", C_BLUE),
        ("Always draw",     "D", "B365CD", C_GREEN),
        ("Always away win", "A", "B365CA", C_RED),
    ]
    ds = d.sort_values("Date").copy()

    # Compute P&L series
    series = {}
    for label, ftr_code, odds_col, _ in strats:
        if odds_col not in ds.columns:
            continue
        wins = (ds["FTR"] == ftr_code)
        pnl  = np.where(wins, (ds[odds_col] - 1) * STAKE, -STAKE)
        series[label] = (ds["Date"].values, np.cumsum(pnl), ftr_code, odds_col)

    phase_idx = int((ds["Date"] < PHASE_SPLIT).sum())

    fig, (ax_ts, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # Panel A — time series
    for label, ftr_code, odds_col, color in strats:
        if label not in series:
            continue
        dates, cp, _, _ = series[label]
        ax_ts.plot(dates, cp, color=color, linewidth=1.8, label=label)

    ax_ts.axhline(0, color="black", linewidth=0.8)
    ax_ts.axvline(PHASE_SPLIT, color=C_GRAY, linewidth=1.4, linestyle="--",
                  label=f"Phase boundary ({PHASE_SPLIT.strftime('%b %Y')})")
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Cumulative profit ($)")
    ax_ts.set_title("(A) Cumulative profit: $100 fixed stake per match")
    ax_ts.legend(fontsize=9)

    # Shade phases
    xmin = ds["Date"].min(); xmax = ds["Date"].max()
    ax_ts.axvspan(xmin, PHASE_SPLIT, alpha=0.04, color=C_BLUE)
    ax_ts.axvspan(PHASE_SPLIT, xmax, alpha=0.04, color=C_RED)

    # Panel B — phase split bar chart
    labels_bar, p1_vals, p2_vals = [], [], []
    for label, ftr_code, odds_col, _ in strats:
        if label not in series:
            continue
        _, cp, _, _ = series[label]
        labels_bar.append(label.replace("Always ", ""))
        p1_vals.append(float(cp[phase_idx - 1]) if phase_idx > 0 else 0)
        p2_vals.append(float(cp[-1]) - (float(cp[phase_idx - 1]) if phase_idx > 0 else 0))

    x  = np.arange(len(labels_bar))
    bw = 0.38

    def bar_color(v):
        return C_BLUE if v >= 0 else C_RED

    b1 = ax_bar.bar(x - bw/2, p1_vals, bw,
                    color=[bar_color(v) for v in p1_vals], alpha=0.9,
                    label="Phase 1 (2019–22)")
    b2 = ax_bar.bar(x + bw/2, p2_vals, bw,
                    color=[bar_color(v) for v in p2_vals], alpha=0.5,
                    label="Phase 2 (2022–26)")

    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_bar, fontsize=9)
    ax_bar.set_ylabel("Cumulative profit ($)")
    ax_bar.set_title("(B) Profit by phase")
    ax_bar.legend(fontsize=8)

    # Value labels
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        va   = "bottom" if h >= 0 else "top"
        ypos = h + 300 if h >= 0 else h - 300
        ax_bar.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"${h:,.0f}", ha="center", va=va, fontsize=7.5)

    fig.suptitle(
        "Figure 3.  Cumulative profit from three naive fixed-stake strategies, "
        "Bet365 closing odds (EPL 2019/20–2025/26, $100 stake per match)",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / "fig3_cumulative_profit.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ── FIGURE 4: TEAM HOME PROFITABILITY ─────────────────────────────────────────
def fig4_team_profitability(d: pd.DataFrame):
    print("Generating Figure 4: Team-level home profitability...")

    if "B365CH" not in d.columns or "HomeTeam" not in d.columns:
        print("  Skipping Figure 4 — required columns not available.")
        return

    teams = {}
    for _, row in d.iterrows():
        t = row.get("HomeTeam")
        if pd.isna(t):
            continue
        win = row["FTR"] == "H"
        pnl = (row["B365CH"] - 1) * STAKE if win else -STAKE
        if t not in teams:
            teams[t] = {"pnl": 0.0, "n": 0, "wins": 0, "odds_sum": 0.0}
        teams[t]["pnl"]      += pnl
        teams[t]["n"]        += 1
        teams[t]["wins"]     += int(win)
        teams[t]["odds_sum"] += row["B365CH"]

    team_df = pd.DataFrame([
        {
            "team":    t,
            "profit":  v["pnl"],
            "n":       v["n"],
            "hw_pct":  100 * v["wins"] / v["n"] if v["n"] > 0 else 0,
            "avg_odds": v["odds_sum"] / v["n"] if v["n"] > 0 else np.nan,
        }
        for t, v in teams.items()
    ])
    team_df = team_df[team_df["n"] >= 20].sort_values("profit")

    fig_h = max(6, len(team_df) * 0.33)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    colors = [C_GREEN if p >= 0 else C_RED for p in team_df["profit"]]
    bars   = ax.barh(team_df["team"], team_df["profit"],
                     color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.9)

    # Annotate: home-win % and average closing odds
    for bar, (_, row) in zip(bars, team_df.iterrows()):
        xw = bar.get_width()
        offset = max(abs(team_df["profit"].max()) * 0.02, 80)
        ha = "left" if xw >= 0 else "right"
        xpos = xw + offset if xw >= 0 else xw - offset
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"HW: {row['hw_pct']:.0f}%  |  avg odds: {row['avg_odds']:.2f}",
                va="center", ha=ha, fontsize=7.5, color="#333333")

    ax.set_xlabel("Total cumulative profit ($)")
    ax.set_title(
        "Figure 4.  Cumulative profit from backing each EPL team at home\n"
        "(Bet365 closing odds, 2019/20–2025/26, $100 stake; ≥20 home matches)",
        fontsize=10
    )
    plt.tight_layout()
    out = FIG_DIR / "fig4_team_home_profitability.png"
    fig.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ── TABLE 1: FOUR-BOOKMAKER CALIBRATION (2024/25 season only) ────────────────
def table1_bookmaker_comparison(df: pd.DataFrame):
    """
    Replicate the four-bookmaker calibration table from the student's draft,
    but with correct numbers from the pipeline.
    Uses 2024/25 season only (the one season where all four have data).
    Bookmakers: Pinnacle (PS*C), Bet365 (B365*C), William Hill (WH*C), Bet&Win/Betway (BW*C)
    """
    print("\nGenerating Table 1: Four-bookmaker calibration (2024/25)...")

    season_mask = (df["Date"] >= "2024-08-01") & (df["Date"] < "2025-06-01")
    df_s = df[season_mask].copy()

    bookmakers = {
        "Pinnacle":          ("PSCH",  "PSCD",  "PSCA"),
        "Bet365":            ("B365CH","B365CD","B365CA"),
        "William Hill":      ("WHCH",  "WHCD",  "WHCA"),
        "Betway / Bet&Win":  ("BWCH",  "BWCD",  "BWCA"),
    }

    rows = []
    for bk_name, (hc, dc, ac) in bookmakers.items():
        if not all(c in df_s.columns for c in [hc, dc, ac]):
            print(f"  Note: {bk_name} closing columns not found — skipping")
            continue
        tmp = df_s[[hc, dc, ac, "FTR"]].copy()
        for c in [hc, dc, ac]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp["FTR"] = tmp["FTR"].astype(str).str.strip()
        tmp = tmp.dropna(subset=[hc, dc, ac])
        tmp = tmp[tmp["FTR"].isin(["H", "D", "A"])]
        if len(tmp) < 10:
            print(f"  Note: {bk_name} has only {len(tmp)} rows — skipping")
            continue

        nv = tmp.apply(lambda r: pd.Series(
            no_vig(r[hc], r[dc], r[ac]), index=["ph", "pd_", "pa"]), axis=1)
        tmp = pd.concat([tmp, nv], axis=1)
        oh = tmp["FTR"].apply(lambda x: pd.Series(
            outcome_onehot(x), index=["yh", "yd", "ya"]))
        tmp = pd.concat([tmp, oh], axis=1).dropna(subset=["yh"])

        briers   = tmp.apply(lambda r: brier_multi(
            np.array([r.ph, r.pd_, r.pa]),
            np.array([r.yh, r.yd,  r.ya])), axis=1)
        lls      = tmp.apply(lambda r: log_loss_multi(
            np.array([r.ph, r.pd_, r.pa]),
            np.array([r.yh, r.yd,  r.ya])), axis=1)
        orounds  = tmp.apply(lambda r: overround_pct(r[hc], r[dc], r[ac]), axis=1)

        b_mean, b_lo, b_hi = bootstrap_mean_ci(briers.values)
        l_mean, l_lo, l_hi = bootstrap_mean_ci(lls.values)

        rows.append({
            "Bookmaker":    bk_name,
            "N":            len(tmp),
            "Avg OR (%)":   f"{orounds.mean():.2f}",
            "Brier":        f"{b_mean:.4f}",
            "Brier 95% CI": f"[{b_lo:.4f}–{b_hi:.4f}]",
            "Log Loss":     f"{l_mean:.4f}",
            "LL 95% CI":    f"[{l_lo:.4f}–{l_hi:.4f}]",
        })

    if not rows:
        print("  No bookmaker data found for 2024/25 — check column names in your CSVs.")
        return

    # Sort by Brier ascending (best first)
    rows.sort(key=lambda r: float(r["Brier"]))
    for i, r in enumerate(rows, 1):
        r["Rank"] = i

    print("\n  TABLE 1 — Bookmaker calibration (2024/25 season, closing no-vig odds)")
    print(f"  {'Rank':<5} {'Bookmaker':<20} {'N':>5} {'OR%':>7} {'Brier':>8} "
          f"{'95% CI Brier':<22} {'LogLoss':>9} {'95% CI LL':<22}")
    print("  " + "-"*100)
    for r in rows:
        print(f"  {r['Rank']:<5} {r['Bookmaker']:<20} {r['N']:>5} "
              f"{r['Avg OR (%)']:>7} {r['Brier']:>8} "
              f"{r['Brier 95% CI']:<22} {r['Log Loss']:>9} {r['LL 95% CI']:<22}")
    print()


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\nLoading data...")
    df_raw = load_all_seasons()

    print("Computing metrics...")
    d = build_metrics(df_raw)

    print_key_numbers(d)
    table1_bookmaker_comparison(df_raw)

    fig1_reliability(d)
    fig2_overround(d)
    fig3_cumulative_profit(d)
    fig4_team_profitability(d)

    print("\nDone. All figures saved to figures/")
    print("Copy the KEY NUMBERS block above into the Results section.")


if __name__ == "__main__":
    np.random.seed(42)
    main()