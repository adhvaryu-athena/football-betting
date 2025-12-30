"""Odds and pricing utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union

import numpy as np

OddsInput = Union[Iterable[float], Tuple[float, float, float]]
NoVigProbs = Dict[str, float]


def _validate_odds(odds: np.ndarray) -> None:
    """Validate odds array is 1-D with three strictly positive finite values."""
    if odds.shape != (3,):
        raise ValueError("Expected exactly three odds values: home, draw, away.")
    if not np.isfinite(odds).all():
        raise ValueError("Odds must be finite numeric values.")
    if (odds <= 0).any():
        raise ValueError("Odds must be greater than zero.")


def compute_no_vig_probs(
    h_odds: float, d_odds: float, a_odds: float
) -> NoVigProbs:
    """
    Convert bookmaker decimal odds into normalized no-vig probabilities.

    Implied probabilities are computed as 1 / odds for each outcome and then
    normalized by the overround so the returned probabilities sum to 1. The
    function raises ValueError for invalid odds (non-positive or non-finite).

    Args:
        h_odds: Decimal odds for home win.
        d_odds: Decimal odds for draw.
        a_odds: Decimal odds for away win.

    Returns:
        Dictionary with keys pH, pD, pA containing no-vig probabilities.
    """
    odds = np.array([h_odds, d_odds, a_odds], dtype=float)
    _validate_odds(odds)

    implied = 1.0 / odds
    overround = implied.sum()
    if overround <= 0:
        raise ValueError("Overround must be positive.")

    normalized = implied / overround
    return {"pH": float(normalized[0]), "pD": float(normalized[1]), "pA": float(normalized[2])}
