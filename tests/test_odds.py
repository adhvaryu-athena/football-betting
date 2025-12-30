import math

import pytest

from football_betting.odds import compute_no_vig_probs


def test_no_vig_probabilities_sum_to_one_and_match_expected():
    # Typical overround book
    result = compute_no_vig_probs(2.0, 3.5, 4.0)
    assert pytest.approx(result["pH"], rel=1e-6) == 0.4827586207
    assert pytest.approx(result["pD"], rel=1e-6) == 0.275862069
    assert pytest.approx(result["pA"], rel=1e-6) == 0.2413793103
    assert pytest.approx(sum(result.values()), rel=1e-12) == 1.0


def test_fair_book_normalizes_evenly():
    result = compute_no_vig_probs(2.0, 2.0, 2.0)
    assert all(pytest.approx(p, rel=1e-12) == 1 / 3 for p in result.values())


@pytest.mark.parametrize(
    "h,d,a",
    [
        (0, 3.0, 4.0),
        (2.0, -1.0, 4.0),
        (2.0, 3.0, math.inf),
    ],
)
def test_invalid_odds_raise_value_error(h, d, a):
    with pytest.raises(ValueError):
        compute_no_vig_probs(h, d, a)
