import math

import pytest

from optlab.pricing import bs_delta, bs_price, implied_vol, strike_for_delta


def test_put_call_parity():
    s, k, t, v = 100.0, 95.0, 0.25, 0.2
    assert bs_price(s, k, t, v, "C") - bs_price(s, k, t, v, "P") == pytest.approx(s - k, abs=1e-9)


def test_implied_vol_round_trip():
    price = bs_price(400, 380, 40 / 365, 0.22, "P")
    assert implied_vol(price, 400, 380, 40 / 365, "P") == pytest.approx(0.22, abs=1e-5)


def test_strike_for_delta_inverts_delta():
    k = strike_for_delta(100, 45 / 365, 0.25, "P", 0.30)
    assert bs_delta(100, k, 45 / 365, 0.25, "P") == pytest.approx(-0.30, abs=1e-6)
    assert k < 100


def test_expiry_is_intrinsic():
    assert bs_price(90, 100, 0, 0.3, "P") == 10
    assert math.isclose(bs_price(110, 100, 0, 0.3, "C"), 10)
