import pytest

from optlab.config import Config
from optlab.data import SyntheticMarket, SyntheticParams


@pytest.fixture(scope="session")
def market():
    return SyntheticMarket(["AAA"], SyntheticParams(seed=3, start="2015-01-01", end="2017-06-30"))


@pytest.fixture
def cfg():
    c = Config()
    c.account.spreads_allowed = True
    c.universe.stocks = ["AAA"]
    return c
