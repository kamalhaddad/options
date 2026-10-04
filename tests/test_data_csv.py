import pandas as pd
import pytest

from optlab.data import CsvMarket, read_cboe_index, read_chains


def _export(market, dates, path):
    frames = []
    for d in dates:
        ch = market.chain(d, "SPY", 20, 60)
        frames.append(
            pd.DataFrame(
                {
                    "underlying_symbol": "SPY",
                    "quote_date": d.strftime("%Y-%m-%d"),
                    "expiration": ch.expiration.dt.strftime("%Y-%m-%d"),
                    "strike": ch.strike,
                    "option_type": ch.kind.map({"C": "call", "P": "put"}),
                    "bid": ch.bid,
                    "ask": ch.ask,
                    "underlying_price": ch.underlying,
                }
            )
        )
    pd.concat(frames).to_csv(path, index=False)


def test_csv_round_trip_fills_greeks(market, tmp_path):
    dates = market.dates[300:303]
    _export(market, dates, tmp_path / "chains.csv")
    vix = tmp_path / "VIX_History.csv"
    pd.DataFrame({"DATE": market.vix().index.strftime("%m/%d/%Y"), "CLOSE": market.vix().values}).to_csv(
        vix, index=False
    )
    m = CsvMarket(read_chains([tmp_path / "chains.csv"]), read_cboe_index(vix), read_cboe_index(vix))
    assert list(m.dates) == list(dates)
    ch = m.chain(dates[0], "SPY", 30, 45)
    assert {"delta", "iv", "mid"} <= set(ch.columns)
    puts = ch[(ch.kind == "P") & ch.delta.notna()]
    near30 = puts.loc[(puts.delta + 0.30).abs().idxmin()]
    original = market.chain(dates[0], "SPY", 30, 45)
    ref = original[
        (original.kind == "P") & (original.strike == near30.strike) & (original.expiration == near30.expiration)
    ]
    # Delta recomputed from the quoted mid lands near the source's own delta.
    assert near30.delta == pytest.approx(float(ref.delta.iloc[0]), abs=0.03)
    q = m.quote(dates[0], "SPY", near30.expiration, near30.strike, "P")
    assert q == (near30.bid, near30.ask)
    # A symbol with no quotes that day gives an empty chain, not an error.
    assert m.chain(dates[0], "XLF", 30, 45).empty
    assert m.atm_iv(dates[0], "XLF") is None
