# optlab

A backtester and daily scanner for a rules-based options strategy: sell
defined-risk premium on liquid indices at 30 to 45 days, only when the
volatility regime is calm and options are rich; buy in-the-money options
only when they are cheap and the trend agrees; keep a small crash hedge.
Every rule and number comes from the project's strategy playbook, and the
defaults are set for a $5,000 cash account.

The scanner only suggests trades. Nothing in this repo places orders or
talks to a broker.

## Quick start

```bash
pip install -e .[dev]          # add [live] for Yahoo data
pytest -q

# Backtest on simulated data (no downloads needed)
optlab backtest --source synthetic --config config/spreads.toml \
    --start 2012-01-01 --end 2020-12-31 --out out/

# Today's scan from free, delayed Yahoo data
pip install -e .[live]
optlab scan --source yahoo

# Paper trading: same scan, recorded in journal/paper.json
optlab paper --source yahoo
```

`optlab backtest` prints CAGR, max drawdown, worst month, Sharpe, Sortino,
win rate and average win and loss, next to a put-writing benchmark, then
runs the playbook's validation gate (Sharpe above 0.8, drawdown under 25%,
and better than the benchmark on both).

## What the rules do

| Playbook rule | Where |
|---|---|
| Regime: VIX/VIX3M, VIX level, SPY vs 200-day, VIX vs 20-day realized; worst check wins; green 1.0, yellow 0.5, red 0 | `signals.regime` |
| IV Rank, IV minus realized vol, trend (50/200-day), 12-1 momentum, straddle momentum | `signals.py` |
| Index sleeve: one new rung a week. Bullish: 30-delta put spread. Neutral: 16-delta iron condor. Bearish: 25-delta call spread. Only when IV Rank is 30 or more | `planner.py` |
| Single-stock sleeve: richest IV Rank and IV-RV first, skip names whose straddles paid off last year | `planner.py` |
| Directional sleeve: 65-delta calls or puts at 60 to 120 DTE when IV Rank is 25 or less, IV is under realized, and trend and momentum agree | `planner.py` |
| Hedge: 10% of premium collected saved toward 5-delta puts at 60 to 90 DTE, rolled at 30 DTE | `planner.py` |
| Exits: 50% profit, 2x credit stop, 21 DTE; long options +100%, -50%, 30 DTE | `Planner.exit_reason` |
| Sizing: 2% max loss per trade, 25% combined, stress test (-10%, IV doubles) under 15%, 10-point vega shock under 8% | `risk.py` |
| Drawdown breakers: half size at 10%, stop selling at 15% until 5 days of contango | `backtest.py`, `scanner.py` |
| Fills: mid minus a quarter of the bid-ask spread, plus $0.65 a contract | `trades.fill_price` |

All settings live in `optlab/config.py`; override any of them with a TOML
file (`config/default.toml` lists the common ones).

### Choices made where the playbook was silent

- **Yellow regime and drawdowns halve each sleeve's total budget, not the
  per-trade cap.** At $5,000, half of the 2% cap is $50, which is less than
  one $1-wide spread risks, so halving per trade would mean no trades at all.
- **Spreads are off by default** (`spreads_allowed = false`) because most
  brokers do not allow them in a cash account. With them off, short premium
  falls back to cash-secured puts, which a $5,000 account almost never fits
  under a 2% cap. Use `config/spreads.toml` if your broker allows spreads.
- **Exits are checked on daily closes.** Real good-till-canceled orders fill
  intraday, so stops in the backtest can be worse than live and profit
  targets can be late by a day.

## Data

The engine reads everything through `optlab.data.MarketData`, so sources
can be swapped without touching the rules.

| Source | Use | Cost |
|---|---|---|
| `synthetic` | Tests and demos. Stochastic-volatility prices with crash jumps, Black-Scholes quotes with skew and spreads. Says nothing about real profitability | Free |
| `yahoo` | Daily scan: price history, ^VIX, ^VIX3M, today's chains (delayed) | Free |
| `csv` | Backtests on real chains. Optopsy-style columns: `underlying_symbol, quote_date, expiration, strike, option_type, bid, ask, underlying_price`, plus optional `implied_volatility, delta` | Your data |

VIX, VIX3M and the PutWrite index (the benchmark) are free daily CSVs from
Cboe (`VIX_History.csv`, `VIX3M_History.csv`, `PUT_History.csv` at
cdn.cboe.com). Historical option chains are not free; ORATS, Cboe DataShop
and Polygon.io sell them, and any of their exports can be renamed to the
CSV layout above.

```bash
optlab backtest --source csv --chains spy_chains_2007_2026.parquet \
    --vix VIX_History.csv --vix3m VIX3M_History.csv --put-index PUT_History.csv
```

## Daily routine

`optlab paper` marks open paper positions, closes anything that hit a target,
stop or time exit, and on the first trading day of each week opens new
trades that pass every check. It writes `journal/paper.json` (git-ignored, so
account details never reach this public repo) and prints a report like:

```
**Regime: YELLOW** (size multiplier 0.5)
VIX 17.2, VIX/VIX3M 0.93, SPY vs 200-day +3.1%, VIX minus realized +4.0 pts
## Close these
- #3 SPY put_spread x1: profit target 50%, close near -0.12/share, P&L about $9.40
## New trades
- SPY iron_condor x1 (index): Sell ... Limit credit 0.24, max loss $76.00
```

`.github/workflows/daily-scan.yml` runs the scan in GitHub Actions and posts
the report to the run summary. It is manual-only until the schedule is
switched on.

## Known gaps

- No earnings calendar yet, so single-stock trades do not skip earnings.
- No ex-dividend or pin-risk handling; trades exit at 21 DTE, well before both.
- Single-stock IV Rank needs a year of IV history. The scanner records it
  daily in the journal; until then those names show IV Rank as n/a.
- Yahoo sometimes blocks cloud servers; the scan then needs another source.
