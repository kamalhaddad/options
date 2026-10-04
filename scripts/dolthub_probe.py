# ruff: noqa: E501
"""Print what DoltHub's free options and stocks databases cover."""

import json
import sys
import urllib.parse
import urllib.request

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/{db}/master?q={q}"


def sql(db: str, q: str):
    url = API.format(db=db, q=urllib.parse.quote(q))
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    if d.get("query_execution_status") != "Success":
        print("ERR", db, q, d.get("query_execution_message"))
    return d.get("rows", [])


for db, q in [
    ("options", "SHOW TABLES"),
    ("options", "DESCRIBE option_chain"),
    ("options", "SELECT MIN(date), MAX(date) FROM option_chain WHERE act_symbol='SPY'"),
    ("options", "SELECT COUNT(*) FROM option_chain WHERE act_symbol='SPY' AND date='2024-03-04'"),
    (
        "options",
        "SELECT * FROM option_chain WHERE act_symbol='SPY' AND date='2024-03-04' AND call_put='Put' ORDER BY expiration, strike LIMIT 60",
    ),
    (
        "options",
        "SELECT date, COUNT(*) FROM option_chain WHERE act_symbol='SPY' AND date BETWEEN '2024-03-01' AND '2024-03-15' GROUP BY date",
    ),
    (
        "options",
        "SELECT act_symbol, COUNT(DISTINCT date) FROM option_chain WHERE act_symbol IN ('T','PFE','INTC','F','SOFI','KMI','XLF','BAC') GROUP BY act_symbol",
    ),
    ("stocks", "SHOW TABLES"),
    ("stocks", "SELECT MIN(date), MAX(date) FROM ohlcv WHERE act_symbol='SPY'"),
]:
    print("==", db, q)
    for row in sql(db, q):
        print(row)
    sys.stdout.flush()
