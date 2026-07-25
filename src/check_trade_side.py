"""Verify what Databento's trade `side` means, using the cached data.

For every cached spread trade, look up the spread book state as of the
trade's timestamp and check whether the trade printed at the bid or at the
ask:
- printed at the ASK  -> a buyer aggressed (lifted the ask)
- printed at the BID  -> a seller aggressed (hit the bid)

If side 'B' trades print at the ask and side 'A' trades print at the bid,
the Databento convention is: side = aggressor side, 'B' = buyer-initiated,
'A' = seller-initiated (what our schema/PnLCalculator assumes).

Run:
    uv run src/check_trade_side.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from pnl_calculator import OrderBook

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
SYMBOL = "ESM6-ESU6"


def main() -> None:
    trade_files = sorted(CACHE_DIR.glob(f"{SYMBOL}_trades_*.pkl"))
    tally: Counter[tuple[str, str]] = Counter()
    examples_printed = 0

    for tf in trade_files:
        trades = pd.read_pickle(tf)
        if trades.empty:
            continue
        book_file = Path(str(tf).replace("_trades_", "_mbp-1_"))
        if not book_file.exists():
            continue
        book_df = pd.read_pickle(book_file)
        if book_df.empty:
            continue
        book = OrderBook(SYMBOL, book_df)

        for ts, tr in trades.iterrows():
            state = book.state_at(ts)
            if state is None:
                tally[(tr["side"], "no book state")] += 1
                continue
            if tr["price"] >= state["ask_px_00"]:
                where = "at ASK (buyer aggressed)"
            elif tr["price"] <= state["bid_px_00"]:
                where = "at BID (seller aggressed)"
            else:
                where = "inside spread (?)"
            tally[(tr["side"], where)] += 1

            if examples_printed < 10:
                print(
                    f"{ts}  side={tr['side']}  price={tr['price']:.2f}  "
                    f"book {state['bid_px_00']:.2f}/{state['ask_px_00']:.2f}  -> {where}"
                )
                examples_printed += 1

    print("\nSummary (side, where it printed) -> count")
    for (side, where), n in sorted(tally.items()):
        print(f"  side={side!r:4}  {where:28} {n}")

    total = sum(tally.values())
    consistent = tally[("B", "at ASK (buyer aggressed)")] + tally[("A", "at BID (seller aggressed)")]
    print(f"\n{consistent}/{total} trades consistent with: "
          "'B' = buyer-initiated, 'A' = seller-initiated (Databento convention)")


if __name__ == "__main__":
    main()
