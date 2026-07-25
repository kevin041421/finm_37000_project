"""Demo script: print ES front/back/spread order books every hour.

Now runs on the real #6/#7 ``MarketDataFetcher`` — symbols, books and the
contract spec are resolved from product_specs.json and Databento rather than
hardcoded. The TEMP ``HardcodedESFetcher`` is gone; its disk caching and
retry behaviour moved into ``CachedMarketDataFetcher``.

Run:
    uv run src/demo_print_books.py

Requires ~/.databento_api_key on the first (uncached) run; see src/util.py.
"""

from __future__ import annotations

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher

# Instrument under test — the back leg (U6) is derived from ES's listing
# cycle in product_specs.json, not hardcoded.
PRODUCT = "ES"
FRONT_MONTH = "M6"

# Two days to demo (ESM6 and ESU6 both active, per the README snapshot week)
DAYS = ["2026-06-09", "2026-06-10"]
HOURS = range(24)
# We only need the book state at each hour, so fetch a short window and
# take the first message per instrument as the snapshot.
WINDOW = pd.Timedelta(seconds=60)

BOOK_LEVELS = 1  # top of book is all this script prints


def _format_top_of_book(name: str, book: pd.DataFrame) -> str:
    if book.empty:
        return f"  {name:<12} (no data)"
    row = book.iloc[0]  # first message in the window = state at the hour
    return (
        f"  {name:<12} "
        f"bid {row['bid_sz_00']:>5.0f} x {row['bid_px_00']:>10.2f}   |   "
        f"ask {row['ask_px_00']:>10.2f} x {row['ask_sz_00']:<5.0f}"
    )


def main() -> None:
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)

    for day in DAYS:
        print(f"\n{'=' * 70}\n{day}\n{'=' * 70}")
        for hour in HOURS:
            t = pd.Timestamp(f"{day}T{hour:02d}:00:00", tz="UTC")
            print(f"\n{t:%H:%M} UTC  fetching...", flush=True)
            try:
                data = fetcher.fetch_calendar_spread_data(
                    FRONT_MONTH, PRODUCT, t, t + WINDOW
                )
            except Exception as exc:  # closed session, API hiccup, etc.
                print(f"  -- fetch failed: {exc}")
                continue

            print(_format_top_of_book(data.front_symbol, data.front))
            print(_format_top_of_book(data.back_symbol, data.back))
            print(_format_top_of_book(data.spread_symbol, data.spread))


if __name__ == "__main__":
    main()
