"""TEMP demo script: print ES front/back/spread order books every hour.

Issues #6/#7 are not implemented yet, so this script hardcodes what they
will eventually provide:
- symbols are fixed to ES June/Sep 2026 (ESM6, ESU6, ESM6-ESU6);
- ``_fetch_single_instrument`` / ``_fetch_single_instrument_trades`` are
  implemented inline with direct Databento calls (mbp-1 / trades).

Delete this hardcoded fetcher once #6/#7 land.

Run:
    uv run src/demo_print_books.py

Requires ~/.databento_api_key (see src/util.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import databento as db
import pandas as pd

from market_data_fetcher import (
    BOOK_INDEX_NAME,
    TRADE_COLUMNS,
    MarketDataFetcher,
    book_columns,
)
from util import get_databento_api_key

# Two days to demo (ESM6 and ESU6 both active, per the README snapshot week)
DAYS = ["2026-06-09", "2026-06-10"]
HOURS = range(24)
# We only need the book state at each hour, so fetch a short window and
# take the first message per instrument as the snapshot.
WINDOW = pd.Timedelta(seconds=60)

# Temp on-disk cache: re-runs read from here instead of hitting the API.
# (The real #6 implementation should do this properly, with parquet.)
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=columns,
        index=pd.DatetimeIndex([], tz="UTC", name=BOOK_INDEX_NAME),
    )


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the frame is indexed by ts_event (UTC)."""
    if df.index.name != BOOK_INDEX_NAME and BOOK_INDEX_NAME in df.columns:
        df = df.set_index(BOOK_INDEX_NAME)
    df.index.name = BOOK_INDEX_NAME
    return df


class HardcodedESFetcher(MarketDataFetcher):
    """TEMP stand-in for Issues #6/#7, hardcoded to the ES Jun/Sep 2026 pair."""

    def _resolve_symbols(self, front_month, product, cycle=None):
        return "ESM6", "ESU6", "ESM6-ESU6"

    def _get_range_with_retry(self, schema, symbol, time_start, time_end, attempts=3):
        """Databento's gateway occasionally 504s; retry with backoff."""
        for attempt in range(1, attempts + 1):
            try:
                return self._client.timeseries.get_range(
                    dataset=self._dataset,
                    schema=schema,
                    symbols=[symbol],
                    start=pd.Timestamp(time_start).isoformat(),
                    end=pd.Timestamp(time_end).isoformat(),
                )
            except Exception:
                if attempt == attempts:
                    raise
                time.sleep(2 * attempt)  # 2s, then 4s

    def _cache_path(self, schema, symbol, time_start, time_end) -> Path:
        start = pd.Timestamp(time_start).strftime("%Y%m%dT%H%M%S")
        end = pd.Timestamp(time_end).strftime("%Y%m%dT%H%M%S")
        return CACHE_DIR / f"{symbol}_{schema}_{start}_{end}.pkl"

    def _fetch_cached(self, schema, symbol, time_start, time_end, columns):
        """Read from the disk cache if present, otherwise fetch and store."""
        path = self._cache_path(schema, symbol, time_start, time_end)
        if path.exists():
            return pd.read_pickle(path)

        dbn = self._get_range_with_retry(schema, symbol, time_start, time_end)
        df = dbn.to_df()
        if df.empty:
            df = _empty_frame(columns)
        else:
            df = _normalize_index(df)[columns]

        CACHE_DIR.mkdir(exist_ok=True)
        df.to_pickle(path)
        return df

    def _fetch_single_instrument(self, symbol, time_start, time_end):
        return self._fetch_cached(
            "mbp-1", symbol, time_start, time_end, book_columns(self._levels)
        )

    def _fetch_single_instrument_trades(self, symbol, time_start, time_end):
        return self._fetch_cached(
            "trades", symbol, time_start, time_end, TRADE_COLUMNS
        )


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
    client = db.Historical(get_databento_api_key())
    fetcher = HardcodedESFetcher(client=client, levels=1)

    for day in DAYS:
        print(f"\n{'=' * 70}\n{day}\n{'=' * 70}")
        for hour in HOURS:
            t = pd.Timestamp(f"{day}T{hour:02d}:00:00", tz="UTC")
            print(f"\n{t:%H:%M} UTC  fetching...", flush=True)
            try:
                data = fetcher.fetch_calendar_spread_data("M6", "ES", t, t + WINDOW)
            except Exception as exc:  # closed session, API hiccup, etc.
                print(f"  -- fetch failed: {exc}")
                continue

            print(_format_top_of_book(data.front_symbol, data.front))
            print(_format_top_of_book(data.back_symbol, data.back))
            print(_format_top_of_book(data.spread_symbol, data.spread))


if __name__ == "__main__":
    main()
