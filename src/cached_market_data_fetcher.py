"""CachedMarketDataFetcher: disk-cached wrapper around the real ``MarketDataFetcher``.

Replaces the caching half of the TEMP ``HardcodedESFetcher`` in
``demo_print_books.py`` now that Issues #6/#7 have landed. The symbol
hardcoding is gone — symbols and contract specs come from the real fetcher —
but the on-disk cache is kept, for two reasons:

- Databento historical data is metered. Re-running the P&L pipeline while
  tuning model parameters (``p``, ``max_position``, seeds) should not re-buy
  the same bytes on every run.
- Cached runs are offline and deterministic, so results are reproducible
  without an API key once the cache is warm.

Everything cached here is *historical* market data for a closed window, so it
is immutable — there is no staleness/invalidation concern. Delete
``data_cache/`` to force a refetch.

Cache layout (pickled DataFrames / spec objects)::

    data_cache/book_ESM6_mbp-1_20260609T160000_20260609T160100.pkl
    data_cache/trades_ESM6_20260609T160000_20260609T160100.pkl
    data_cache/spec_ES_M6.pkl
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from market_data_fetcher import (
    CalendarSpreadContractSpec,
    MarketDataFetcher,
    TimeLike,
    _schema_for_levels,
)

#: Default cache location: ``<repo>/data_cache`` (git-ignored), the same
#: directory the TEMP demo script used.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def _stamp(value: TimeLike) -> str:
    """Filename-safe UTC timestamp, e.g. ``20260609T160000``."""
    ts = pd.Timestamp(value)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%dT%H%M%S")


class CachedMarketDataFetcher(MarketDataFetcher):
    """``MarketDataFetcher`` that memoizes fetch results to disk.

    Overrides only the three methods that actually reach out to Databento;
    all symbol resolution, chunking and spec-building logic is inherited
    unchanged from #6/#7.
    """

    def __init__(
        self,
        *args,
        cache_dir: Optional[Path] = None,
        retry_attempts: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self._retry_attempts = retry_attempts

    # ------------------------------------------------------------------ #
    # Cache plumbing
    # ------------------------------------------------------------------ #

    def _with_retry(self, compute: Callable[[], object]) -> object:
        """Run ``compute``, retrying with linear backoff.

        Databento's gateway intermittently 504s on longer windows; a retry
        almost always succeeds. Only cache misses reach here, so a warm
        cache never sleeps. Applied around the whole fetch (rather than the
        raw ``get_range``) so books, trades and definitions all get it.
        """
        for attempt in range(1, self._retry_attempts + 1):
            try:
                return compute()
            except Exception:
                if attempt == self._retry_attempts:
                    raise
                time.sleep(2 * attempt)  # 2s, then 4s, ...
        raise AssertionError("unreachable")

    def _cached(self, name: str, compute: Callable[[], object]) -> object:
        """Return ``name`` from the cache, else run ``compute`` and store it.

        Writes via a temporary file and an atomic rename so an interrupted
        run (or a Databento timeout mid-write) can never leave a truncated
        pickle behind that a later run would happily load.
        """
        path = self._cache_dir / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                return pickle.load(fh)

        value = self._with_retry(compute)

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh)
        tmp.replace(path)
        return value

    # ------------------------------------------------------------------ #
    # Cached overrides
    # ------------------------------------------------------------------ #

    def _fetch_single_instrument(
        self,
        symbol: str,
        time_start: TimeLike,
        time_end: TimeLike,
    ) -> pd.DataFrame:
        # The book depth is part of the key: an mbp-1 frame has different
        # columns than an mbp-10 one, so they must not share a cache entry.
        schema = _schema_for_levels(self._levels)
        name = f"book_{symbol}_{schema}_{_stamp(time_start)}_{_stamp(time_end)}"
        return self._cached(
            name, lambda: super(CachedMarketDataFetcher, self)._fetch_single_instrument(
                symbol, time_start, time_end
            )
        )

    def _fetch_single_instrument_trades(
        self,
        symbol: str,
        time_start: TimeLike,
        time_end: TimeLike,
    ) -> pd.DataFrame:
        name = f"trades_{symbol}_{_stamp(time_start)}_{_stamp(time_end)}"
        return self._cached(
            name,
            lambda: super(
                CachedMarketDataFetcher, self
            )._fetch_single_instrument_trades(symbol, time_start, time_end),
        )

    def fetch_calendar_spread_contract_specification(
        self,
        front_month: str,
        product: str,
        cycle: Optional[str] = None,
    ) -> CalendarSpreadContractSpec:
        """Cached contract spec.

        Cached separately from the book/trade frames because it has no time
        window in its key: a listed contract's static parameters (tick sizes,
        multiplier, expiration) do not change over its life, so one lookup
        per (product, front month) is enough.
        """
        name = f"spec_{product}_{front_month}"
        return self._cached(
            name,
            lambda: super(
                CachedMarketDataFetcher, self
            ).fetch_calendar_spread_contract_specification(front_month, product, cycle),
        )
