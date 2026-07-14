"""MarketDataFetcher: interface + data contracts for calendar-spread market data (Issue #6/#7).

This module defines the *schema* that downstream modules (legging cost #8,
expected P&L engine #9, lead-lag analysis #11) can build against, plus a
skeleton ``MarketDataFetcher`` class.

Issue #6 (this pass) implements the actual Databento fetching + trading-day
chunking: ``_fetch_single_instrument``, ``_fetch_single_instrument_trades``,
``_trading_days``. Issue #7's methods (``next_contract_month``,
``_fetch_definitions``, ``_build_contract_spec``) remain ``NotImplementedError``
here — they depend on the contract mapping file (product_specs.json) landing
separately, and are out of scope for this change.

Data contracts
--------------
1. Book frame (one per instrument), the unit of market data:
   - index: ``ts_event`` — tz-aware UTC ``DatetimeIndex``, ascending.
   - columns (for each level ``lvl`` in ``0..levels-1``, Databento MBP naming):
       ``bid_px_{lvl:02d}``, ``ask_px_{lvl:02d}`` : float (NaN if level empty)
       ``bid_sz_{lvl:02d}``, ``ask_sz_{lvl:02d}`` : float
   Level 00 is top of book. Each book keeps its own raw event timestamps —
   the fetcher does not align the three books onto a common index. Consumers
   that need cross-instrument comparison should use as-of semantics (e.g.
   ``pd.merge_asof``): at time t, a book's state is its most recent row at
   or before t.

2. Trade frame (one per instrument), the unit of executed-trade data used to
   simulate fills:
   - index: ``ts_event`` — tz-aware UTC ``DatetimeIndex``, ascending.
   - columns:
       ``price`` : float — trade price in points
       ``size``  : float — traded quantity (contracts)
       ``side``  : str   — aggressor side, ``"A"`` (ask/seller-initiated),
                   ``"B"`` (bid/buyer-initiated) or ``"N"`` (unknown)
   Trades are events, not state. A simulation replays them in time order,
   reading each book's last state at the trade's timestamp (as-of lookup).

3. ``CalendarSpreadData`` — the three raw book frames (front leg, back leg,
   spread instrument), the three raw trade frames, plus their symbols.

4. ``CalendarSpreadContractSpec`` — the static CME parameters needed to turn
   points into dollars (tick sizes, multiplier, quote scaling, price format).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Union

import pandas as pd

if TYPE_CHECKING:  # avoid a hard databento dependency at import time
    import databento as db

TimeLike = Union[str, pd.Timestamp]

CME_DATASET = "GLBX.MDP3"
BOOK_INDEX_NAME = "ts_event"

#: CME month codes in calendar order.
MONTH_CODES = "FGHJKMNQUVXZ"

_SYMBOL_RE = re.compile(rf"^(?P<root>[A-Z0-9]+?)(?P<month>[{MONTH_CODES}])(?P<year>\d)$")

#: Databento only exposes two MBP book depths — there is no schema for an
#: arbitrary N in between. A `levels` value outside this set can't be
#: satisfied by any real Databento schema, so it's validated at construction
#: time rather than failing later inside a fetch call.
VALID_BOOK_LEVELS = {1, 10}


#: Columns of a trade frame (index: ``ts_event``).
TRADE_COLUMNS = ["price", "size", "side"]


def book_columns(levels: int = 10) -> list[str]:
    """Column names of a book frame with ``levels`` price levels."""
    cols: list[str] = []
    for lvl in range(levels):
        cols += [
            f"bid_px_{lvl:02d}",
            f"ask_px_{lvl:02d}",
            f"bid_sz_{lvl:02d}",
            f"ask_sz_{lvl:02d}",
        ]
    return cols


def _normalize_time(value: TimeLike) -> pd.Timestamp:
    """Coerce a TimeLike (str or Timestamp) into a tz-aware UTC Timestamp.

    Idempotent on an already-aware Timestamp; localizes a naive Timestamp
    or a bare date/datetime string to UTC rather than guessing a different
    zone. Centralized here so every call site normalizes identically —
    the existing project_fetch_data.py script builds its own one-off
    UTC-aware Timestamp inline; this is the same pattern, shared.
    """
    ts = pd.Timestamp(value)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _schema_for_levels(levels: int) -> str:
    """Map a book depth to the (only two) real Databento MBP schema names."""
    if levels not in VALID_BOOK_LEVELS:
        raise ValueError(
            f"levels must be one of {sorted(VALID_BOOK_LEVELS)} — Databento "
            f"only provides mbp-1 and mbp-10 book schemas, got {levels!r}"
        )
    return f"mbp-{levels}"


#: Default contract cycle: quarterly (equity index, FX, rates).
#: Products listing every month (e.g. CL, NG) should pass cycle=MONTH_CODES.
QUARTERLY_CYCLE = "HMUZ"


def next_contract_month(front_month: str, cycle: str = QUARTERLY_CYCLE) -> str:
    """Return the contract month following ``front_month`` in the product's listed cycle.

    Used to derive the back leg of a calendar spread: the back month is the
    *next listed contract* after the front month, and which contract that is
    depends on the product's listing cycle. Equity index futures (ES, NQ, ...)
    list quarterly (H=Mar, M=Jun, U=Sep, Z=Dec), so the contract after M6 is
    U6. Energy products (CL, NG, ...) list every month, so the contract after
    F6 (Jan) is G6 (Feb) — for those, pass ``cycle=MONTH_CODES``.

    Args:
        front_month: Month-year code of the front leg, e.g. ``"M6"`` = June 2026.
        cycle: The product's listed contract months, in calendar order, as a
            string of CME month codes. Defaults to the quarterly cycle
            ``"HMUZ"``. Raises ``ValueError`` if ``front_month``'s month code
            is not in ``cycle`` — so calling with a monthly product's contract
            (e.g. ``"F6"``) under the default fails loudly instead of
            silently picking a wrong back month.

    Must handle year rollover: the contract after Z6 is H7 (quarterly) or F7
    (monthly). Must raise ``ValueError`` for a month code not in ``cycle``.

    Expected behavior once implemented::

        next_contract_month("M6")                 -> "U6"
        next_contract_month("Z6")                 -> "H7"
        next_contract_month("F6", cycle=MONTH_CODES) -> "G6"

    TODO(#7): the cycle should not be caller-supplied — look it up per
    product from the contract mapping file (e.g. ``"ES": "HMUZ"``,
    ``"CL": MONTH_CODES``) so the fetcher identifies the right back month
    from the product code alone.
    """
    raise NotImplementedError  # Issue #7


def split_symbol(symbol: str) -> tuple[str, str, str]:
    """Split a CME outright symbol into (root, month_code, year_digit).

    >>> split_symbol("ESM6")
    ('ES', 'M', '6')
    """
    m = _SYMBOL_RE.match(symbol)
    if m is None:
        raise ValueError(f"Cannot parse CME outright symbol: {symbol!r}")
    return m.group("root"), m.group("month"), m.group("year")


@dataclass(frozen=True)
class CalendarSpreadData:
    """Order books and trades for the two legs and the spread instrument.

    All frames keep their own raw event timestamps; the three books are not
    aligned onto a common index. To read a book's state at an arbitrary time
    (e.g. the moment a trade happened), use as-of semantics such as
    ``pd.merge_asof``.
    """

    front_symbol: str
    back_symbol: str
    spread_symbol: str
    front: pd.DataFrame
    back: pd.DataFrame
    spread: pd.DataFrame
    front_trades: pd.DataFrame
    back_trades: pd.DataFrame
    spread_trades: pd.DataFrame


@dataclass(frozen=True)
class CalendarSpreadContractSpec:
    """Static CME contract parameters for a calendar spread (Issue #7 schema).

    ``contract_multiplier`` is dollars per index point (ES: 50), so
    tick value in dollars = tick size × multiplier.
    """

    product_code: str            # e.g. "ES"
    front_symbol: str            # e.g. "ESM6"
    back_symbol: str             # e.g. "ESU6"
    spread_symbol: str           # e.g. "ESM6-ESU6"
    outright_tick_size: float    # points, e.g. 0.25
    spread_tick_size: float      # points, e.g. 0.05
    contract_multiplier: float   # $ per point, e.g. 50.0
    quote_scaling: float = 1.0   # raw-quote -> point conversion (e.g. cents quotes)
    price_display_format: str = "decimal"  # "decimal" | "fractional"
    fractional_denominator: Optional[int] = None  # e.g. 32 for ZN, if fractional
    front_expiration: Optional[pd.Timestamp] = None
    back_expiration: Optional[pd.Timestamp] = None

    @property
    def outright_tick_value(self) -> float:
        """Dollar value of one outright tick."""
        return self.outright_tick_size * self.contract_multiplier

    @property
    def spread_tick_value(self) -> float:
        """Dollar value of one spread tick."""
        return self.spread_tick_size * self.contract_multiplier


class MarketDataFetcher:
    """Fetches calendar-spread market data (books + trades) from Databento.

    Public interface (stable — downstream modules code against this):
      - ``fetch_calendar_spread_data(front_month, product, time_start, time_end)``
        — whole window at once; fine for short windows.
      - ``iter_calendar_spread_data(front_month, product, time_start, time_end)``
        — generator yielding one day at a time; use for long windows.
      - ``fetch_calendar_spread_contract_specification(front_month, product)``

    The private fetch/definition methods are the implementation surface
    for Issues #6/#7. As of this change, the #6 methods
    (``_fetch_single_instrument``, ``_fetch_single_instrument_trades``,
    ``_trading_days``) are implemented; the #7 methods
    (``_fetch_definitions``, ``_build_contract_spec``) and
    ``next_contract_month`` still raise ``NotImplementedError`` pending the
    contract mapping file.
    """

    def __init__(
        self,
        client: Optional["db.Historical"] = None,
        dataset: str = CME_DATASET,
        levels: int = 1,
    ) -> None:
        """``levels`` defaults to 1 (MBP-1, top of book): sufficient for the
        P&L simulation (#9) and far cheaper to fetch and hold in memory than
        full depth. Pass a higher value only where depth is actually needed
        (e.g. the implied-vs-real book comparison, #15).

        Raises ``ValueError`` immediately if ``levels`` isn't a depth
        Databento actually supports (1 or 10) — fail at construction, not
        on the first fetch call.
        """
        _schema_for_levels(levels)  # validates; raises early if invalid
        self._client = client
        self._dataset = dataset
        self._levels = levels

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_calendar_spread_data(
        self,
        front_month: str,
        product: str,
        time_start: TimeLike,
        time_end: TimeLike,
        cycle: str = QUARTERLY_CYCLE,
    ) -> CalendarSpreadData:
        """Fetch raw books and trades for the front month, back month, and spread.

        Args:
            front_month: Month-year code of the front leg, e.g. ``"M6"``.
            product: CME product code, e.g. ``"ES"``.
            time_start / time_end: Requested time window.
            cycle: Listed contract cycle used to derive the back month
                (default quarterly ``"HMUZ"``; use ``MONTH_CODES`` for
                monthly products like CL).

        The back month is the next contract in ``cycle`` after
        ``front_month``. Each instrument's books and trades are fetched
        separately and returned on their raw event timestamps — no
        cross-instrument alignment is performed (consumers use as-of
        lookups where needed, e.g. book state at a trade's timestamp).

        NOTE: still raises NotImplementedError via ``_resolve_symbols`` ->
        ``next_contract_month``, which is Issue #7 scope (depends on the
        contract mapping file). The #6 fetch methods this calls into
        (``_fetch_single_instrument`` etc.) are implemented and independently
        testable with an already-resolved symbol in the meantime.
        """
        front_symbol, back_symbol, spread_symbol = self._resolve_symbols(
            front_month, product, cycle
        )

        front = self._fetch_single_instrument(front_symbol, time_start, time_end)
        back = self._fetch_single_instrument(back_symbol, time_start, time_end)
        spread = self._fetch_single_instrument(spread_symbol, time_start, time_end)

        front_trades = self._fetch_single_instrument_trades(front_symbol, time_start, time_end)
        back_trades = self._fetch_single_instrument_trades(back_symbol, time_start, time_end)
        spread_trades = self._fetch_single_instrument_trades(spread_symbol, time_start, time_end)

        return CalendarSpreadData(
            front_symbol=front_symbol,
            back_symbol=back_symbol,
            spread_symbol=spread_symbol,
            front=front,
            back=back,
            spread=spread,
            front_trades=front_trades,
            back_trades=back_trades,
            spread_trades=spread_trades,
        )

    def iter_calendar_spread_data(
        self,
        front_month: str,
        product: str,
        time_start: TimeLike,
        time_end: TimeLike,
        cycle: str = QUARTERLY_CYCLE,
    ) -> Iterator[CalendarSpreadData]:
        """Lazily yield one ``CalendarSpreadData`` per trading day in the window.

        Memory-bounded alternative to ``fetch_calendar_spread_data`` for long
        windows (e.g. 10 days): only one day of data is in memory at a time.
        Consumers process each day and keep only the small daily results
        (P&L, fill stats), aggregating at the end.
        """
        for day_start, day_end in self._trading_days(time_start, time_end):
            yield self.fetch_calendar_spread_data(
                front_month, product, day_start, day_end, cycle
            )

    def fetch_calendar_spread_contract_specification(
        self,
        front_month: str,
        product: str,
        cycle: str = QUARTERLY_CYCLE,
    ) -> CalendarSpreadContractSpec:
        """Fetch the static contract parameters for the legs and their spread."""
        front_symbol, back_symbol, spread_symbol = self._resolve_symbols(
            front_month, product, cycle
        )
        definitions = self._fetch_definitions(front_symbol, back_symbol, spread_symbol)
        return self._build_contract_spec(
            front_symbol, back_symbol, spread_symbol, definitions
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _resolve_symbols(
        self,
        front_month: str,
        product: str,
        cycle: str = QUARTERLY_CYCLE,
    ) -> tuple[str, str, str]:
        """Resolve (front, back, spread) symbols from a front month and product.

        >>> MarketDataFetcher()._resolve_symbols("M6", "ES")
        ('ESM6', 'ESU6', 'ESM6-ESU6')
        """
        back_month = next_contract_month(front_month, cycle)
        front_symbol = f"{product}{front_month}"
        back_symbol = f"{product}{back_month}"
        spread_symbol = self._build_spread_symbol(front_symbol, back_symbol)
        return front_symbol, back_symbol, spread_symbol

    def _build_spread_symbol(self, symbol1: str, symbol2: str) -> str:
        """Build and validate the CME calendar spread symbol, e.g. ``ESM6-ESU6``.

        Validates that both legs share the same product root and that
        ``symbol1`` expires before ``symbol2``.
        """
        root1, month1, year1 = split_symbol(symbol1)
        root2, month2, year2 = split_symbol(symbol2)
        if root1 != root2:
            raise ValueError(
                f"Legs must share a product root: {symbol1!r} vs {symbol2!r}"
            )
        order1 = (int(year1), MONTH_CODES.index(month1))
        order2 = (int(year2), MONTH_CODES.index(month2))
        if order1 >= order2:
            raise ValueError(
                f"Front leg must expire before back leg: {symbol1!r} !< {symbol2!r}"
            )
        return f"{symbol1}-{symbol2}"

    def _fetch_single_instrument(
        self,
        symbol: str,
        time_start: TimeLike,
        time_end: TimeLike,
    ) -> pd.DataFrame:
        """Fetch the MBP book time series for one instrument.

        Returns a book frame (see module docstring): ``ts_event`` UTC index,
        ``bid/ask _px/_sz`` columns for ``self._levels`` levels.
        """
        start = _normalize_time(time_start)
        end = _normalize_time(time_end)
        schema = _schema_for_levels(self._levels)
        cols = book_columns(self._levels)

        dbn = self._client.timeseries.get_range(
            dataset=self._dataset,
            schema=schema,
            symbols=[symbol],
            start=start.isoformat(),
            end=end.isoformat(),
        )
        df = dbn.to_df()

        if df.empty:
            empty = pd.DataFrame(columns=cols)
            empty.index = pd.DatetimeIndex([], tz="UTC", name=BOOK_INDEX_NAME)
            return empty

        df = df[cols].copy()
        df.index = _ensure_utc_index(df.index)
        df.index.name = BOOK_INDEX_NAME
        return df.sort_index()

    def _fetch_single_instrument_trades(
        self,
        symbol: str,
        time_start: TimeLike,
        time_end: TimeLike,
    ) -> pd.DataFrame:
        """Fetch executed trades for one instrument (Databento ``trades`` schema).

        Returns a trade frame (see module docstring): ``ts_event`` UTC index,
        ``price``, ``size``, ``side`` columns, in ascending time order.
        """
        start = _normalize_time(time_start)
        end = _normalize_time(time_end)

        dbn = self._client.timeseries.get_range(
            dataset=self._dataset,
            schema="trades",
            symbols=[symbol],
            start=start.isoformat(),
            end=end.isoformat(),
        )
        df = dbn.to_df()

        if df.empty:
            empty = pd.DataFrame(columns=TRADE_COLUMNS)
            empty.index = pd.DatetimeIndex([], tz="UTC", name=BOOK_INDEX_NAME)
            return empty

        df = df[TRADE_COLUMNS].copy()
        df.index = _ensure_utc_index(df.index)
        df.index.name = BOOK_INDEX_NAME
        return df.sort_index()

    def _trading_days(
        self,
        time_start: TimeLike,
        time_end: TimeLike,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Split a time window into per-day (start, end) sub-windows

        Does NOT account for CME holidays — on an actual exchange holiday
        this will still yield a window for that day, and the resulting fetch
        will just come back empty (handled the same way as any other
        thin/quiet period — see `_fetch_single_instrument`'s empty-result
        handling). That's a deliberate simplification: this method only
        needs to bound memory usage ("one day of data at a time" per
        `iter_calendar_spread_data`'s docstring), not model real CME trading
        sessions — so it doesn't pull in a market-calendar dependency to
        shave off a handful of wasted, harmless empty fetches per year.
        """
        start = _normalize_time(time_start)
        end = _normalize_time(time_end)
        if start >= end:
            raise ValueError(f"time_start ({start}) must be before time_end ({end})")

        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        day = pd.Timestamp(start.date(), tz="UTC")
        while day < end:
            next_day = day + pd.Timedelta(days=1)
            if day.weekday() < 5:  # Monday=0 .. Friday=4; skip Saturday/Sunday
                day_start = max(start, day)
                day_end = min(end, next_day)
                if day_start < day_end:
                    windows.append((day_start, day_end))
            day = next_day

        return windows

    def _fetch_definitions(
        self,
        symbol1: str,
        symbol2: str,
        spread_symbol: str,
    ) -> pd.DataFrame:
        """Fetch Databento instrument definitions for the legs and the spread."""
        raise NotImplementedError  # Issue #7

    def _build_contract_spec(
        self,
        symbol1: str,
        symbol2: str,
        spread_symbol: str,
        definitions: pd.DataFrame,
    ) -> CalendarSpreadContractSpec:
        """Map raw definitions (+ static JSON mapping file) to a ``CalendarSpreadContractSpec``."""
        raise NotImplementedError  # Issue #7


def _ensure_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize a DataFrame's DatetimeIndex to tz-aware UTC, whether
    Databento handed it back naive or already tz-aware."""
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")