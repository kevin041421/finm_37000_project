"""MarketDataFetcher: interface + data contracts for calendar-spread market data (Issues #6/#7).

This module defines the *schema* that downstream modules (legging cost #8,
expected P&L engine #9, lead-lag analysis #11) can build against, plus the
``MarketDataFetcher`` class.

Both Issue #6 (Databento fetching + trading-day chunking:
``_fetch_single_instrument``, ``_fetch_single_instrument_trades``,
``_trading_days``) and Issue #7 (``next_contract_month``,
``_fetch_definitions``, ``_build_contract_spec``, backed by the contract
mapping file — product_specs.py / product_specs.json) are implemented.

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
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Union

import pandas as pd

import product_specs

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
    """
    if len(front_month) != 2:
        raise ValueError(
            f"front_month must be a month code plus a single year digit, "
            f"e.g. 'M6' — got {front_month!r}"
        )
    month_code, year_digit = front_month[0], front_month[1]
    if month_code not in cycle:
        raise ValueError(
            f"Month code {month_code!r} (from front_month={front_month!r}) "
            f"is not in cycle {cycle!r}"
        )
    if not year_digit.isdigit():
        raise ValueError(
            f"front_month's second character must be a digit — got {front_month!r}"
        )

    idx = cycle.index(month_code)
    year = int(year_digit)
    if idx + 1 < len(cycle):
        return f"{cycle[idx + 1]}{year}"
    # wrap to the first month of the cycle, next year
    return f"{cycle[0]}{(year + 1) % 10}"


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
    outright_tick_size: Decimal  # points, e.g. Decimal("0.25")
    spread_tick_size: Decimal    # points, e.g. Decimal("0.05")
    contract_multiplier: Decimal  # $ per point, e.g. Decimal("50")
    quote_scaling: Decimal = Decimal("1")  # raw-quote -> point conversion
    price_display_format: str = "decimal"  # "decimal" | "fractional"
    fractional_denominator: Optional[int] = None  # e.g. 32 for ZN, if fractional
    front_expiration: Optional[pd.Timestamp] = None
    back_expiration: Optional[pd.Timestamp] = None

    @property
    def outright_tick_value(self) -> Decimal:
        """Dollar value of one outright tick."""
        return self.outright_tick_size * self.contract_multiplier

    @property
    def spread_tick_value(self) -> Decimal:
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

    The private fetch/definition methods were the implementation surface
    for Issues #6/#7; both are now implemented:
    ``_fetch_single_instrument``, ``_fetch_single_instrument_trades``,
    ``_trading_days`` (#6), and ``_fetch_definitions``,
    ``_build_contract_spec``, ``next_contract_month`` (#7, backed by
    product_specs.py / product_specs.json).
    """

    def __init__(
        self,
        client: Optional["db.Historical"] = None,
        dataset: str = CME_DATASET,
        levels: int = 1,
        product_specs_path: Optional[Path] = None,
    ) -> None:
        """``levels`` defaults to 1 (MBP-1, top of book): sufficient for the
        P&L simulation (#9) and far cheaper to fetch and hold in memory than
        full depth. Pass a higher value only where depth is actually needed
        (e.g. the implied-vs-real book comparison, #15).

        Raises ``ValueError`` immediately if ``levels`` isn't a depth
        Databento actually supports (1 or 10) — fail at construction, not
        on the first fetch call.

        ``product_specs_path`` overrides where product_specs.json is read
        from (defaults to product_specs.DEFAULT_SPECS_PATH, i.e. co-located
        with product_specs.py). The file itself is loaded lazily on first
        use (not here in __init__) and cached — so constructing a fetcher
        for tests or for methods that don't need contract specs works fine
        even without product_specs.json present.
        """
        _schema_for_levels(levels)  # validates; raises early if invalid
        self._client = client
        self._dataset = dataset
        self._levels = levels
        self._product_specs_path = product_specs_path or product_specs.DEFAULT_SPECS_PATH
        self._product_specs_cache: Optional[dict] = None

    def _specs(self) -> dict:
        """Lazily load + cache product_specs.json. Not called from
        __init__ so that constructing a MarketDataFetcher never requires
        the file to exist unless a method that actually needs it is called."""
        if self._product_specs_cache is None:
            self._product_specs_cache = product_specs.load_product_specs(self._product_specs_path)
        return self._product_specs_cache

    @classmethod
    def from_databento_key(
        cls,
        dataset: str = CME_DATASET,
        levels: int = 1,
        product_specs_path: Optional[Path] = None,
        api_key_path: Optional[Path] = None,
    ) -> "MarketDataFetcher":
        """Construct a MarketDataFetcher with a real Databento client,
        reading the API key via util.get_databento_api_key() (defaults to
        ~/.databento_api_key) instead of requiring the caller to import
        databento and util themselves. util.py lives in src/, alongside
        this module, so the plain `from util import ...` import below
        resolves as a sibling-module import.
        """
        import databento as db_module
        from util import get_databento_api_key

        api_key = get_databento_api_key(api_key_path)
        client = db_module.Historical(api_key)
        return cls(client=client, dataset=dataset, levels=levels, product_specs_path=product_specs_path)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_calendar_spread_data(
        self,
        front_month: str,
        product: str,
        time_start: TimeLike,
        time_end: TimeLike,
        cycle: Optional[str] = None,
    ) -> CalendarSpreadData:
        """Fetch raw books and trades for the front month, back month, and spread.

        Args:
            front_month: Month-year code of the front leg, e.g. ``"M6"``.
            product: CME product code, e.g. ``"ES"``.
            time_start / time_end: Requested time window.
            cycle: Listed contract cycle used to derive the back month.
                Defaults to a product_specs.json lookup for ``product``
                (raises if the product isn't tracked there); pass
                explicitly (e.g. ``MONTH_CODES``) to override or to use a
                product not yet onboarded into product_specs.json.

        The back month is the next contract in ``cycle`` after
        ``front_month``. Each instrument's books and trades are fetched
        separately and returned on their raw event timestamps — no
        cross-instrument alignment is performed (consumers use as-of
        lookups where needed, e.g. book state at a trade's timestamp).
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
        cycle: Optional[str] = None,
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
        cycle: Optional[str] = None,
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
        cycle: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """Resolve (front, back, spread) symbols from a front month and product.

        ``cycle`` defaults to a product_specs.json lookup (see
        product_specs.get_product_spec) rather than a hardcoded quarterly
        default
        next_contract_month: a product not yet in product_specs.json raises
        ProductSpecError here rather than silently guessing quarterly and
        potentially computing a wrong back month for a monthly product.
        Pass ``cycle`` explicitly to bypass the lookup entirely (e.g. for a
        product not yet onboarded into product_specs.json).

        >>> MarketDataFetcher()._resolve_symbols("M6", "ES")
        ('ESM6', 'ESU6', 'ESM6-ESU6')
        """
        if cycle is None:
            cycle = product_specs.get_product_spec(product, specs=self._specs()).listing_cycle
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
        """Fetch Databento instrument definitions for the legs and the spread.

        Both bounds are day-aligned (midnight UTC). Databento rejects a
        sub-day-precise ``start`` when ``end`` is omitted — it forward-fills
        the definition state from ``start`` and can only do that from a day
        boundary — so ``pd.Timestamp.now()`` cannot be passed through
        unrounded (422 data_start_too_precise_to_forward_fill).
        """
        now = pd.Timestamp.now(tz="UTC")
        start = (now - pd.Timedelta(days=_DEFINITION_LOOKBACK_DAYS)).normalize()
        end = now.normalize()
        dbn = self._client.timeseries.get_range(
            dataset=self._dataset,
            schema="definition",
            symbols=[symbol1, symbol2, spread_symbol],
            start=start.isoformat(),
            end=end.isoformat(),
        )
        df = dbn.to_df()
        if df.empty:
            raise ValueError(
                f"No definition data returned for {[symbol1, symbol2, spread_symbol]} "
                f"in the last {_DEFINITION_LOOKBACK_DAYS} days"
            )
        return df

    def _build_contract_spec(
        self,
        symbol1: str,
        symbol2: str,
        spread_symbol: str,
        definitions: pd.DataFrame,
    ) -> CalendarSpreadContractSpec:
        """Map raw definitions (+ static JSON mapping file) to a ``CalendarSpreadContractSpec``.

        Static fields (tick sizes, multiplier, price format) come from
        product_specs.json via ``product_specs.get_product_spec`` — see
        that module's docstring. Expirations are computed from the
        product's termination_rule, then cross-checked against
        ``definitions``'s own ``expiration`` column where available; a
        mismatch raises rather than silently trusting the computed rule
        (see product_specs.py CAUTION #1 on why the computed date isn't
        fully trusted on its own).
        """
        product_code, front_month_code, front_year_digit = split_symbol(symbol1)
        _, back_month_code, back_year_digit = split_symbol(symbol2)

        spec = product_specs.get_product_spec(product_code, specs=self._specs())

        now = pd.Timestamp.now(tz="UTC")
        front_year = _full_year(front_year_digit, reference=now)
        back_year = _full_year(back_year_digit, reference=now)

        front_expiration = product_specs.compute_termination_date(spec, front_month_code, front_year)
        back_expiration = product_specs.compute_termination_date(spec, back_month_code, back_year)

        self._validate_expiration(definitions, symbol1, front_expiration)
        self._validate_expiration(definitions, symbol2, back_expiration)

        return CalendarSpreadContractSpec(
            product_code=product_code,
            front_symbol=symbol1,
            back_symbol=symbol2,
            spread_symbol=spread_symbol,
            outright_tick_size=spec.outright_tick_size,
            spread_tick_size=spec.spread_tick_size,
            contract_multiplier=spec.contract_multiplier,
            quote_scaling=spec.quote_scaling,
            price_display_format=spec.price_display_format,
            fractional_denominator=spec.fractional_denominator,
            front_expiration=front_expiration,
            back_expiration=back_expiration,
        )

    @staticmethod
    def _validate_expiration(
        definitions: pd.DataFrame,
        symbol: str,
        computed_expiration: pd.Timestamp,
        tolerance_days: int = 1,
    ) -> None:
        """Cross-check a rule-computed expiration against Databento's own
        reported expiration for `symbol`, if present in `definitions`.
        Raises ValueError on a mismatch beyond `tolerance_days` — the
        1-day default tolerance absorbs the no-holiday-calendar imprecision
        flagged in product_specs.py's CAUTION #1, not a sign the check is
        loose; a mismatch beyond that means the encoded rule itself is
        probably wrong and needs a human to re-check it against CME's spec
        sheet, not that this check should be widened further.

        Silently does nothing if `definitions` has no row for `symbol` or
        no `expiration` column — this validation is a backstop, not a hard
        requirement, since not every Databento definition schema version is
        guaranteed to expose it.
        """
        if "raw_symbol" not in definitions.columns or "expiration" not in definitions.columns:
            return
        rows = definitions[definitions["raw_symbol"] == symbol]
        if rows.empty:
            return
        reported = pd.Timestamp(rows.iloc[-1]["expiration"])
        if reported.tzinfo is None:
            reported = reported.tz_localize("UTC")
        else:
            reported = reported.tz_convert("UTC")
        computed = computed_expiration
        if computed.tzinfo is None:
            computed = computed.tz_localize("UTC")

        if abs((reported.normalize() - computed.normalize()).days) > tolerance_days:
            raise ValueError(
                f"{symbol}: computed expiration {computed.date()} from the "
                f"product's termination_rule differs from Databento's "
                f"reported expiration {reported.date()} by more than "
                f"{tolerance_days} day(s). The encoded termination_rule for "
                f"this product is likely wrong — verify against CME's "
                f"product spec sheet."
            )


def _ensure_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize a DataFrame's DatetimeIndex to tz-aware UTC, whether
    Databento handed it back naive or already tz-aware."""
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


#: How far back to search for an instrument's definition record. Definitions
#: don't change over a contract's life, so this just needs to comfortably
#: span "however long ago the contract could have been listed."
_DEFINITION_LOOKBACK_DAYS = 400


def _full_year(year_digit: str, reference: Optional[pd.Timestamp] = None) -> int:
    """Resolve a single CME year digit (e.g. '6') to a full year, assuming
    the nearest calendar year to `reference` (default: now) whose last
    digit matches.

    CAUTION: single-digit year symbols are only unambiguous within roughly
    a 5-year window — 'M6' means June 2026 today but will mean June 2036 in
    2031. This picks whichever full year is closest to `reference`; for a
    far-dated or historical contract outside that window, it will silently
    pick the wrong decade. This ambiguity isn't specific to this function —
    every symbol-year usage in this codebase (project_fetch_data.py, the
    test suite) shares it — but it matters most here, since a wrong year
    feeds directly into compute_termination_date with no other check to
    catch it before _validate_expiration compares against Databento (which
    would catch a wrong year as a large mismatch, but only after the fact).
    """
    reference = reference or pd.Timestamp.now(tz="UTC")
    digit = int(year_digit)
    candidates = [(reference.year // 10) * 10 + digit + offset * 10 for offset in (-1, 0, 1)]
    return min(candidates, key=lambda y: abs(y - reference.year))