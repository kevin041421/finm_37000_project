"""Unit tests for MarketDataFetcher (Issues #6 and #7 implemented).

- pure helpers (symbol parsing, spread symbol building, schema, spec math)
  are tested directly;
- the public methods are tested as *composition* with mocked privates,
  pinning down the call contract independent of implementation;
- Issue #6's methods (_fetch_single_instrument, _fetch_single_instrument_trades,
  _trading_days) have real coverage against a mocked Databento client;
- Issue #7's methods (next_contract_month, _fetch_definitions,
  _build_contract_spec) also have real coverage now — the tests that were
  previously xfail (pending Issue #7) have had that mark removed per the
  original file's own instruction to do so once implemented.
"""

import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from market_data_fetcher import (
    BOOK_INDEX_NAME,
    MONTH_CODES,
    QUARTERLY_CYCLE,
    TRADE_COLUMNS,
    CalendarSpreadContractSpec,
    CalendarSpreadData,
    MarketDataFetcher,
    _normalize_time,
    book_columns,
    next_contract_month,
    split_symbol,
)

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

LEVELS = 2  # small books are enough for unit tests — NOTE: this is a book
# *shape* convenience for test data only (make_book/book_columns are pure
# functions, independent of any MarketDataFetcher instance). It is NOT a
# valid `levels` value to construct MarketDataFetcher with: Databento only
# exposes mbp-1 and mbp-10 as real schemas, and __init__ validates against
# exactly that set (Issue #6). See test_init_rejects_non_databento_levels
# below. The `fetcher` fixture therefore uses levels=1, not LEVELS.


def make_book(timestamps, bid_px, ask_px, bid_sz=10.0, ask_sz=10.0, levels=LEVELS):
    """Build a schema-conforming book frame with identical prices offset per level."""
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True), name=BOOK_INDEX_NAME)
    data = {}
    for lvl in range(levels):
        data[f"bid_px_{lvl:02d}"] = [px - 0.25 * lvl for px in bid_px]
        data[f"ask_px_{lvl:02d}"] = [px + 0.25 * lvl for px in ask_px]
        data[f"bid_sz_{lvl:02d}"] = bid_sz
        data[f"ask_sz_{lvl:02d}"] = ask_sz
    return pd.DataFrame(data, index=idx)[book_columns(levels)]


def make_trades(timestamps, prices, sizes=None, sides=None):
    """Build a schema-conforming trade frame."""
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True), name=BOOK_INDEX_NAME)
    n = len(prices)
    return pd.DataFrame(
        {
            "price": prices,
            "size": sizes if sizes is not None else [1.0] * n,
            "side": sides if sides is not None else ["B"] * n,
        },
        index=idx,
    )


@pytest.fixture
def es_books():
    """README ES snapshot as one-row book frames (front, back, spread)."""
    ts = ["2026-06-10T16:00:00Z"]
    return {
        "front": make_book(ts, bid_px=[7340.00], ask_px=[7340.25]),
        "back": make_book(ts, bid_px=[7401.00], ask_px=[7401.25]),
        "spread": make_book(ts, bid_px=[60.70], ask_px=[60.75]),
    }


@pytest.fixture
def es_trades():
    """Sample trade frames: a buyer lifts the spread ask, a seller hits the front bid."""
    return {
        "ESM6": make_trades(["2026-06-10T16:00:00.5Z"], [7340.00], [5.0], ["A"]),
        "ESU6": make_trades(["2026-06-10T16:00:00.7Z"], [7401.25], [2.0], ["B"]),
        "ESM6-ESU6": make_trades(["2026-06-10T16:00:00.9Z"], [60.75], [10.0], ["B"]),
    }


@pytest.fixture
def es_spec():
    return CalendarSpreadContractSpec(
        product_code="ES",
        front_symbol="ESM6",
        back_symbol="ESU6",
        spread_symbol="ESM6-ESU6",
        outright_tick_size=Decimal("0.25"),
        spread_tick_size=Decimal("0.05"),
        contract_multiplier=Decimal("50"),
    )


@pytest.fixture
def fetcher():
    # levels=1, not LEVELS(=2) — see the LEVELS comment above. This fixture
    # is used by composition tests that mock out the private fetch methods
    # entirely, so the specific valid level chosen here doesn't affect them.
    return MarketDataFetcher(client=None, levels=1)


def _client_returning(df: pd.DataFrame) -> MagicMock:
    """A mocked Databento Historical client whose timeseries.get_range(...).to_df()
    returns a fixed DataFrame, for testing the Issue #6 fetch methods directly
    (as opposed to the composition tests below, which mock the fetch methods
    themselves and never touch a client)."""
    client = MagicMock()
    result = MagicMock()
    result.to_df.return_value = df
    client.timeseries.get_range.return_value = result
    return client


# --------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------- #

def test_book_columns_names_and_count():
    cols = book_columns(2)
    assert cols == [
        "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
        "bid_px_01", "ask_px_01", "bid_sz_01", "ask_sz_01",
    ]
    assert len(book_columns(10)) == 40


def test_make_book_conforms_to_schema(es_books):
    front = es_books["front"]
    assert front.index.name == BOOK_INDEX_NAME
    assert str(front.index.tz) == "UTC"
    assert list(front.columns) == book_columns(LEVELS)
    assert front["bid_px_00"].iloc[0] == 7340.00
    assert front["ask_px_00"].iloc[0] == 7340.25


def test_spec_tick_values(es_spec):
    # ES: one outright tick (0.25 pt) = $12.50, one spread tick (0.05 pt) = $2.50
    # Exact equality, not approx — Decimal arithmetic has no float rounding
    # error to approximate away; that precision is the entire point.
    assert es_spec.outright_tick_value == Decimal("12.50")
    assert es_spec.spread_tick_value == Decimal("2.50")


def test_make_trades_conforms_to_schema(es_trades):
    trades = es_trades["ESM6-ESU6"]
    assert trades.index.name == BOOK_INDEX_NAME
    assert str(trades.index.tz) == "UTC"
    assert list(trades.columns) == TRADE_COLUMNS
    assert trades["price"].iloc[0] == 60.75
    assert trades["side"].iloc[0] == "B"


def test_calendar_spread_data_holds_raw_frames(es_books, es_trades):
    data = CalendarSpreadData(
        front_symbol="ESM6", back_symbol="ESU6", spread_symbol="ESM6-ESU6",
        front=es_books["front"], back=es_books["back"], spread=es_books["spread"],
        front_trades=es_trades["ESM6"],
        back_trades=es_trades["ESU6"],
        spread_trades=es_trades["ESM6-ESU6"],
    )
    # frames keep their own raw timestamps — no alignment is imposed
    assert data.front["bid_px_00"].iloc[0] == 7340.00
    assert data.spread_trades.index[0] > data.front.index[0]


# --------------------------------------------------------------------- #
# Symbol helpers
# --------------------------------------------------------------------- #

def test_split_symbol():
    assert split_symbol("ESM6") == ("ES", "M", "6")
    assert split_symbol("ZNH7") == ("ZN", "H", "7")
    assert split_symbol("6EU6") == ("6E", "U", "6")


def test_split_symbol_rejects_garbage():
    with pytest.raises(ValueError):
        split_symbol("NOT_A_SYMBOL")


# next_contract_month, implemented (Issue #7).

def test_next_contract_quarterly():
    assert next_contract_month("H6") == "M6"
    assert next_contract_month("M6") == "U6"
    assert next_contract_month("U6") == "Z6"
    assert next_contract_month("Z6") == "H7"  # year rollover


def test_next_contract_monthly_cycle():
    assert next_contract_month("F6", cycle=MONTH_CODES) == "G6"
    assert next_contract_month("Z6", cycle=MONTH_CODES) == "F7"


def test_next_contract_rejects_bad_input():
    with pytest.raises(ValueError):
        next_contract_month("A6")  # 'A' not a month code in the quarterly cycle
    with pytest.raises(ValueError):
        next_contract_month("M")  # missing year digit


def test_resolve_symbols(fetcher):
    # ES: cycle auto-resolved from product_specs.json (quarterly)
    assert fetcher._resolve_symbols("M6", "ES") == ("ESM6", "ESU6", "ESM6-ESU6")
    # 6E: also auto-resolved from product_specs.json (quarterly)
    assert fetcher._resolve_symbols("Z6", "6E") == ("6EZ6", "6EH7", "6EZ6-6EH7")
    # CL: auto-resolved from product_specs.json (monthly) — no explicit
    # cycle needed anymore; this used to require cycle=MONTH_CODES or it
    # would silently (and wrongly) default to quarterly.
    assert fetcher._resolve_symbols("F6", "CL") == ("CLF6", "CLG6", "CLF6-CLG6")
    # explicit cycle still overrides the product_specs.json lookup
    assert fetcher._resolve_symbols("F6", "CL", cycle=MONTH_CODES) == (
        "CLF6", "CLG6", "CLF6-CLG6"
    )


def test_resolve_symbols_untracked_product_raises(fetcher):
    # "NQ" has no product_specs.json entry — raises rather than silently
    # guessing quarterly, unless the caller supplies cycle explicitly.
    with pytest.raises(Exception):
        fetcher._resolve_symbols("Z6", "NQ")
    # explicit cycle bypasses the lookup entirely, so an untracked product
    # still works if the caller supplies it.
    assert fetcher._resolve_symbols("Z6", "NQ", cycle=QUARTERLY_CYCLE) == (
        "NQZ6", "NQH7", "NQZ6-NQH7"
    )


def test_build_spread_symbol(fetcher):
    assert fetcher._build_spread_symbol("ESM6", "ESU6") == "ESM6-ESU6"


def test_build_spread_symbol_rejects_mixed_roots(fetcher):
    with pytest.raises(ValueError, match="root"):
        fetcher._build_spread_symbol("ESM6", "NQU6")


def test_build_spread_symbol_rejects_wrong_leg_order(fetcher):
    with pytest.raises(ValueError, match="before"):
        fetcher._build_spread_symbol("ESU6", "ESM6")  # back leg first
    with pytest.raises(ValueError, match="before"):
        fetcher._build_spread_symbol("ESM6", "ESM6")  # same contract


def test_build_spread_symbol_handles_year_rollover(fetcher):
    # Z6 (Dec 2026) expires before H7 (Mar 2027)
    assert fetcher._build_spread_symbol("ESZ6", "ESH7") == "ESZ6-ESH7"


# --------------------------------------------------------------------- #
# fetch_calendar_spread_data — composition contract (privates mocked)
# --------------------------------------------------------------------- #

def test_fetch_calendar_spread_data_composes_privates(fetcher, es_books, es_trades):
    books = {"ESM6": es_books["front"], "ESU6": es_books["back"],
             "ESM6-ESU6": es_books["spread"]}

    with patch.object(
        fetcher, "_resolve_symbols",
        return_value=("ESM6", "ESU6", "ESM6-ESU6"),
    ), patch.object(
        fetcher, "_fetch_single_instrument",
        side_effect=lambda sym, start, end: books[sym],
    ) as mock_fetch, patch.object(
        fetcher, "_fetch_single_instrument_trades",
        side_effect=lambda sym, start, end: es_trades[sym],
    ) as mock_trades:
        result = fetcher.fetch_calendar_spread_data(
            "M6", "ES", "2026-06-10T15:00:00Z", "2026-06-10T17:00:00Z"
        )

    # each instrument's book and trades fetched separately, with the requested window
    for mock in (mock_fetch, mock_trades):
        assert mock.call_count == 3
        fetched_symbols = [c.args[0] for c in mock.call_args_list]
        assert fetched_symbols == ["ESM6", "ESU6", "ESM6-ESU6"]
        for c in mock.call_args_list:
            assert c.args[1] == "2026-06-10T15:00:00Z"
            assert c.args[2] == "2026-06-10T17:00:00Z"

    # result contract
    assert isinstance(result, CalendarSpreadData)
    assert result.front_symbol == "ESM6"
    assert result.back_symbol == "ESU6"
    assert result.spread_symbol == "ESM6-ESU6"
    # books and trades pass through untouched, on raw event timestamps
    assert result.front["bid_px_00"].iloc[0] == 7340.00
    assert result.spread["ask_px_00"].iloc[0] == 60.75
    assert result.spread_trades["price"].iloc[0] == 60.75
    assert result.front_trades["side"].iloc[0] == "A"


def test_fetch_calendar_spread_data_validates_before_fetching(fetcher):
    with patch.object(fetcher, "_fetch_single_instrument") as mock_fetch:
        with pytest.raises(ValueError):
            fetcher.fetch_calendar_spread_data("A6", "ES", "t0", "t1")  # bad month code
    mock_fetch.assert_not_called()


# --------------------------------------------------------------------- #
# iter_calendar_spread_data — lazy per-day composition (privates mocked)
# --------------------------------------------------------------------- #

def test_iter_calendar_spread_data_is_lazy_and_yields_per_day(fetcher):
    windows = [
        (pd.Timestamp("2026-06-10T00:00:00Z"), pd.Timestamp("2026-06-11T00:00:00Z")),
        (pd.Timestamp("2026-06-11T00:00:00Z"), pd.Timestamp("2026-06-12T00:00:00Z")),
    ]
    day1, day2 = object(), object()  # sentinels standing in for CalendarSpreadData

    with patch.object(
        fetcher, "_trading_days", return_value=windows
    ), patch.object(
        fetcher, "fetch_calendar_spread_data", side_effect=[day1, day2]
    ) as mock_fetch:
        gen = fetcher.iter_calendar_spread_data(
            "M6", "ES", "2026-06-10T00:00:00Z", "2026-06-12T00:00:00Z"
        )

        # generator: nothing fetched until consumed
        mock_fetch.assert_not_called()

        assert next(gen) is day1
        assert mock_fetch.call_count == 1  # only one day in memory so far

        assert next(gen) is day2
        with pytest.raises(StopIteration):
            next(gen)

    # each day fetched with its own sub-window; cycle defaults to None here
    # (resolution against product_specs.json happens one layer deeper, in
    # _resolve_symbols, which fetch_calendar_spread_data is mocked out and
    # never actually calls in this composition test)
    assert mock_fetch.call_args_list[0].args == ("M6", "ES", *windows[0], None)
    assert mock_fetch.call_args_list[1].args == ("M6", "ES", *windows[1], None)


# --------------------------------------------------------------------- #
# fetch_calendar_spread_contract_specification — composition contract
# --------------------------------------------------------------------- #

def test_fetch_contract_specification_composes_privates(fetcher, es_spec):
    fake_defs = pd.DataFrame({"raw_symbol": ["ESM6", "ESU6", "ESM6-ESU6"]})

    with patch.object(
        fetcher, "_resolve_symbols",
        return_value=("ESM6", "ESU6", "ESM6-ESU6"),
    ), patch.object(
        fetcher, "_fetch_definitions", return_value=fake_defs
    ) as mock_defs, patch.object(
        fetcher, "_build_contract_spec", return_value=es_spec
    ) as mock_build:
        spec = fetcher.fetch_calendar_spread_contract_specification("M6", "ES")

    mock_defs.assert_called_once_with("ESM6", "ESU6", "ESM6-ESU6")
    mock_build.assert_called_once_with("ESM6", "ESU6", "ESM6-ESU6", fake_defs)
    assert spec == es_spec


# --------------------------------------------------------------------- #
# Issue #6 — implemented fetch methods (real behavior, mocked Databento client)
#
# These call the actual private methods (not mocked out, unlike the
# composition tests above) against a MagicMock client standing in for
# databento.Historical, so the real fetch/normalization/schema logic gets
# exercised rather than just the call contract.
# --------------------------------------------------------------------- #

def test_init_rejects_non_databento_levels():
    # Databento only exposes mbp-1 and mbp-10 — LEVELS(=2), used above purely
    # for compact test book shapes, is not a real schema and must be rejected
    # at construction rather than failing confusingly on first fetch.
    with pytest.raises(ValueError, match="mbp-1 and mbp-10"):
        MarketDataFetcher(client=None, levels=LEVELS)


def test_init_accepts_valid_levels():
    MarketDataFetcher(client=None, levels=1)
    MarketDataFetcher(client=None, levels=10)


def test_fetch_single_instrument_uses_mbp1_schema_for_levels_1():
    raw = make_book(["2026-06-10T16:00:00Z", "2026-06-10T16:00:01Z"],
                     bid_px=[7340.00, 7340.25], ask_px=[7340.25, 7340.50], levels=1)
    client = _client_returning(raw)
    f = MarketDataFetcher(client=client, levels=1)

    result = f._fetch_single_instrument("ESM6", "2026-06-10T16:00:00Z", "2026-06-10T16:00:02Z")

    _, kwargs = client.timeseries.get_range.call_args
    assert kwargs["schema"] == "mbp-1"
    assert kwargs["symbols"] == ["ESM6"]
    assert list(result.columns) == book_columns(1)
    assert result.index.name == BOOK_INDEX_NAME
    assert str(result.index.tz) == "UTC"
    assert result.index.is_monotonic_increasing


def test_fetch_single_instrument_uses_mbp10_schema_for_levels_10():
    raw = make_book(["2026-06-10T16:00:00Z"], bid_px=[7340.00], ask_px=[7340.25], levels=10)
    client = _client_returning(raw)
    f = MarketDataFetcher(client=client, levels=10)

    f._fetch_single_instrument("ESM6", "2026-06-10T16:00:00Z", "2026-06-10T16:00:01Z")

    _, kwargs = client.timeseries.get_range.call_args
    assert kwargs["schema"] == "mbp-10"


def test_fetch_single_instrument_empty_result_keeps_schema():
    empty = pd.DataFrame(columns=book_columns(1))
    client = _client_returning(empty)
    f = MarketDataFetcher(client=client, levels=1)

    result = f._fetch_single_instrument("ESM6", "2026-06-10T16:00:00Z", "2026-06-10T16:00:01Z")

    assert result.empty
    assert list(result.columns) == book_columns(1)
    assert str(result.index.tz) == "UTC"
    assert result.index.name == BOOK_INDEX_NAME


def test_fetch_single_instrument_normalizes_naive_databento_index_to_utc():
    # Databento's to_df() output isn't guaranteed tz-aware; the fetcher must
    # not silently misinterpret a naive index as some other zone.
    idx = pd.DatetimeIndex(["2026-06-10T16:00:00"], name=BOOK_INDEX_NAME)  # naive
    raw = pd.DataFrame({c: [1.0] for c in book_columns(1)}, index=idx)
    client = _client_returning(raw)
    f = MarketDataFetcher(client=client, levels=1)

    result = f._fetch_single_instrument("ESM6", "2026-06-10T16:00:00Z", "2026-06-10T16:00:01Z")
    assert str(result.index.tz) == "UTC"


def test_fetch_single_instrument_trades_schema_and_columns():
    trades = make_trades(["2026-06-10T16:00:00.5Z"], [7340.00], [5.0], ["A"])
    client = _client_returning(trades)
    f = MarketDataFetcher(client=client, levels=1)

    result = f._fetch_single_instrument_trades(
        "ESM6", "2026-06-10T16:00:00Z", "2026-06-10T16:00:01Z"
    )

    _, kwargs = client.timeseries.get_range.call_args
    assert kwargs["schema"] == "trades"
    assert list(result.columns) == TRADE_COLUMNS
    assert result["price"].iloc[0] == 7340.00


def test_trading_days_clips_first_and_last_session():
    f = MarketDataFetcher(client=MagicMock(), levels=1)
    # Mon 2026-06-08 through Wed 2026-06-10 — 3 sessions, no holiday in range
    windows = f._trading_days("2026-06-08T18:00:00Z", "2026-06-10T12:00:00Z")

    assert len(windows) == 3
    first_start, _ = windows[0]
    _, last_end = windows[-1]
    assert first_start == _normalize_time("2026-06-08T18:00:00Z")
    assert last_end == _normalize_time("2026-06-10T12:00:00Z")
    for s, e in windows:
        assert s < e
        assert s.tzinfo is not None and e.tzinfo is not None


def test_trading_days_skips_weekend():
    f = MarketDataFetcher(client=MagicMock(), levels=1)
    windows = f._trading_days("2026-06-05T00:00:00Z", "2026-06-08T23:59:59Z")
    session_dates = {s.date() for s, _ in windows}
    assert pd.Timestamp("2026-06-06").date() not in session_dates  # Saturday


def test_trading_days_raises_on_inverted_range():
    f = MarketDataFetcher(client=MagicMock(), levels=1)
    with pytest.raises(ValueError):
        f._trading_days("2026-06-10T00:00:00Z", "2026-06-09T00:00:00Z")


# --------------------------------------------------------------------- #
# Issue #7 — _fetch_definitions and _build_contract_spec
# (real behavior, mocked Databento client, real product_specs.json)
# --------------------------------------------------------------------- #

def _fetcher_with_definitions_client(definitions_df) -> MagicMock:
    client = MagicMock()
    result = MagicMock()
    result.to_df.return_value = definitions_df
    client.timeseries.get_range.return_value = result
    return client


def test_fetch_definitions_queries_all_three_symbols():
    defs = pd.DataFrame({"raw_symbol": ["ESM6", "ESU6", "ESM6-ESU6"]})
    client = _fetcher_with_definitions_client(defs)
    f = MarketDataFetcher(client=client, levels=1)

    result = f._fetch_definitions("ESM6", "ESU6", "ESM6-ESU6")

    _, kwargs = client.timeseries.get_range.call_args
    assert kwargs["schema"] == "definition"
    assert kwargs["symbols"] == ["ESM6", "ESU6", "ESM6-ESU6"]
    assert len(result) == 3


def test_fetch_definitions_raises_on_empty_result():
    client = _fetcher_with_definitions_client(pd.DataFrame())
    f = MarketDataFetcher(client=client, levels=1)
    with pytest.raises(ValueError):
        f._fetch_definitions("ESM6", "ESU6", "ESM6-ESU6")


def test_build_contract_spec_uses_product_specs_json():
    # ES's real termination_rule (3rd Friday) -> 2026-06-19 for M6, matches
    # Databento's reported expiration exactly, so no ValueError.
    defs = pd.DataFrame({
        "raw_symbol": ["ESM6", "ESU6"],
        "expiration": [
            pd.Timestamp("2026-06-19", tz="UTC"),
            pd.Timestamp("2026-09-18", tz="UTC"),
        ],
    })
    f = MarketDataFetcher(client=MagicMock(), levels=1)

    spec = f._build_contract_spec("ESM6", "ESU6", "ESM6-ESU6", defs)

    assert spec.product_code == "ES"
    assert spec.outright_tick_size == Decimal("0.25")
    assert spec.spread_tick_size == Decimal("0.05")
    assert spec.contract_multiplier == Decimal("50")
    assert spec.front_expiration.date() == pd.Timestamp("2026-06-19").date()


def test_build_contract_spec_raises_on_expiration_mismatch():
    # Databento reports an expiration far from what ES's termination_rule
    # would compute — should raise rather than silently trust the rule.
    defs = pd.DataFrame({
        "raw_symbol": ["ESM6", "ESU6"],
        "expiration": [
            pd.Timestamp("2026-06-01", tz="UTC"),  # nowhere near the 3rd Friday
            pd.Timestamp("2026-09-18", tz="UTC"),
        ],
    })
    f = MarketDataFetcher(client=MagicMock(), levels=1)

    with pytest.raises(ValueError, match="differs from Databento"):
        f._build_contract_spec("ESM6", "ESU6", "ESM6-ESU6", defs)


def test_build_contract_spec_skips_validation_when_expiration_column_absent():
    # No "expiration" column at all — validation should no-op, not crash.
    defs = pd.DataFrame({"raw_symbol": ["ESM6", "ESU6"]})
    f = MarketDataFetcher(client=MagicMock(), levels=1)
    spec = f._build_contract_spec("ESM6", "ESU6", "ESM6-ESU6", defs)
    assert spec.product_code == "ES"


def test_build_contract_spec_unknown_product_raises():
    f = MarketDataFetcher(client=MagicMock(), levels=1)
    with pytest.raises(Exception):  # ProductSpecError — no JSON entry for "ZZ"
        f._build_contract_spec("ZZM6", "ZZU6", "ZZM6-ZZU6", pd.DataFrame())