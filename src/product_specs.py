"""product_specs.py: loader + expiration-rule engine for the CME contract
mapping file (Issue #7).

Static structural data (tick sizes, multiplier, listing cycle, termination
rule) lives in product_specs.json, generated/audited by
generate_product_specs.py and reviewed like any other config change (see
that module's docstring for the generate/audit workflow).

This module is the READ side: a plain lookup (`get_product_spec`) plus a
pure function that evaluates a product's termination_rule against a
specific contract month/year (`compute_termination_date`). No Databento
calls happen here — this is intentionally offline. Cross-checking a
computed date against a live Databento definition is the caller's job (see
market_data_fetcher.MarketDataFetcher._build_contract_spec).

Termination rule types (tagged union, matches product_specs.json's
"termination_rule" field):

  nth_weekday_of_month:
      {"type": "nth_weekday_of_month", "weekday": "Friday", "n": 3,
       "offset_business_days": <int, optional>}
      -> the n-th <weekday> of the contract month, then shifted back by
         `offset_business_days` plain business days if given (e.g. 6E:
         2 business days before the 3rd Wednesday).

  business_days_before_month_end:
      {"type": "business_days_before_month_end", "days": <int>}
      -> N business days before the last business day of the contract month.

  business_days_before_day_of_prior_month:
      {"type": "business_days_before_day_of_prior_month", "days": <int>,
       "day_of_month": <int>}
      -> N business days before the given calendar day of the month prior
         to the contract month (e.g. CL: 3 business days before the 25th
         of the prior month).

CAUTION #1 — no CME holiday calendar: "business days" here means plain
Monday-Friday, with NO holiday calendar applied (the same tradeoff made in
market_data_fetcher._trading_days, for the same reason: no market-calendar
dependency in this repo). A computed date can be off by a day or two around
a CME holiday. This is why _build_contract_spec cross-checks every computed
expiration against Databento's own `definition` schema `expiration` field
and raises on mismatch rather than trusting this computation blind — treat
this as a fast offline first guess, not ground truth.

CAUTION #2 — single-digit year ambiguity: CME symbols carry only the last
digit of the year (e.g. "M6" = June 2026). That's only unambiguous within
roughly a 5-year window of "now" — resolving "6" to a full year requires a
"nearest to today" heuristic (see market_data_fetcher._full_year), which
will silently pick the wrong decade for far-dated or historical contracts.
This isn't specific to this module — every symbol in this codebase has the
same ambiguity — but it's most consequential here, where a wrong year
produces a wrong expiration date with no other check to catch it. Flagged
as a known, unresolved limitation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_SPECS_PATH = Path(__file__).parent / "product_specs.json"

MONTH_CODE_TO_CALENDAR_MONTH = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

WEEKDAY_NAME_TO_INT = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

_MANUAL_FIELDS_SENTINEL = "REQUIRES_MANUAL_ENTRY"


class ProductSpecError(Exception):
    """Raised for missing products, malformed entries, or unsupported rule types."""


@dataclass(frozen=True)
class ProductSpec:
    """One product's static structural metadata, as loaded from
    product_specs.json. Numeric fields are Decimal (see
    generate_product_specs.py's docstring for why: tick sizes like 0.00005
    or 0.01 aren't exactly representable in binary float, and that error
    compounds once multiplied into dollar P&L)."""

    product_code: str
    outright_tick_size: Decimal
    spread_tick_size: Decimal
    contract_multiplier: Decimal
    quote_scaling: Decimal
    price_display_format: str
    fractional_denominator: Optional[int]
    listing_cycle: str  # e.g. "HMUZ", in calendar order
    termination_rule: dict
    asset_class: str


def load_product_specs(path: Path = DEFAULT_SPECS_PATH) -> dict[str, ProductSpec]:
    """Load and parse product_specs.json into ProductSpec objects.

    Call once per process and reuse the result (or use MarketDataFetcher,
    which caches this internally) — this hits disk, not Databento, but
    there's still no reason to re-parse the file on every lookup.

    Raises ProductSpecError if any product's manually-authored fields
    (asset_class, termination_rule, price_display_format) are still the
    "REQUIRES_MANUAL_ENTRY" placeholder generate_product_specs.py leaves
    for a newly-onboarded product — better to fail loudly here than
    silently compute a nonsense expiration date from a missing rule.
    """
    if not path.exists():
        raise ProductSpecError(
            f"No product_specs.json found at {path}. Run "
            f"generate_product_specs.py to create it."
        )
    with path.open() as f:
        raw = json.load(f)

    specs: dict[str, ProductSpec] = {}
    for code, entry in raw.items():
        for manual_field in ("asset_class", "termination_rule", "price_display_format"):
            if entry.get(manual_field) == _MANUAL_FIELDS_SENTINEL:
                raise ProductSpecError(
                    f"{code}: {manual_field!r} in product_specs.json still needs "
                    f"manual entry (see generate_product_specs.py) before this "
                    f"product can be used."
                )
        try:
            specs[code] = ProductSpec(
                product_code=code,
                outright_tick_size=Decimal(entry["outright_tick_size"]),
                spread_tick_size=Decimal(entry["spread_tick_size"]),
                contract_multiplier=Decimal(entry["contract_multiplier"]),
                quote_scaling=Decimal(entry.get("quote_scaling", "1")),
                price_display_format=entry["price_display_format"],
                fractional_denominator=entry.get("fractional_denominator"),
                listing_cycle="".join(entry["listing_cycle"]),
                termination_rule=entry["termination_rule"],
                asset_class=entry["asset_class"],
            )
        except KeyError as e:
            raise ProductSpecError(f"{code}: product_specs.json entry missing field {e}") from e

    return specs


def get_product_spec(
    product_code: str,
    specs: Optional[dict[str, ProductSpec]] = None,
    path: Path = DEFAULT_SPECS_PATH,
) -> ProductSpec:
    """Look up one product's static spec. Pass a pre-loaded `specs` dict
    (from load_product_specs) to avoid re-reading the file on every call;
    if omitted, loads fresh from `path`.

    Raises ProductSpecError (not KeyError) on a missing product, with the
    list of known products — this is the boundary marker for "out of
    currently-supported scope" the fetcher relies on.
    """
    specs = specs if specs is not None else load_product_specs(path)
    try:
        return specs[product_code]
    except KeyError:
        raise ProductSpecError(
            f"No product_specs.json entry for {product_code!r}. "
            f"Known products: {sorted(specs)}"
        ) from None


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """The n-th occurrence (1-indexed) of `weekday` (Mon=0..Sun=6) in the
    given calendar month/year."""
    first = pd.Timestamp(year=year, month=month, day=1)
    first_weekday_offset = (weekday - first.weekday()) % 7
    first_occurrence = first + pd.Timedelta(days=first_weekday_offset)
    return first_occurrence + pd.Timedelta(weeks=n - 1)


def _shift_business_days(date: pd.Timestamp, business_days: int) -> pd.Timestamp:
    """Shift `date` by `business_days` (negative = backward), counting
    plain Mon-Fri business days. No holiday calendar — see module
    docstring CAUTION #1."""
    return date + pd.tseries.offsets.BDay(business_days)


def _last_business_day_of_month(year: int, month: int) -> pd.Timestamp:
    month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return pd.offsets.BDay().rollback(month_end)


def compute_termination_date(spec: ProductSpec, month_code: str, year: int) -> pd.Timestamp:
    """Evaluate `spec`'s termination_rule for a specific contract month/year.

    Args:
        spec: the product's ProductSpec (from get_product_spec).
        month_code: CME month letter, e.g. "M" for June.
        year: full 4-digit year (see market_data_fetcher._full_year for
            resolving a symbol's single year digit to a full year first).

    Raises ProductSpecError for an unrecognized month_code, weekday name,
    or termination_rule type — fails loudly rather than guessing.
    """
    rule = spec.termination_rule
    rule_type = rule.get("type")
    calendar_month = MONTH_CODE_TO_CALENDAR_MONTH.get(month_code)
    if calendar_month is None:
        raise ProductSpecError(f"Unknown CME month code {month_code!r}")

    if rule_type == "nth_weekday_of_month":
        weekday_name = rule.get("weekday")
        weekday = WEEKDAY_NAME_TO_INT.get(weekday_name)
        if weekday is None:
            raise ProductSpecError(
                f"{spec.product_code}: unknown weekday {weekday_name!r} in termination_rule"
            )
        date = _nth_weekday_of_month(year, calendar_month, weekday, rule["n"])
        offset_bd = rule.get("offset_business_days")
        if offset_bd:
            date = _shift_business_days(date, offset_bd)
        return date

    if rule_type == "business_days_before_month_end":
        last_bday = _last_business_day_of_month(year, calendar_month)
        return _shift_business_days(last_bday, -rule["days"])

    if rule_type == "business_days_before_day_of_prior_month":
        prior_month = calendar_month - 1
        prior_year = year
        if prior_month == 0:
            prior_month = 12
            prior_year -= 1
        anchor = pd.Timestamp(year=prior_year, month=prior_month, day=rule["day_of_month"])
        return _shift_business_days(anchor, -rule["days"])

    raise ProductSpecError(
        f"{spec.product_code}: unsupported termination_rule type {rule_type!r}"
    )