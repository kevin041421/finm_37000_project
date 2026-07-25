"""Unit tests for product_specs.py: loading product_specs.json and
computing termination dates from each rule type. Uses the real
product_specs.json checked into src/ (not a mock) so these also serve as a
sanity check on the JSON's own content, plus a couple of temp-file cases
for error handling that shouldn't depend on the real file's contents.
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import product_specs as ps  # noqa: E402


@pytest.fixture
def specs():
    return ps.load_product_specs()  # real src/product_specs.json


def test_load_product_specs_has_all_four_products(specs):
    assert set(specs) == {"ES", "ZN", "CL", "6E"}


def test_get_product_spec_known_product(specs):
    spec = ps.get_product_spec("ES", specs=specs)
    assert spec.product_code == "ES"
    assert spec.listing_cycle == "HMUZ"


def test_get_product_spec_unknown_product_raises(specs):
    with pytest.raises(ps.ProductSpecError, match="ZZ"):
        ps.get_product_spec("ZZ", specs=specs)


def test_decimal_precision_preserved_not_float(specs):
    from decimal import Decimal
    # 0.00005 and 0.01 aren't exactly representable in binary float — this
    # is the whole reason tick sizes are stored/loaded as Decimal, not float.
    assert ps.get_product_spec("6E", specs=specs).outright_tick_size == Decimal("0.00005")
    assert ps.get_product_spec("CL", specs=specs).outright_tick_size == Decimal("0.01")


# --------------------------------------------------------------------- #
# compute_termination_date — one test per rule type actually in use
# --------------------------------------------------------------------- #

def test_es_third_friday(specs):
    spec = ps.get_product_spec("ES", specs=specs)
    result = ps.compute_termination_date(spec, "M", 2026)  # June 2026
    assert result.date() == pd.Timestamp("2026-06-19").date()  # verified 3rd Friday


def test_6e_third_wednesday_minus_two_business_days(specs):
    spec = ps.get_product_spec("6E", specs=specs)
    result = ps.compute_termination_date(spec, "M", 2026)  # June 2026
    # 3rd Wednesday of June 2026 is the 17th; 2 business days before is the 15th (Monday)
    assert result.date() == pd.Timestamp("2026-06-15").date()


def test_cl_business_days_before_25th_of_prior_month(specs):
    spec = ps.get_product_spec("CL", specs=specs)
    result = ps.compute_termination_date(spec, "M", 2026)  # June contract -> prior month May
    # 25th of May 2026 is a Monday; 3 business days before -> Wed May 20
    assert result.date() == pd.Timestamp("2026-05-20").date()


def test_cl_handles_year_rollover_for_january_contract(specs):
    spec = ps.get_product_spec("CL", specs=specs)
    # January contract's "prior month" is December of the previous year
    result = ps.compute_termination_date(spec, "F", 2027)
    assert result.year == 2026
    assert result.month == 12


def test_zn_business_days_before_month_end(specs):
    spec = ps.get_product_spec("ZN", specs=specs)
    result = ps.compute_termination_date(spec, "M", 2026)  # June 2026
    # last business day of June 2026 is Tue Jun 30; 7 business days before -> Jun 19
    assert result.date() == pd.Timestamp("2026-06-19").date()


def test_compute_termination_date_unknown_month_code_raises(specs):
    spec = ps.get_product_spec("ES", specs=specs)
    with pytest.raises(ps.ProductSpecError):
        ps.compute_termination_date(spec, "A", 2026)  # 'A' isn't a CME month code


# --------------------------------------------------------------------- #
# Error handling — temp files, independent of the real JSON's content
# --------------------------------------------------------------------- #

def test_load_product_specs_missing_file_raises():
    with pytest.raises(ps.ProductSpecError, match="No product_specs.json"):
        ps.load_product_specs(path=Path("/nonexistent/product_specs.json"))


def test_load_product_specs_rejects_manual_entry_sentinel():
    payload = {
        "XX": {
            "asset_class": "REQUIRES_MANUAL_ENTRY",
            "outright_tick_size": "0.25",
            "spread_tick_size": "0.05",
            "contract_multiplier": "50",
            "quote_scaling": "1",
            "price_display_format": "decimal",
            "fractional_denominator": None,
            "listing_cycle": ["H", "M", "U", "Z"],
            "termination_rule": {"type": "nth_weekday_of_month", "weekday": "Friday", "n": 3},
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ps.ProductSpecError, match="asset_class"):
            ps.load_product_specs(path=temp_path)
    finally:
        temp_path.unlink()


def test_compute_termination_date_unsupported_rule_type_raises():
    spec = ps.ProductSpec(
        product_code="XX",
        outright_tick_size=None, spread_tick_size=None, contract_multiplier=None,
        quote_scaling=None, price_display_format="decimal", fractional_denominator=None,
        listing_cycle="HMUZ",
        termination_rule={"type": "some_future_rule_type"},
        asset_class="equity",
    )
    with pytest.raises(ps.ProductSpecError, match="unsupported"):
        ps.compute_termination_date(spec, "M", 2026)