"""Representative CME exchange fee/rebate rates for testing (Issue #9/#10).

Approximate per-side, per-contract exchange fees for E-mini equity index
futures, read from the CME fee schedule effective 2026-06-01
(https://www.cmegroup.com/company/clearing-fees.html). VERIFY exact cents
against the official XLS / Non-Member Fee Finder before citing in the report
-- these are testing placeholders, and fees change (last change 2026-04-01).

Conventions (matching ``PnLCalculator``):
- ``passive_fee``: charged per passively filled SPREAD contract -> the
  Globex *spreads* rate for the tier.
- ``aggressive_fee``: charged per spread contract legged out. Legging is two
  outright fills (front + back) -> approximately 2 x the Globex *outrights*
  rate for the tier.
- Negative values would represent a per-contract rebate. CME has no standing
  maker rebate for these products -- fee tiers only go down to (near) zero
  via membership/incentive programs -- so a negative ``passive_fee`` models
  the *hypothetical* incentive whose break-even level Issue #10 solves for.

TODO(#7): fold into the contract mapping file so fees live with the rest of
the per-product static data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeRates:
    """Per-side, per-contract exchange fees in dollars."""

    passive_fee: float     # per passively filled spread contract
    aggressive_fee: float  # per spread contract legged out (2 outright fills)


#: fee tier -> rates, per product. E-mini equity index products (ES, NQ, YM,
#: RTY) share the CME E-mini equity fee schedule.
_EMINI_EQUITY_TIERS: dict[str, FeeRates] = {
    # baseline: no membership, no incentive program
    "non_member": FeeRates(passive_fee=1.30, aggressive_fee=2.76),
    # clearing/equity member firms (approx. mid-tier member rates)
    "member_firm": FeeRates(passive_fee=0.45, aggressive_fee=0.90),
    # individual members (lowest published tier)
    "individual_member": FeeRates(passive_fee=0.09, aggressive_fee=0.18),
    # full fee waiver -- an incentive program that charges nothing
    "fee_waiver": FeeRates(passive_fee=0.0, aggressive_fee=0.0),
}

FEE_SCHEDULE: dict[str, dict[str, FeeRates]] = {
    "ES": _EMINI_EQUITY_TIERS,
    "NQ": _EMINI_EQUITY_TIERS,
    "YM": _EMINI_EQUITY_TIERS,
    "RTY": _EMINI_EQUITY_TIERS,
}


def get_fee_rates(product: str, tier: str = "non_member") -> FeeRates:
    """Look up representative fee rates for a product and fee tier.

    >>> get_fee_rates("ES").passive_fee
    1.3
    >>> get_fee_rates("ES", "individual_member").aggressive_fee
    0.18
    """
    try:
        tiers = FEE_SCHEDULE[product]
    except KeyError:
        raise KeyError(
            f"No fee data for product {product!r} "
            f"(have: {sorted(FEE_SCHEDULE)})"
        ) from None
    try:
        return tiers[tier]
    except KeyError:
        raise KeyError(
            f"Unknown fee tier {tier!r} (have: {sorted(tiers)})"
        ) from None
