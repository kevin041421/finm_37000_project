"""Run the deferred book validation metrics on the cached ES demo sessions (Issue #15).

Wires ``CachedMarketDataFetcher`` (offline path) into the pure functions of
``deferred_metrics.py`` and ``implied_outright.py``.

It compares the modeled deferred book (front + spread) against the real 
deferred book and quantifies the joint-book assumption.

"""

from __future__ import annotations

import pandas as pd

from analysis_config import (
    BOOK_LEVELS,
    DAYS,
    FRONT_MONTH,
    PRODUCT,
    SESSION_LENGTH,
    SESSION_START_UTC,
)
from cached_market_data_fetcher import CachedMarketDataFetcher
from implied_common import infer_spread_convention
from implied_outright import implied_back_book
from deferred_metrics import (
    decompose_liquidity,
    divergence_episodes,
    validation_summary
)


def _print_summary(day: str, t1: pd.Timestamp, stats: dict) -> None:
    """Format and print the session validation metrics."""
    # Programmatic validation rule: Touch matches >90% of the time, and
    # more than 50% of the native liquidity is implied from the front/spread.
    validated = stats['touch_match_pct'] >= 90.0 and stats['pct_implied'] >= 50.0
    status = "VALIDATED" if validated else "WEAK COUPLING"
    
    print(f"\n{day} {SESSION_START_UTC}-{t1:%H:%M:%S} UTC")
    print(f"  Updates aligned         {stats['updates']:,}")
    print(f"  Touch match rate        {stats['touch_match_pct']:.2f}%")
    print(
        f"  Divergence episodes     {stats['divergence_count']:,} "
        f"(Avg: {stats['avg_divergence_ms']:.1f}ms, "
        f"Max: {stats['max_divergence_ms']:.1f}ms, "
        f"Max Gap: {stats['max_gap_ticks']:.1f} ticks)"
    )
    print(
        f"  Avg depth (contracts)   native {stats['avg_nat_depth']:.1f}  "
        f"implied {stats['avg_imp_depth']:.1f}  "
        f"direct {stats['avg_dir_depth']:.1f}"
    )
    print(f"  Liquidity makeup        {stats['pct_implied']:.1f}% implied / {stats['pct_direct']:.1f}% direct")
    print(f"  Joint-book assumption   [{status}]")


def main() -> None:
    fetcher = CachedMarketDataFetcher(client=None, levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(FRONT_MONTH, PRODUCT)
    tick = float(spec.outright_tick_size)

    print(
        f"=== Deferred Book Validation: {spec.back_symbol} ===\n"
        f"Comparing modeled {spec.back_symbol} (implied from front + spread)\n"
        f"against the native {spec.back_symbol} book pulled from Databento."
    )

    legs = None
    for day in DAYS:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t1)

        if legs is None:
            # Infer convention once so the modeled book acts as a strict,
            # independent benchmark moving forward, never touching data.back.
            legs = infer_spread_convention(data.front, data.back, data.spread)
            print(f"\nInferred convention: spread = {legs.name}")

        # 1. Build the modeled book
        implied_back = implied_back_book(data, spec, legs=legs)

        # 2. Decompose against the real book
        decomp = decompose_liquidity(implied_back, data.back, tick)
        
        # 3. Extract episode data
        episodes = divergence_episodes(decomp)
        
        # 4. Summarize and print
        stats = validation_summary(decomp, episodes)
        _print_summary(day, t1, stats)


if __name__ == "__main__":
    main()