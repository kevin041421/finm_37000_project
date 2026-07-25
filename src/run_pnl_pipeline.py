
from __future__ import annotations

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from fee_schedule import get_fee_rates
from legging_cost_calculator import LeggingCostCalculator
from pnl_calculator import PnLCalculator, replay

PRODUCT = "ES"
FRONT_MONTH = "M6"

# Two demo sessions: a 60-second window on each day, matching the windows
# demo_print_books.py already cached to disk — so this runs offline.
DAYS = ["2026-06-09", "2026-06-10"]
SESSION_START_UTC = "16:00:00"  # 10:00 AM CT, the README snapshot hour
SESSION_LENGTH = pd.Timedelta(seconds=60)

# Model parameters (Issue #9) — tune these
P_QUEUE_HEAD = 0.5      # probability a trade fills us in full (queue head)
MAX_POSITION = 5.0      # max spread contracts we are willing to hold
SEED = 42               # fills are random; fix the seed for reproducibility.
                        # Average net_pnl over several seeds for expected P&L.
FEE_TIER = "non_member" # see fee_schedule.py: non_member | member_firm |
                        # individual_member | fee_waiver

BOOK_LEVELS = 1         # top of book is all the P&L model reads


def main() -> None:
    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(FRONT_MONTH, PRODUCT)
    contract_multiplier = float(spec.contract_multiplier)

    fees = get_fee_rates(spec.product_code, FEE_TIER)

    print(
        f"{spec.spread_symbol}  ({spec.front_symbol} / {spec.back_symbol})\n"
        f"  tick: outright {spec.outright_tick_size}, spread {spec.spread_tick_size}"
        f"   multiplier: ${contract_multiplier:g}/pt\n"
        f"  fees ({FEE_TIER}): passive ${fees.passive_fee:.2f}, "
        f"aggressive ${fees.aggressive_fee:.2f} per contract"
    )

    cost_calculator = LeggingCostCalculator(spec)
    pnl_calculator = PnLCalculator(
        p=P_QUEUE_HEAD,
        max_position=MAX_POSITION,
        passive_fee=fees.passive_fee,
        aggressive_fee=fees.aggressive_fee,
        contract_multiplier=contract_multiplier,
        seed=SEED,
    )

    for day in DAYS:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        print(f"\n{day} {SESSION_START_UTC}–{t1:%H:%M:%S} UTC  fetching...", flush=True)

        data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t1)
        print(
            f"  book events: front={len(data.front)}, back={len(data.back)}, "
            f"spread={len(data.spread)}; spread trades={len(data.spread_trades)}"
        )

        consumed = replay(data, cost_calculator, pnl_calculator)
        print(f"  consumed {consumed} spread transaction(s)")

    print(f"\nTotal transactions consumed: {pnl_calculator.transaction_count}")
    print("\nExpected P&L summary:")
    print(pnl_calculator.generate_pnl().to_string())


if __name__ == "__main__":
    main()
