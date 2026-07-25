"""Run the P&L pipeline end-to-end (Issue #9 demo).

Wires the components together:

    MarketDataFetcher  ->  CalendarSpreadData  ->  replay  ->  PnLCalculator

Uses the TEMP ``HardcodedESFetcher`` (ES Jun/Sep 2026, real Databento data,
disk-cached) until Issues #6/#7 land. The ``LeggingCostCalculator`` is the
real #8 logic (adopted locally from PR #21 with the side-convention fix).

Run:
    uv run src/run_pnl_pipeline.py
"""

from __future__ import annotations

import databento as db
import pandas as pd

from demo_print_books import HardcodedESFetcher
from fee_schedule import get_fee_rates
from legging_cost_calculator import LeggingCostCalculator
from market_data_fetcher import CalendarSpreadContractSpec
from pnl_calculator import PnLCalculator, replay
from util import get_databento_api_key

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
FEES = get_fee_rates("ES", FEE_TIER)
CONTRACT_MULTIPLIER = 50.0  # ES: $ per index point

# TEMP hardcoded ES spec until #7's mapping file provides it
ES_SPEC = CalendarSpreadContractSpec(
    product_code="ES",
    front_symbol="ESM6",
    back_symbol="ESU6",
    spread_symbol="ESM6-ESU6",
    outright_tick_size=0.25,
    spread_tick_size=0.05,
    contract_multiplier=CONTRACT_MULTIPLIER,
)


def main() -> None:
    client = db.Historical(get_databento_api_key())
    fetcher = HardcodedESFetcher(client=client, levels=1)
    cost_calculator = LeggingCostCalculator(ES_SPEC)  # real #8 logic (from PR #21, fixed)
    pnl_calculator = PnLCalculator(
        p=P_QUEUE_HEAD,
        max_position=MAX_POSITION,
        passive_fee=FEES.passive_fee,
        aggressive_fee=FEES.aggressive_fee,
        contract_multiplier=CONTRACT_MULTIPLIER,
        seed=SEED,
    )

    for day in DAYS:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        print(f"\n{day} {SESSION_START_UTC}–{t1:%H:%M:%S} UTC  fetching...", flush=True)

        data = fetcher.fetch_calendar_spread_data("M6", "ES", t0, t1)
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
