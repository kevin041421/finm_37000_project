"""LeggingCostCalculator: cost of hedging a spread fill in the outrights (Issue #8)."""
from __future__ import annotations
from typing import Optional

import pandas as pd
from market_data_fetcher import CalendarSpreadContractSpec

class LeggingCostCalculator:
    """Computes the mechanical cost of legging out a spread fill (Issue #8).

    Takes a ready-made ``CalendarSpreadContractSpec`` rather than building one
    from bare symbols: since #7 landed, the spec carries the tick sizes and
    contract multiplier needed to turn points into dollars, and it is produced
    by ``MarketDataFetcher.fetch_calendar_spread_contract_specification``.
    Constructing one here from four strings alone is not possible.

    Sign convention
    ---------------
    CME quotes the equity-index calendar spread as BACK minus FRONT, so the
    implied leg differential must be built the same way. Verified against
    ESM6-ESU6 on 2026-06-09 16:00 UTC: front 7312.25/7312.50, back
    7372.75/7373.25, spread quoted 60.75/60.80 — i.e. back - front = +61,
    not front - back = -61.

    Getting this backwards does not merely flip the sign of the answer: the
    implied and quoted legs then have opposite signs, so differencing them
    yields roughly *twice the spread level* (~120 points, ~$6,000) instead of
    a residual of a tick or two. Any cost far above a few ticks means this
    convention has drifted out of line with the venue's quote — check it
    before trusting the P&L.
    """

    def __init__(self, spec: CalendarSpreadContractSpec):
        """
        Args:
            spec: Static contract parameters for the calendar spread (#7).
        """
        self.spec = spec
        self.outright_point_value = float(spec.contract_multiplier)
        self.spread_point_value = float(spec.contract_multiplier)

    def get_cost(
        self,
        symbol_front_snapshot: pd.Series,
        symbol_back_snapshot: pd.Series,
        symbol_spread_snapshot: pd.Series,
        trade_side: str,
    ) -> Optional[float]:
        """
        Computes the instantaneous mechanical dollar cost of crossing the outright books.

        Parameters:
        - symbol_front_snapshot: Top of book series for the front month leg.
        - symbol_back_snapshot: Top of book series for the back month leg.
        - symbol_spread_snapshot: Top of book series for the spread instrument.
        - trade_side: 'A' (Ask/Seller-initiated) or 'B' (Bid/Buyer-initiated) from the trade frame.

        Returns:
        - The cost in dollars (positive = loss/cost, negative = profit), or
          ``None`` if the aggressor side is unknown ('N') or any required
          top-of-book price is missing (empty level -> NaN).
        """
        # Extract top-of-book (Level 00) prices
        front_bid = symbol_front_snapshot["bid_px_00"]
        front_ask = symbol_front_snapshot["ask_px_00"]
        back_bid = symbol_back_snapshot["bid_px_00"]
        back_ask = symbol_back_snapshot["ask_px_00"]
        spread_bid = symbol_spread_snapshot["bid_px_00"]
        spread_ask = symbol_spread_snapshot["ask_px_00"]

        if trade_side == "A":
            # Aggressor hits the Spread Bid -> Market Maker is passively LONG the spread.
            # Long the spread = long the back leg, short the front leg (see the
            # sign convention above). To flatten, MM synthetically SELLS the
            # spread: Sell Back (hit its bid), Buy Front (lift its ask).
            if pd.isna(back_bid) or pd.isna(front_ask) or pd.isna(spread_bid):
                return None
            implied_bid = float(back_bid) - float(front_ask)

            # Convert points to dollar value using the spec's multiplier
            implied_bid_usd = implied_bid * self.outright_point_value
            spread_bid_usd = float(spread_bid) * self.spread_point_value

            # If there's a cost incurred through the hedge, cost_usd is positive.
            # Passively LONG the spread is a cost while SELL the spread to hedge is a benefit.
            cost_usd = spread_bid_usd - implied_bid_usd

        elif trade_side == "B":
            # Aggressor lifts the Spread Ask -> Market Maker is passively SHORT the spread.
            # To flatten, MM synthetically BUYS the spread: Buy Back (lift its
            # ask), Sell Front (hit its bid).
            if pd.isna(back_ask) or pd.isna(front_bid) or pd.isna(spread_ask):
                return None
            implied_ask = float(back_ask) - float(front_bid)

            # Convert points to dollar value using the spec's multiplier
            implied_ask_usd = implied_ask * self.outright_point_value
            spread_ask_usd = float(spread_ask) * self.spread_point_value

            # If there's a cost incurred through the hedge, cost_usd is positive.
            # Passively SHORT the spread is a benefit while LONG the spread to hedge is a cost.
            cost_usd = implied_ask_usd - spread_ask_usd

        else:
            # Unknown trade side
            return None

        return float(cost_usd)
