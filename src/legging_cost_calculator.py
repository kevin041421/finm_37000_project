"""LeggingCostCalculator: cost of hedging a spread fill in the outrights (Issue #8)."""

import pandas as pd
from market_data_fetcher import CalendarSpreadContractSpec

class LeggingCostCalculator:
    """Computes the mechanical cost of legging out a spread fill (Issue #8)."""

    def __init__(self, product_code: str, front_symbol: str, back_symbol: str, spread_symbol: str):
        """
        Initializes the calculator with the static contract parameters required 
        to normalize point differentials into standardized dollar metrics.
        """
        self.spec = CalendarSpreadContractSpec(product_code, front_symbol, back_symbol, spread_symbol)

    def get_cost(
        self,
        symbol_front_snapshot: pd.Series,
        symbol_back_snapshot: pd.Series,
        symbol_spread_snapshot: pd.Series,
        trade_side: str
    ) -> float:
        """
        Computes the instantaneous mechanical dollar cost of crossing the outright books.
        
        Parameters:
        - symbol_front_snapshot: Top of book series for the front month leg.
        - symbol_back_snapshot: Top of book series for the back month leg.
        - symbol_spread_snapshot: Top of book series for the spread instrument.
        - trade_side: 'A' (Ask/Seller-initiated) or 'B' (Bid/Buyer-initiated) from the trade frame.
        
        Returns:
        - float_number: The cost in dollars (positive = loss/cost, negative = profit).
        """
        # Extract top-of-book (Level 00) prices
        front_bid = symbol_front_snapshot["bid_px_00"]
        front_ask = symbol_front_snapshot["ask_px_00"]
        back_bid = symbol_back_snapshot["bid_px_00"]
        back_ask = symbol_back_snapshot["ask_px_00"]
        spread_bid = symbol_spread_snapshot["bid_px_00"]
        spread_ask = symbol_spread_snapshot["ask_px_00"]

        if trade_side == 'A':
            # Aggressor hits the Spread Bid -> Market Maker is passively LONG the spread.
            # To flatten, MM must synthetically SELL the spread: Sell Back (hit bid), Buy Front (pay ask).
            synthetic_trade = front_ask - back_bid
            
            # Convert points to dollar value using the spec's point_value fields
            synthetic_trade_usd = synthetic_trade * self.spec.outright_point_value
            spread_bid_usd = spread_bid * self.spec.spread_point_value
            
            # If there's a cost incurred through the hedge, cost_usd is positive.
            # Given the Market Maker is passively long the spread, spread_bid_usd is a cost for Market Maker and thus the sign before spread_bid_usd is positive.
            cost_usd = synthetic_trade_usd + spread_bid_usd

        elif trade_side == 'B':
            # Aggressor hits the Spread Ask -> Market Maker is passively SHORT the spread.
            # To flatten, MM must synthetically LONG the spread: Buy Back (pay ask), Sell Front (hit bid).
            synthetic_trade = back_ask - front_bid
            
            # Convert points to dollar value using the spec's point_value fields
            synthetic_trade_usd = synthetic_trade * self.spec.outright_point_value
            spread_ask_usd = spread_ask * self.spec.spread_point_value
            
            # If there's a cost incurred through the hedge, cost_usd is positive.
            # Given the Market Makee is passively SHORT the spread, spread_ask_usd is a benefit for Market Maker ans thus the sign before spreae_bid_usd is negative.
            cost_usd = synthetic_trade_usd - spread_ask_usd
            
        else:
            # Unknown trade side
            return None

        return float(cost_usd)
