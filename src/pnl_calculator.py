"""PnLCalculator: expected P&L engine for calendar-spread market making (Issue #9).

Streaming design — data is consumed one spread transaction at a time, so it
works identically whether the driver iterates a list of daily
``CalendarSpreadData`` or the memory-bounded ``iter_calendar_spread_data``
generator:

    calc = PnLCalculator()
    cost_calc = LeggingCostCalculator()          # Issue #8
    for datum in fetcher.iter_calendar_spread_data(...):
        replay(datum, cost_calc, calc)
    pnl_of_the_period = calc.generate_pnl()

Components:
- ``OrderBook``      — as-of book state lookup for one instrument (for
                       ad-hoc queries and analysis).
- ``replay``         — driver: merges book updates and spread trades of one
                       ``CalendarSpreadData`` into a single time-ordered
                       event stream and walks it once. Book events refresh
                       the cached current state; each trade event gets the
                       legging cost (#8) and is fed to the accumulator with
                       the three cached snapshots.
- ``PnLCalculator``  — stateful accumulator: ``consume`` one transaction at
                       a time, ``generate_pnl`` aggregates at the end.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from legging_cost_calculator import LeggingCostCalculator
from market_data_fetcher import BOOK_INDEX_NAME, CalendarSpreadData


class OrderBook:
    """Time-series order book for one instrument with as-of state lookup.

    Wraps a book frame (see ``market_data_fetcher`` schema): ``ts_event``
    UTC index, ``bid/ask_px/sz_{level}`` columns.
    """

    def __init__(self, symbol: str, book: pd.DataFrame) -> None:
        if not isinstance(book.index, pd.DatetimeIndex) or book.index.tz is None:
            raise ValueError(
                f"{symbol}: book index must be a tz-aware DatetimeIndex "
                f"named {BOOK_INDEX_NAME!r}"
            )
        if not book.index.is_monotonic_increasing:
            raise ValueError(f"{symbol}: book index must be ascending")
        if "bid_px_00" not in book.columns or "ask_px_00" not in book.columns:
            raise ValueError(f"{symbol}: book frame missing top-of-book columns")
        self.symbol = symbol
        self._book = book

    def state_at(self, ts: pd.Timestamp | str) -> Optional[pd.Series]:
        """Book state as of ``ts``: the most recent row at or before ``ts``.

        Returns ``None`` if the book has no state yet (``ts`` earlier than
        the first event).
        """
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        pos = self._book.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        return self._book.iloc[pos]

    def best_bid(self, ts: pd.Timestamp | str) -> Optional[float]:
        """Best bid price as of ``ts`` (None if no state yet)."""
        state = self.state_at(ts)
        return None if state is None else state["bid_px_00"]

    def best_ask(self, ts: pd.Timestamp | str) -> Optional[float]:
        """Best ask price as of ``ts`` (None if no state yet)."""
        state = self.state_at(ts)
        return None if state is None else state["ask_px_00"]

    def mid(self, ts: pd.Timestamp | str) -> Optional[float]:
        """Mid price as of ``ts`` (None if no state yet)."""
        state = self.state_at(ts)
        if state is None:
            return None
        return (state["bid_px_00"] + state["ask_px_00"]) / 2.0

    def __len__(self) -> int:
        return len(self._book)


@dataclass(frozen=True)
class SpreadTransaction:
    """One spread trade with the market context the P&L model needs."""

    ts: pd.Timestamp
    price: float
    size: float
    side: str  # aggressor: "B" buyer-initiated, "A" seller-initiated, "N" unknown
    front_snapshot: pd.Series
    back_snapshot: pd.Series
    spread_snapshot: pd.Series
    cost: float  # legging cost ($ per spread contract) from Issue #8's calculator


class PnLCalculator:
    """Stateful expected-P&L accumulator for spread transactions (Issue #9).

    Model parameters (set by the user):
      - ``p``: probability of being at the head of the best bid/ask queue at
        any moment. Applied per trade as a Bernoulli draw: each historical
        spread trade fills us *in full* with probability ``p`` (and not at
        all otherwise). One replay is therefore a single random sample of
        the P&L — run several seeds and average to estimate the expected
        P&L. Pass ``seed`` for reproducibility.
      - ``max_position``: maximum absolute spread position ``m`` we are
        willing to hold. When a fill would push us beyond ``m``, we still
        take the fill (we were resting in the queue) but immediately flatten
        the excess by legging the outrights, paying the #8 legging cost plus
        the aggressive fee on the hedged contracts.
      - ``passive_fee`` / ``aggressive_fee``: $/contract. ``passive_fee`` is
        charged on every passively filled contract (negative = rebate);
        ``aggressive_fee`` is charged per spread contract legged out (use it
        to reflect both outright legs' fees).
      - ``contract_multiplier``: $ per point (ES: 50) — converts point-based
        trade/mark prices into dollars (legging costs from #8 arrive already
        in dollars).

    Side convention (aggressor side of the trade): ``"B"`` = buyer lifted
    the ask, so we (passive) SOLD at the trade price; ``"A"`` = seller hit
    the bid, so we BOUGHT. ``"N"`` (unknown) trades are skipped.

    Accounting is cash-based: fills move cash at the trade price, the
    residual position is marked at the last seen spread mid in
    ``generate_pnl`` — so spread-capture edge and inventory moves are both
    captured without separate bookkeeping.
    """

    def __init__(
        self,
        p: float,
        max_position: float,
        passive_fee: float = 0.0,
        aggressive_fee: float = 0.0,
        contract_multiplier: float = 50.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if max_position <= 0:
            raise ValueError(f"max_position must be > 0, got {max_position}")
        self.p = p
        self._rng = random.Random(seed)
        self.max_position = max_position
        self.passive_fee = passive_fee
        self.aggressive_fee = aggressive_fee
        self.contract_multiplier = contract_multiplier

        self._transactions: list[SpreadTransaction] = []
        self._position = 0.0        # signed spread contracts
        self._cash = 0.0            # $ from fills at trade prices
        self._passive_fees = 0.0    # $ paid on passive fills (neg = rebates earned)
        self._hedge_costs = 0.0     # $ legging cost + aggressive fees on hedged qty
        self._filled_qty = 0.0      # contracts filled passively
        self._hedged_qty = 0.0      # contracts immediately legged out
        self._skipped = 0           # trades with unknown aggressor side
        self._last_spread_mid: Optional[float] = None

    def consume(
        self,
        front_snapshot: pd.Series,
        back_snapshot: pd.Series,
        spread_snapshot: pd.Series,
        trade: pd.Series,
        cost: float,
    ) -> None:
        """Process one spread transaction (full fill with probability ``p``).

        Args:
            front_snapshot / back_snapshot / spread_snapshot: book-frame rows
                as of the trade's timestamp.
            trade: trade-frame row (``price``, ``size``, ``side``; its
                ``name`` is the event timestamp).
            cost: legging cost in dollars per spread contract from the #8
                calculator, for the contracts hedged at the position cap.
        """
        self._transactions.append(
            SpreadTransaction(
                ts=trade.name,
                price=float(trade["price"]),
                size=float(trade["size"]),
                side=str(trade["side"]),
                front_snapshot=front_snapshot,
                back_snapshot=back_snapshot,
                spread_snapshot=spread_snapshot,
                cost=float(cost),
            )
        )
        self._last_spread_mid = (
            spread_snapshot["bid_px_00"] + spread_snapshot["ask_px_00"]
        ) / 2.0

        side = str(trade["side"])
        if side == "B":
            direction = -1.0  # buyer aggressed: we sold
        elif side == "A":
            direction = +1.0  # seller aggressed: we bought
        else:
            self._skipped += 1
            return

        # Bernoulli fill: the trade fills us in full with probability p
        if self._rng.random() >= self.p:
            return  # not at the head of the queue for this trade

        price = float(trade["price"])
        q = float(trade["size"])
        if q == 0.0:
            return

        # position cap: hedge the part that would exceed max_position
        new_pos = self._position + direction * q
        excess = max(0.0, abs(new_pos) - self.max_position)
        hedged = min(q, excess)
        retained = q - hedged

        # all q contracts were filled passively -> passive fee/rebate
        self._filled_qty += q
        self._passive_fees += self.passive_fee * q

        # retained contracts change position and cash at the trade price
        self._position += direction * retained
        self._cash += -direction * price * self.contract_multiplier * retained

        # hedged contracts are legged out immediately: lose the legging cost
        # ($/contract, from #8) plus the aggressive fee, no position behind
        if hedged > 0.0:
            self._hedged_qty += hedged
            self._hedge_costs += (cost + self.aggressive_fee) * hedged

    @property
    def transaction_count(self) -> int:
        return len(self._transactions)

    @property
    def position(self) -> float:
        return self._position

    def generate_pnl(self) -> pd.Series:
        """Aggregate consumed transactions into the period's P&L (one sample).

        Returns a summary Series. ``net_pnl`` = cash from fills + residual
        position marked at the last seen spread mid - passive fees - hedge
        costs. Fills are random (probability ``p`` per trade), so this is
        one sample path — average ``net_pnl`` over several seeds to estimate
        the expected P&L that Issue #10 holds against zero.
        """
        mark_price = self._last_spread_mid if self._last_spread_mid is not None else 0.0
        position_mark = self._position * mark_price * self.contract_multiplier
        net_pnl = self._cash + position_mark - self._passive_fees - self._hedge_costs
        return pd.Series(
            {
                "transactions": float(self.transaction_count),
                "skipped_unknown_side": float(self._skipped),
                "filled_contracts": self._filled_qty,
                "hedged_contracts": self._hedged_qty,
                "final_position": self._position,
                "cash": self._cash,
                "position_mark": position_mark,
                "passive_fees": self._passive_fees,
                "hedge_costs": self._hedge_costs,
                "net_pnl": net_pnl,
            }
        )


def _merged_events(data: CalendarSpreadData):
    """Yield all events (book updates + spread trades) in time order.

    Each event is ``(ts, priority, kind, payload)``. At equal timestamps,
    book updates are processed before trades (priority 0 < 1): a trade
    executed against the book state stamped at the same instant.
    """
    streams = [
        ("front", 0, data.front),
        ("back", 0, data.back),
        ("spread", 0, data.spread),
        ("trade", 1, data.spread_trades),
    ]
    events = []
    for kind, priority, frame in streams:
        for ts, row in frame.iterrows():
            events.append((ts, priority, kind, row))
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def replay(
    data: CalendarSpreadData,
    cost_calculator: LeggingCostCalculator,
    pnl_calculator: PnLCalculator,
) -> int:
    """Drive one ``CalendarSpreadData`` through the P&L pipeline.

    Event-driven replay: all book updates and spread trades are merged into
    one time-ordered stream and walked once (O(n + m)). Book updates refresh
    the cached "current state" per instrument; each spread trade triggers a
    legging-cost query (#8) and is fed to ``pnl_calculator.consume`` with
    the three cached snapshots. Trades that occur before all three books
    have a state are skipped.

    Returns the number of transactions consumed.
    """
    books: dict[str, Optional[pd.Series]] = {"front": None, "back": None, "spread": None}

    consumed = 0
    for ts, _priority, kind, payload in _merged_events(data):
        if kind != "trade":
            books[kind] = payload  # update cached book state
            continue

        if books["front"] is None or books["back"] is None or books["spread"] is None:
            continue  # a book has no state yet at this timepoint

        cost = cost_calculator.get_cost(
            books["front"], books["back"], books["spread"], str(payload["side"])
        )
        pnl_calculator.consume(
            books["front"], books["back"], books["spread"], payload, cost
        )
        consumed += 1
    return consumed
