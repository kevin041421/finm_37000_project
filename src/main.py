"""Fetch ESM6, ESU6, and their calendar spread instrument data from Databento."""

import databento as db
import pandas as pd
from util import get_databento_api_key

CME = "GLBX.MDP3"
START_DATE = "2026-01-01"

LEG_SYMBOLS   = ["ESM6", "ESU6"]
PARENT_SYMBOL = "ES.FUT"


def fetch_leg_definitions(client: db.Historical) -> pd.DataFrame:
    """Fetch instrument definitions for ESM6 and ESU6."""
    dbn = client.timeseries.get_range(
        dataset=CME,
        schema="definition",
        symbols=LEG_SYMBOLS,
        start=START_DATE,
    )
    df = dbn.to_df()
    return df


def fetch_spread_definition(client: db.Historical) -> pd.DataFrame:
    """Fetch definition for the ES calendar spread instruments (all classes)."""
    dbn = client.timeseries.get_range(
        dataset=CME,
        schema="definition",
        symbols=PARENT_SYMBOL,
        stype_in="parent",
        start=START_DATE,
    )
    df = dbn.to_df()
    # Keep only calendar spread instruments (legs M6 and U6)
    spreads = df[df["instrument_class"] == db.InstrumentClass.FUTURE_SPREAD].copy()
    mask = spreads["raw_symbol"].str.contains("ESM6") & spreads["raw_symbol"].str.contains("ESU6")
    return spreads[mask]



def fetch_orderbook_snapshot(
    client: db.Historical,
    symbols: list[str],
    snapshot_ts: str,
    levels: int = 10,
) -> pd.DataFrame:
    """Fetch a single MBP-10 orderbook snapshot for the given symbols.

    Args:
        client: Databento Historical client.
        symbols: List of instrument symbols (raw) to query.
        snapshot_ts: ISO timestamp string for the start of the 1-second window.
        levels: Number of price levels (max 10 for mbp-10 schema).

    Returns:
        DataFrame with bid/ask price and size at each level.
    """
    end_ts = (
        pd.Timestamp(snapshot_ts, tz="UTC") + pd.Timedelta(seconds=1)
    ).isoformat()

    schema = f"mbp-{min(levels, 10)}"
    dbn = client.timeseries.get_range(
        dataset=CME,
        schema=schema,
        symbols=symbols,
        start=snapshot_ts,
        end=end_ts,
    )
    df = dbn.to_df()
    if df.empty:
        return df

    # Take the first message per symbol as the snapshot
    df = df.groupby("symbol").first().reset_index()

    rows = []
    for _, row in df.iterrows():
        sym = row["symbol"]
        for lvl in range(levels):
            bid_px  = row.get(f"bid_px_{lvl:02d}", float("nan"))
            ask_px  = row.get(f"ask_px_{lvl:02d}", float("nan"))
            bid_sz  = row.get(f"bid_sz_{lvl:02d}", float("nan"))
            ask_sz  = row.get(f"ask_sz_{lvl:02d}", float("nan"))
            rows.append(
                {
                    "symbol": sym,
                    "level": lvl,
                    "bid_size": bid_sz,
                    "bid_price": bid_px,
                    "ask_price": ask_px,
                    "ask_size": ask_sz,
                }
            )
    return pd.DataFrame(rows)


def print_orderbook(ob: pd.DataFrame) -> None:
    """Pretty-print all symbol orderbooks side by side, asks above bids."""
    if ob.empty:
        print("  (no data)")
        return

    books = {sym: grp.set_index("level") for sym, grp in ob.groupby("symbol")}
    symbols = list(books.keys())
    levels  = sorted(ob["level"].unique())

    col_w = 24  # width per symbol block
    sep = "     "  # 5 spaces between books
    # Header row
    print("  " + sep.join(f"{sym:^{col_w}}" for sym in symbols))
    print("  " + sep.join(f"{'Lvl':>4}  {'Sz':>8}  {'Px':>10}" for _ in symbols))
    print("  " + sep.join(f"{'-'*4}  {'-'*8}  {'-'*10}" for _ in symbols))

    # Asks: level 9 → 0 (descending)
    for lvl in sorted(levels, reverse=True):
        cols = []
        for sym in symbols:
            r = books[sym].loc[lvl]
            cols.append(f"{lvl:>4}  {r['ask_size']:>8.0f}  {r['ask_price']:>10.2f}")
        print("  " + sep.join(cols))

    # Divider
    print("  " + sep.join(f"{'----':>4}  {'--------':>8}  {'----------':>10}" for _ in symbols))

    # Bids: level 0 → 9 (ascending)
    for lvl in sorted(levels):
        cols = []
        for sym in symbols:
            r = books[sym].loc[lvl]
            cols.append(f"{lvl:>4}  {r['bid_size']:>8.0f}  {r['bid_price']:>10.2f}")
        print("  " + sep.join(cols))


def main():
    client = db.Historical(get_databento_api_key())

    print("=== Leg definitions (ESM6, ESU6) ===")
    leg_defs = fetch_leg_definitions(client)
    print(leg_defs[["raw_symbol", "instrument_id", "instrument_class",
                     "expiration", "min_price_increment", "unit_of_measure_qty"]].to_string())

    print("\n=== Calendar spread definitions (ESM6–ESU6) ===")
    spread_defs = fetch_spread_definition(client)
    print(spread_defs[["raw_symbol", "instrument_id", "instrument_class",
                        "expiration", "min_price_increment"]].to_string())

    spread_symbols = spread_defs["raw_symbol"].unique().tolist()

    # --- Orderbook snapshot ---
    # Use 10:00 AM CT (16:00 UTC) on 2026-06-10 as a representative moment
    SNAPSHOT_TS = "2026-06-10T16:00:00+00:00"
    all_symbols = LEG_SYMBOLS + spread_symbols

    print(f"\n=== Orderbook snapshot @ {SNAPSHOT_TS} ===")
    ob = fetch_orderbook_snapshot(client, all_symbols, SNAPSHOT_TS)
    print_orderbook(ob)

    return {
        "leg_defs": leg_defs,
        "spread_defs": spread_defs,
        "orderbook": ob,
    }


if __name__ == "__main__":
    data = main()
