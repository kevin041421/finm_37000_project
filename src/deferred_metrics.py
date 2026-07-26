"""Quantitative metrics for validating the deferred book (Issue #15).

Pure functions over aligned top-of-book DataFrames. The goal is to separate
direct vs. implied liquidity and track microstructural price divergences.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from implied_common import align_books, top_of_book


def decompose_liquidity(
    implied: pd.DataFrame, 
    native: pd.DataFrame, 
    tick_size: float
) -> pd.DataFrame:
    """Decompose native depth into direct and implied components.

    Args:
        implied: The implied deferred book (output of implied_back_book).
        native: The raw Databento deferred book.
        tick_size: The outright tick size for gap calculation.

    Returns a time-aligned DataFrame with the size breakdown and gap metrics.
    """
    aligned = align_books(imp=implied, nat=top_of_book(native))
    imp, nat = aligned["imp"], aligned["nat"]

    # Match criteria (using a tiny tolerance for float safety)
    bid_match = np.isclose(imp["bid_px"], nat["bid_px"], atol=1e-7)
    ask_match = np.isclose(imp["ask_px"], nat["ask_px"], atol=1e-7)

    # Price superiority checks
    bid_nat_better = nat["bid_px"] > imp["bid_px"]
    ask_nat_better = nat["ask_px"] < imp["ask_px"]

    # Size accounting: 
    # - If prices match, CME combines the resting liquidity -> Direct = Native - Implied. 
    # - If native is tighter, the quote is organically driven -> 100% Direct.
    # - If implied is tighter, direct size at the native touch is 0.
    bid_direct_sz = np.where(
        bid_match, 
        np.maximum(0.0, nat["bid_sz"] - imp["bid_sz"]), 
        np.where(bid_nat_better, nat["bid_sz"], 0.0)
    )
    
    ask_direct_sz = np.where(
        ask_match, 
        np.maximum(0.0, nat["ask_sz"] - imp["ask_sz"]), 
        np.where(ask_nat_better, nat["ask_sz"], 0.0)
    )

    return pd.DataFrame(
        {
            "imp_bid_px": imp["bid_px"],
            "nat_bid_px": nat["bid_px"],
            "bid_match": bid_match,
            "bid_gap_ticks": (nat["bid_px"] - imp["bid_px"]) / tick_size,
            "nat_bid_sz": nat["bid_sz"],
            "imp_bid_sz": imp["bid_sz"],
            "dir_bid_sz": bid_direct_sz,
            
            "imp_ask_px": imp["ask_px"],
            "nat_ask_px": nat["ask_px"],
            "ask_match": ask_match,
            "ask_gap_ticks": (imp["ask_px"] - nat["ask_px"]) / tick_size,
            "nat_ask_sz": nat["ask_sz"],
            "imp_ask_sz": imp["ask_sz"],
            "dir_ask_sz": ask_direct_sz,
            
            "is_divergent": ~(bid_match & ask_match)
        },
        index=aligned["imp"].index
    )


def divergence_episodes(decomp: pd.DataFrame) -> pd.DataFrame:
    """Group contiguous divergence ticks into discrete episodes.
    
    A divergence is a moment where the modeled touch detaches from the real 
    touch. This function measures how long the implied engine or arbitrageurs 
    take to snap the books back in line.
    """
    is_div = decomp["is_divergent"]
    if not is_div.any():
        return pd.DataFrame()

    # Create a unique group ID for contiguous blocks of identical states
    group_id = (is_div != is_div.shift()).cumsum()
    
    # Filter to only the divergent blocks
    div_blocks = decomp[is_div].groupby(group_id)

    rows = []
    for _, group in div_blocks:
        start_ts = group.index[0]
        end_ts = group.index[-1]
        
        # Duration in milliseconds
        dur_ms = (end_ts - start_ts).total_seconds() * 1000.0 if len(group) > 1 else 0.0
        
        # Max gap magnitude across the episode
        max_bid_gap = group["bid_gap_ticks"].abs().max()
        max_ask_gap = group["ask_gap_ticks"].abs().max()

        rows.append({
            "start_time": start_ts,
            "end_time": end_ts,
            "ticks": len(group),
            "duration_ms": dur_ms,
            "max_gap_ticks": max(max_bid_gap, max_ask_gap)
        })

    return pd.DataFrame(rows)


def validation_summary(decomp: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    """Aggregate the decomposition into the final validation metrics."""
    touch_match_pct = (~decomp["is_divergent"]).mean() * 100.0
    
    # Combine bid/ask to evaluate total top-of-book depth
    avg_nat_depth = (decomp["nat_bid_sz"] + decomp["nat_ask_sz"]).mean() / 2.0
    avg_imp_depth = (decomp["imp_bid_sz"] + decomp["imp_ask_sz"]).mean() / 2.0
    avg_dir_depth = (decomp["dir_bid_sz"] + decomp["dir_ask_sz"]).mean() / 2.0
    
    pct_implied = min(100.0, (avg_imp_depth / avg_nat_depth * 100.0)) if avg_nat_depth > 0 else 0.0

    # Divergence stats
    if not episodes.empty:
        avg_dur_ms = episodes["duration_ms"].mean()
        max_dur_ms = episodes["duration_ms"].max()
        max_gap = episodes["max_gap_ticks"].max()
    else:
        avg_dur_ms = max_dur_ms = max_gap = 0.0

    return {
        "updates": len(decomp),
        "touch_match_pct": touch_match_pct,
        "avg_nat_depth": avg_nat_depth,
        "avg_imp_depth": avg_imp_depth,
        "avg_dir_depth": avg_dir_depth,
        "pct_implied": pct_implied,
        "pct_direct": 100.0 - pct_implied,
        "divergence_count": len(episodes),
        "avg_divergence_ms": avg_dur_ms,
        "max_divergence_ms": max_dur_ms,
        "max_gap_ticks": max_gap
    }