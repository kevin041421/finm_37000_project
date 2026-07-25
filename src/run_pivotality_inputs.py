"""Generate the P&L sample set that Issue #10 consumes (producer for #9 -> #10).

Issue #10 (Exchange Rebate Pivotality Analysis) needs the P&L engine's output
across many model configurations, not a single run. ``RebatePivotalityAnalyzer``
ingests ``List[pd.Series]`` of ``generate_pnl()`` output and averages the
Bernoulli seeds, optionally taking a second, higher-``p`` set to represent DMM
priority matching. This script produces exactly that, on disk, so #10 can be
developed and re-run without touching Databento.

Why a sweep rather than one run
-------------------------------
``PnLCalculator`` fills passively with probability ``p`` per trade, so a single
replay is one random sample of the P&L, not the expectation. Averaging over
``N_SEEDS`` estimates the expectation. Sweeping ``p`` and ``max_position`` on
top of that gives #10 the surface it needs: the required rebate depends on how
often you are at the head of the queue and on how much inventory you tolerate
before legging out.

Break-even rebate
-----------------
Net P&L is linear in the passive fee, since that fee is charged per passively
filled contract and nothing else in the model depends on it:

    net(f) = cash + position_mark - f * filled - hedge_costs

Setting net = 0 and solving gives the passive fee at which the strategy breaks
even:

    breakeven_passive_fee = (cash + position_mark - hedge_costs) / filled

``required_rebate_per_contract`` is then the distance from the fee actually
charged. Positive means the exchange must give back that much per contract; if
it exceeds the current fee, a discount is not enough and a true rebate (a
negative fee) is required. Note ``hedge_costs`` retains the aggressive fee on
legged-out contracts, which no passive rebate offsets. This matches Scenario D
in ``rebate_pivotality_analyzer.py`` and is computed here only so the summary
file is directly readable; #10 remains the authority on incentive analysis.

Output (written to ``results/pivotality/``)
-------------------------------------------
``pnl_samples.csv``    one row per (session, p, max_position, seed) with every
                       ``generate_pnl()`` field. This is the file #10 reads and
                       groups into the ``List[pd.Series]`` its API expects.
``daily_summary.csv``  one row per (session, p, max_position): mean and std
                       across seeds, plus the break-even fee and rebate.
``run_manifest.json``  parameters, resolved symbols, contract spec, fee tier
                       and data windows — so a sample file is reproducible.

Run:
    uv run src/run_pivotality_inputs.py

Reads from the disk cache; only an unseeded cache needs ~/.databento_api_key.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cached_market_data_fetcher import CachedMarketDataFetcher
from fee_schedule import get_fee_rates
from legging_cost_calculator import LeggingCostCalculator
from pnl_calculator import PnLCalculator, consume_all, extract_transactions

# ---------------------------------------------------------------------- #
# Instrument and sessions — the windows already in data_cache/, so this
# script runs offline. Widen these once a longer fetch is affordable.
# ---------------------------------------------------------------------- #
PRODUCT = "ES"
FRONT_MONTH = "M6"
DAYS = ["2026-06-09", "2026-06-10"]
SESSION_START_UTC = "16:00:00"
SESSION_LENGTH = pd.Timedelta(seconds=300)
BOOK_LEVELS = 1

# ---------------------------------------------------------------------- #
# Sweep grid
# ---------------------------------------------------------------------- #
#: Probability of being at the head of the queue. The low end is a realistic
#: non-privileged market maker; 1.0 is the upper bound where every trade fills
#: us. #10 reads the high-p cells as the DMM priority-matching flow.
P_GRID = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)

#: Spread contracts we tolerate before legging out the excess. Drives the
#: trade-off between paying legging costs and carrying inventory risk.
MAX_POSITION_GRID = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0)

#: Seeds per grid cell. Each is one Bernoulli sample path; #10 averages them.
N_SEEDS = 50
BASE_SEED = 1000

FEE_TIER = "non_member"

#: Sentinel session label for the pooled run across every day, where a single
#: calculator carries inventory from one session into the next (as
#: run_pnl_pipeline.py does). Not the same as summing the per-day rows.
POOLED_LABEL = "ALL"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "pivotality"


#: Identifier columns in pnl_samples.csv that are not metrics.
_KEY_COLUMNS = ["session", "seed"]


def load_samples(
    session: str = POOLED_LABEL,
    p_queue_head: float | None = None,
    max_position: float | None = None,
    path: Path | None = None,
) -> list[pd.Series]:
    """Load one grid cell from ``pnl_samples.csv`` as Issue #10 expects it.

    ``RebatePivotalityAnalyzer.generate_pivotality_report`` takes a
    ``List[pd.Series]`` and averages it with ``pd.DataFrame(samples).mean()``,
    which raises ``TypeError`` on any non-numeric column. The raw CSV rows
    carry a string ``session`` label, so they cannot be passed through
    directly — this drops the identifier columns and returns numeric-only
    Series.

    ``p_queue_head`` and ``max_position`` are deliberately kept: they are
    numeric and constant within a cell, so the averaged Series still records
    which configuration it describes.

    Args:
        session: a day from ``DAYS``, or ``POOLED_LABEL`` for the pooled run.
        p_queue_head / max_position: grid cell to select; ``None`` means do
            not filter on that axis.
        path: override the samples file location.

    Example — the two arguments #10's report takes::

        std = load_samples(p_queue_head=0.50, max_position=5.0)
        dmm = load_samples(p_queue_head=1.00, max_position=5.0)
        RebatePivotalityAnalyzer(1.30, 2.76).generate_pivotality_report(std, dmm)
    """
    path = path or (OUTPUT_DIR / "pnl_samples.csv")
    df = pd.read_csv(path)
    df = df[df["session"] == session]
    if p_queue_head is not None:
        df = df[df["p_queue_head"] == p_queue_head]
    if max_position is not None:
        df = df[df["max_position"] == max_position]
    if df.empty:
        raise ValueError(
            f"No samples for session={session!r}, p_queue_head={p_queue_head}, "
            f"max_position={max_position} in {path}"
        )
    df = df.drop(columns=_KEY_COLUMNS)
    return [row for _, row in df.iterrows()]


def _breakeven(row: pd.Series, passive_fee: float) -> tuple[float, float]:
    """Passive fee giving net P&L = 0, and the rebate that implies.

    Returns ``(nan, nan)`` when nothing filled: with no passive volume there
    is no per-contract lever, so break-even is undefined rather than zero.
    """
    filled = row["filled_contracts"]
    if filled <= 0:
        return float("nan"), float("nan")
    be_fee = (row["cash"] + row["position_mark"] - row["hedge_costs"]) / filled
    return be_fee, passive_fee - be_fee


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fetcher = CachedMarketDataFetcher.from_databento_key(levels=BOOK_LEVELS)
    spec = fetcher.fetch_calendar_spread_contract_specification(FRONT_MONTH, PRODUCT)
    contract_multiplier = float(spec.contract_multiplier)
    fees = get_fee_rates(spec.product_code, FEE_TIER)
    cost_calculator = LeggingCostCalculator(spec)

    print(f"{spec.spread_symbol}  ({spec.front_symbol} / {spec.back_symbol})")

    # ---- Extract each session's transactions once. -------------------- #
    # Independent of the model parameters, so the ~60k-event merge per day
    # happens once rather than once per grid cell.
    sessions: dict[str, list] = {}
    windows: dict[str, dict[str, str]] = {}
    for day in DAYS:
        t0 = pd.Timestamp(f"{day}T{SESSION_START_UTC}", tz="UTC")
        t1 = t0 + SESSION_LENGTH
        windows[day] = {"start": t0.isoformat(), "end": t1.isoformat()}
        print(f"  {day} loading...", flush=True)

        data = fetcher.fetch_calendar_spread_data(FRONT_MONTH, PRODUCT, t0, t1)
        sessions[day] = extract_transactions(data, cost_calculator)
        print(
            f"    books front={len(data.front)} back={len(data.back)} "
            f"spread={len(data.spread)} -> {len(sessions[day])} transaction(s)"
        )

    n_cells = len(P_GRID) * len(MAX_POSITION_GRID) * N_SEEDS * (len(DAYS) + 1)
    print(f"\nsweeping {n_cells} configurations...", flush=True)

    # ---- Replay the grid. --------------------------------------------- #
    rows: list[dict] = []
    for p in P_GRID:
        for max_position in MAX_POSITION_GRID:
            for i in range(N_SEEDS):
                seed = BASE_SEED + i

                def run(label: str, streams: list[list]) -> None:
                    """One calculator over one or more sessions, in order."""
                    calc = PnLCalculator(
                        p=p,
                        max_position=max_position,
                        passive_fee=fees.passive_fee,
                        aggressive_fee=fees.aggressive_fee,
                        contract_multiplier=contract_multiplier,
                        seed=seed,
                    )
                    for stream in streams:
                        consume_all(stream, calc)
                    summary = calc.generate_pnl()
                    be_fee, rebate = _breakeven(summary, fees.passive_fee)
                    rows.append(
                        {
                            "session": label,
                            "seed": seed,
                            **summary.to_dict(),
                            "breakeven_passive_fee": be_fee,
                            "required_rebate_per_contract": rebate,
                        }
                    )

                for day in DAYS:
                    run(day, [sessions[day]])
                # pooled: inventory carries across sessions
                run(POOLED_LABEL, [sessions[d] for d in DAYS])

    samples = pd.DataFrame(rows)
    key = ["session", "p_queue_head", "max_position"]
    metrics = [c for c in samples.columns if c not in key + ["seed"]]

    summary = samples.groupby(key)[metrics].agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    summary = summary.reset_index()
    summary.insert(3, "n_seeds", N_SEEDS)

    # Break-even on the EXPECTED P&L: a ratio of means, not a mean of ratios.
    # Solving E[net] = 0 gives E[cash + mark - hedge] / E[filled]. Averaging
    # the per-sample ``breakeven_passive_fee`` column instead averages a
    # ratio, which by Jensen's inequality is a different — and biased —
    # statistic (they disagree by ~$0.26/contract here). This column is the
    # one to quote, and it is what RebatePivotalityAnalyzer arrives at from
    # averaged metrics. The per-sample column is kept in pnl_samples.csv for
    # its distribution (quantiles, dispersion across seeds), not its mean.
    numerator = (
        summary["cash_mean"] + summary["position_mark_mean"] - summary["hedge_costs_mean"]
    )
    filled = summary["filled_contracts_mean"]
    summary["breakeven_passive_fee_expected"] = numerator.where(filled > 0) / filled.where(
        filled > 0
    )
    summary["required_rebate_expected"] = (
        fees.passive_fee - summary["breakeven_passive_fee_expected"]
    )

    samples_path = OUTPUT_DIR / "pnl_samples.csv"
    summary_path = OUTPUT_DIR / "daily_summary.csv"
    manifest_path = OUTPUT_DIR / "run_manifest.json"

    samples.to_csv(samples_path, index=False)
    summary.to_csv(summary_path, index=False)

    spec_json = {k: (str(v) if v is not None else None) for k, v in asdict(spec).items()}
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "product": PRODUCT,
                "front_month": FRONT_MONTH,
                "contract_spec": spec_json,
                "fee_tier": FEE_TIER,
                "passive_fee": fees.passive_fee,
                "aggressive_fee": fees.aggressive_fee,
                "book_levels": BOOK_LEVELS,
                "sessions": windows,
                "pooled_label": POOLED_LABEL,
                "grid": {
                    "p_queue_head": list(P_GRID),
                    "max_position": list(MAX_POSITION_GRID),
                    "n_seeds": N_SEEDS,
                    "base_seed": BASE_SEED,
                },
                "rows": len(samples),
                "files": {
                    "pnl_samples.csv": (
                        "one row per (session, p_queue_head, max_position, seed); "
                        "every generate_pnl() field, plus the break-even fee/rebate"
                    ),
                    "daily_summary.csv": (
                        "mean and std across seeds per (session, p_queue_head, "
                        "max_position); quote breakeven_passive_fee_expected / "
                        "required_rebate_expected (ratio of means), not the mean "
                        "of the per-sample breakeven column"
                    ),
                },
            },
            indent=2,
        )
    )

    print(f"\nwrote {len(samples)} samples -> {samples_path}")
    print(f"wrote {len(summary)} summary rows -> {summary_path}")
    print(f"wrote manifest -> {manifest_path}")

    pooled = summary[summary["session"] == POOLED_LABEL]
    print(f"\nPooled ({POOLED_LABEL}) — required rebate $/contract, by p and cap:")
    pivot = pooled.pivot(
        index="p_queue_head",
        columns="max_position",
        values="required_rebate_expected",
    )
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()
