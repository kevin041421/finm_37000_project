# Calendar Spread Market-Making Incentives

## Goal

This project studies **calendar spread market-making on CME** and pursues two goals.

### Goal 1 — When and where is an exchange incentive pivotal?

> **What is the expected P&L of quoting the calendar spread under realistic fill and hedging dynamics; under what conditions does an exchange incentive — fee discount, maker rebate, or other compensation — become necessary; and which market should that incentive target?**

A trader can express a calendar view either by trading the two outright contracts separately (legging) or by trading the exchange-listed spread instrument as a single order. The spread instrument quotes a tighter bid–ask than the implied cost of crossing both outright books, so traders prefer it — and the market maker on the other side faces a real **legging cost** if it crosses out into the outrights to flatten (the worked example below puts this at \$25/contract for ES).

But that legging cost is not the maker's expected loss. A maker that crosses out *immediately* on every fill would lose; a real desk does not. Its actual P&L also includes the bid–ask it earns on round-trip and offsetting flow, passive (resting) fills that avoid crossing, and CME implied matching — all of which can turn the activity profitable without any subsidy. The legging cost is therefore a floor on what the maker must overcome, not a verdict.

The deliverable is a model of the maker's **expected** P&L — legging cost, passive vs. aggressive fill probabilities, adverse selection, and inventory/holding risk — estimated bottom-up from real order-book data (via Databento) rather than assumed. We then ask the sharper question: holding that P&L below break-even, **how large a rebate would be pivotal**, and how does that compare to CME's actual maker-incentive programs for these products? We start from the `project_fetch_data.py` snapshot and the legging-cost analysis below and build out from there.

A related question runs through Goal 1: **how does market making the calendar spread differ from market making a normal product/instrument?** The spread is two-legged, links to the outright books via CME implied matching, carries basis rather than directional inventory risk, and lives in a thinner, roll-driven market — so the maker's role, risks, and the incentive it needs may all differ from the outright case. We treat this difference as a question to investigate rather than assume.

### Goal 2 — Relationship analysis and a market-making strategy for the deferred leg and the spread

> **What is the statistical relationship between the front month, the back month, and the calendar spread — and, building on it, what is a sound market-making strategy for the deferred leg and the calendar spread markets?**

First we characterize the three markets empirically: their liquidity (bid–ask, depth, quoting and trade activity) and the lead–lag / co-movement relationships among them. The working hypothesis, to be confirmed in the data, is that the **front month leads** — it is the most liquid and carries outright price discovery — while the **calendar spread leads the basis** (the term structure), and the **deferred leg follows both**, its quotes being largely implied from front + spread.

On that footing we build the quoting strategy. Because the deferred outright and the spread are the two thin, linked markets that follow the liquid front month, we make markets in *both* of them: anchor fair value to the front-month book and a carry-based estimate of the basis, hedge directional risk in the liquid front leg, and manage inventory and legging risk rather than forecasting index direction. The deferred leg and the spread are quoted consistently (since one is implied from the other plus the front leg), so the strategy treats them as a joint book.

The two goals are linked: the quote width and quoting behavior the strategy can sustain depend directly on the incentive analysis in Goal 1. The rebate determined there sets how tightly the maker can afford to quote and still break even, so Goal 1 fixes the economic floor and Goal 2 builds the quoting policy on top of it.

### Instruments in scope

We begin with the **E-mini S&P 500 (ES) calendar spread** (e.g. ESM6–ESU6) as the worked example, then extend the analysis to calendar spreads on other CME products. CME Globex lists exchange-defined calendar (inter-delivery) spreads as single tradable instruments — each with its own order book — across many asset classes, so the same legging-cost dynamic can be measured for each. Candidate products include:

| Asset class | Product (code) |
|---|---|
| Equity index | E-mini S&P 500 (ES), E-mini Nasdaq-100 (NQ), E-mini Dow (YM), E-mini Russell 2000 (RTY) |
| Energy | WTI Crude Oil (CL), Henry Hub Natural Gas (NG), RBOB Gasoline (RB), Heating Oil (HO) |
| Metals | Gold (GC), Silver (SI), Copper (HG) |
| Grains | Corn (ZC), Soybeans (ZS), Wheat (ZW) |

All of the E-mini equity index futures share the same structure as ES — a quarterly (Mar/Jun/Sep/Dec) cycle with the same roll-driven demand — and their calendar spreads quote in 0.05-point ticks, so the legging-cost analysis transfers directly across the equity index complex.

These are *futures* calendar spreads (one order, both legs). They are distinct from **Calendar Spread Options (CSOs)** — options written on the spread — which CME also lists but which are outside this project's scope.

## Motivation

Market participants have a clear incentive to trade the spread, while the market maker appears to lose money — on the surface. That tension — a real legging cost on one side, persistent tight two-sided quotes on the other — is what motivates both goals: modeling the maker's *true* expected P&L and when an incentive is pivotal (Goal 1), and characterizing the front/back/spread relationship to quote the deferred leg and the spread (Goal 2).

### Why traders need the spread: rolling positions

There is a constant, structural demand for calendar spreads because traders routinely need to **roll their positions from the front month to the back month**. A trader who is long (or short) the expiring front contract but wants to keep the exposure must, before expiry, close the front-month position and re-open it in the deferred month. That roll *is* a calendar spread: sell the front and buy the back (or vice versa), executed as a single package. Because contracts expire on a fixed cycle, this rolling demand recurs every quarter and concentrates around the roll period — which is exactly the flow the spread instrument is designed to serve.

### The two ways to trade a calendar spread

Suppose a trader wants to roll a long position forward — i.e. be **long the ESU6–ESM6 calendar** (buy the back month U6, sell the front month M6). They have two routes:

1. **Leg the outrights** — sell ESM6 and buy ESU6 as two separate orders, each crossing its own bid–ask spread.
2. **Trade the spread instrument** — submit one order on the exchange-listed ESM6–ESU6 spread, which has its own, much tighter, bid–ask.

Because the spread instrument is quoted far tighter than the two outright books combined, route 2 is cheaper for the trader. That is exactly why the spread instrument exists and attracts flow.

### Worked example: ES calendar spread (order-book snapshot)

We illustrate the dynamic with the ES calendar spread; the same calculation will be repeated for the other products in scope. The order books of the two outright legs and the spread instrument, from a representative snapshot taken at **2026-06-10 16:00:00 UTC (10:00 AM CT)**, are shown below (asks on top, bids below; the best bid and best ask sit just above and below the divider):

```
        ESM6 (front)            ESU6 (back)          ESM6-ESU6 spread
       Sz         Px           Sz         Px           Sz         Px
     ------  ---------       ------  ---------       ------  ---------
        120    7341.25          110    7402.25          350      60.95   asks
         95    7341.00           80    7402.00          300      60.90
         60    7340.75           55    7401.75          250      60.85
         40    7340.50           35    7401.50          180      60.80
         25    7340.25 <ask      20    7401.25 <ask      90      60.75 <ask
     ------  ---------       ------  ---------       ------  ---------
         30    7340.00 <bid      22    7401.00 <bid      85      60.70 <bid
         55    7339.75           50    7400.75          175      60.65
         80    7339.50           75    7400.50          240      60.60
        110    7339.25          100    7400.25          290      60.55   bids
        140    7339.00          130    7400.00          340      60.50
```

The relevant touch prices are therefore: ESM6 best bid **7340.00**, ESU6 best ask **7401.25**, and ES spread best ask **60.75** (best bid **60.70**, a 0.05-point market). Note the spread instrument is quoted **0.05 points wide**, far tighter than the **0.25-point** tick on each outright leg.

The ES multiplier is **\$50 per index point**, and one tick (0.25 pt) is worth \$12.50.

A trader who buys the spread instrument pays the **spread ask = 60.75 points**. The market maker is the counterparty: it **sells the spread**, leaving it short the spread, i.e. **long ESM6 / short ESU6**.

To flatten immediately, the market maker legs out in the outright books:

- Sell ESM6 → hits the **bid** at 7340.00
- Buy ESU6 → hits the **ask** at 7401.25

$$\text{Cost of legging out} = 7401.25 - 7340.00 = 61.25 \text{ points}$$

But the maker only collected the spread ask it sold:

$$\text{P\&L} = 60.75 - 61.25 = -0.50 \text{ points} = -0.50 \times \$50 = \boxed{-\$25 \text{ per contract}}$$

### The punchline

The market maker **loses \$25 per contract** if it hedges by legging out the instant it is filled. The spread instrument trades tighter (here a 0.05-point bid–ask) than the implied cost of crossing both outright books (a 0.25-point half-spread on each side, ≈ 0.50 points of net disadvantage). The trader captures that tightness as a saving; the market maker absorbs the **legging cost** and the inventory risk of holding an unhedged position while waiting for offsetting flow.

So the surface picture is:

- **Trader:** saves ~0.50 points by using the tight spread instrument instead of legging.
- **Market maker:** appears to lose ~\$25/contract on every immediately-hedged fill.

A maker that crossed out on every fill would indeed lose — but the \$25 is a **floor**, not its expected P&L. Liquid, tight spread markets exist precisely because the economics close elsewhere: earning the bid–ask on round-trip and offsetting flow, passive fills that avoid crossing, implied matching, and disciplined inventory management. Two questions follow, and they are the project's goals: *given* those dynamics, what is the maker's expected P&L and when does an exchange incentive actually become pivotal (**Goal 1**); and, using the statistical relationship among the front month, back month, and spread, how should the maker quote the deferred leg and the spread (**Goal 2**).
