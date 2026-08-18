# Market & Strategy Design

## 1. Target Market: ERCOT

**ERCOT** (Electric Reliability Council of Texas) is the ISO selected for this
project for the following reasons:

- **Data availability**: Historical Day-Ahead and Real-Time settlement-point
  prices are freely available via ERCOT's public data portal and the
  `gridstatus` Python library.
- **Market structure**: ERCOT operates as an island grid with no external AC
  interconnections to neighbouring systems, making its prices almost entirely
  endogenous.  This reduces the noise introduced by inter-regional power flows
  and simplifies the modelling problem.
- **Renewable penetration**: ERCOT has one of the largest installed wind
  capacities in the US.  Wind forecast errors are a primary driver of DA/RT
  spread volatility and form a strong predictor in this strategy.
- **Volatility regime**: ERCOT experiences some of the most extreme price
  events in North American power markets (e.g. Winter Storm Uri, 2021),
  providing rich training data for stress-testing.

---

## 2. Trading Instrument: Virtual Bids

ERCOT's nodal market allows **virtual bids (VBs) and virtual offers (VOs)** at
any settlement point.  A virtual participant acts as a financial intermediary
with no physical generation or load obligation:

| Position | DA action | RT action | Profit condition |
|---|---|---|---|
| **Virtual offer (short DA)** | Sell power at DAM price | Buy back in RTM | `DA > RT` → positive spread |
| **Virtual bid (long DA)** | Buy power at DAM price | Sell back in RTM | `RT > DA` → negative spread |

**Settlement**: For each MWh of virtual position at settlement point *i* in
hour *t*:

```
P&L = position_sign × (DA_LMP_{i,t} − RT_LMP_{i,t}) × MW_quantity
```

where `position_sign = +1` for a virtual offer and `−1` for a virtual bid.

---

## 3. Node vs Hub Strategy

We focus on the **four main ERCOT trading hubs**:

| Hub | Geography |
|---|---|
| `HB_NORTH` | North zone (Dallas–Fort Worth region) |
| `HB_SOUTH` | South zone (San Antonio region) |
| `HB_WEST` | West zone (wind-heavy Permian Basin) |
| `HB_HOUSTON` | Houston Ship Channel region |

**Rationale for hubs over nodes**: Individual settlement-point nodes exhibit
highly idiosyncratic congestion components that are difficult to model without
detailed transmission topology data.  Hub prices aggregate across many nodes,
producing smoother, more predictable spread dynamics while retaining exposure
to the main market-wide drivers.  A nodal extension is listed as a Phase 9
enhancement.

---

## 4. Trading Frequency and Holding Period

- **Frequency**: One trade opportunity per hour-hub pair, per day.
- **Decision time**: Trades must be submitted before the DAM closes
  (~10:00 AM CPT the day before delivery).  All features used in the model
  must be available at this point no look-ahead bias.
- **Holding period**: Exactly one hour (positions are settled at hour-end by
  ERCOT).  There is no intra-day exit option; the position is always held to
  settlement.

---

## 5. Market Mechanics

### Day-Ahead Market (DAM)
- Clears once daily for all 24 hours of the following operating day.
- Participants submit energy bids and offers by ~10:00 AM CPT.
- Prices are hourly Settlement Point Prices (SPPs) in $/MWh.
- Based on a security-constrained unit commitment (SCUC) + economic dispatch.

### Real-Time Market (SCED)
- Clears every 5 minutes via Security-Constrained Economic Dispatch (SCED).
- 15-minute Real-Time Settlement Point Prices (RTSPPs) are published after
  each interval.
- For P&L calculation, the relevant RT price per hour is the arithmetic mean
  of the four 15-minute intervals: `RT_hourly = mean(RTSPP_{t, t+15, t+30, t+45})`.

### Virtual Bid Settlement
1. Participant submits a virtual offer/bid before DAM close with MW quantity
   and a price.
2. If the bid clears in the DAM, the participant is financially committed to
   that volume at the DAM price.
3. At the end of each real-time interval, the offsetting position is settled
   at the RT SPP.
4. Net cash flow = `(DA SPP − RT SPP) × cleared MW` for a virtual offer.

---

## 6. Economic Hypothesis

The DA/RT spread should be predictable to a degree because it is driven by
**information available at DAM close that is imperfectly incorporated into
DAM prices**:

1. **Load forecast error**: If ERCOT's DAM load forecast underestimates actual
   real-time demand, RT prices will exceed DA prices as the system dispatches
   more expensive peaking units.  Load forecast errors of ±2–5% are common.

2. **Renewable forecast error**: ERCOT West has high wind variability.  If
   wind underperforms its DAM forecast, RT prices will spike above DA prices
   (and vice versa if wind overperforms).

3. **Generator outages**: Unplanned unit trips after DAM close reduce RT
   supply, pushing RT prices above DA prices.

4. **Calendar and congestion patterns**: Certain hours, seasons, and
   congestion regimes have persistent historical biases in the DA/RT spread
   that the market may not fully arbitrage.

**Limits to arbitrage**: Virtual bidding activity tends to reduce
predictability over time as more capital chases the same signals.  The model
is expected to show declining out-of-sample alpha post ~2019 as the ERCOT
virtual market became more competitive.  This is an honest limitation to
document and is consistent with evidence in the academic literature (e.g.
Hadsell 2011, Birge et al. 2017).

---

## 7. Performance Benchmark

The **naive zero-trade benchmark** is chosen as the primary comparison: a
strategy that never trades, always returning zero.

Secondary benchmarks:
- **Always-short DA**: Takes a virtual offer at every hub-hour regardless of
  the signal (equivalent to always betting `DA > RT`).
- **Seasonal mean strategy**: Trades in the direction of the historical
  seasonal-average spread sign for that hub and hour-of-day.

These benchmarks quantify whether the model adds value beyond simple heuristics.

---

## 8. Key Assumptions and Known Limitations

| Assumption | Justification | Limitation |
|---|---|---|
| Hub-level positions only | Simplifies modelling; avoids nodal topology data | Misses nodal congestion alpha |
| No capacity constraints | Simplifies position sizing | Real markets have MW bid caps |
| Transaction cost = flat $/MWh | Approximates bid-offer spread | Actual costs vary by liquidity |
| Perfect DA fill at published price | Conservative | Partial fills possible in reality |
| No credit/collateral costs | Simplifies P&L | Material for large positions |

---

## References

- ERCOT Nodal Protocols: https://www.ercot.com/mktrules/nprotocols
- Hadsell, L. (2011). *Virtual Bidding in New York's Electricity Market.*
- Birge, J., Hortaçsu, A., et al. (2017). *Virtual Bidding and Electricity Market Design.*
- EIA: https://www.eia.gov/electricity/wholesale/
