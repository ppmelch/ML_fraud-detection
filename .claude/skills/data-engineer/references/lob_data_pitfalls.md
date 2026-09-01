# LOB Data Pitfalls

Problems specific to order book data, and what each looks like when it reaches
a downstream stage undetected.

---

## Aggregate volume is not enough to reconstruct queue position

The most consequential limitation in the whole project.

A feed reports that volume at price 100.50 dropped by 500 shares. It does not
report whether those 500 sat ahead of or behind a given resting order. Queue
position determines fill probability, and fill probability drives the P&L
difference between a strategy that looks profitable and one that does not.

**The consequence:** the simulator must assume something. Phase 3 makes the
assumption explicit and sweepable — `optimistic`, `proportional`, `pessimistic`
— and reports the band between extremes as an uncertainty range rather than
hiding a choice inside the code.

**Executions are different.** An execution at our price necessarily consumed
volume ahead of us, because that is what price-time priority means. So
execution-driven position reduction is exact, and only the cancel case requires
an assumption. Keep the two code paths distinct so the asymmetry stays visible.

---

## Crossed books

A crossed book has `best_bid >= best_ask`, which should be impossible.

**Causes:**
- Timestamp resolution: two events in the same millisecond, applied in the
  wrong order
- A dropped or reordered event in the feed
- A reconstruction bug — usually a cancel applied to the wrong side

**Detection:** `validate_lob_balance` (issue 0010) checks this on every
snapshot and reports `crossed_book_pct`.

**What not to do:** silently drop crossed snapshots. A crossed book indicates
either a data problem or a reconstruction bug, and both need investigating.
The `drop_crossed_book_events` flag in the cleaner defaults to off for this
reason.

---

## Equal timestamps carry ordering information

Several events legitimately share a millisecond. Their relative order is not
arbitrary — it determines FIFO queue position.

**Consequences:**
- Sorting must be **stable**: `df.sort_values("timestamp", kind="stable")`
- Validation requires timestamps **non-decreasing**, not strictly increasing
- `sequence_number` carries the strict ordering that timestamps cannot
- The simulator gates order arrival on sequence, never on timestamp comparison

A non-stable sort silently permutes same-millisecond events and changes every
queue position downstream, with no error anywhere.

---

## Orphan cancels and executes

An event referencing an `order_id` that never appeared in an `add`.

**Usual cause:** the stream starts mid-session, so orders resting before the
first recorded event are invisible. This is normal and expected at the start
of any capture.

**Handling:** the cleaner drops them and records the count. What matters is
that the count is reported — a large orphan fraction means the capture window
is too short relative to order lifetimes, which affects how much of the book
the reconstruction can see.

**Note:** an `execute` does not close an order. A partial fill leaves it open
and eligible for further events, so only `cancel` removes an id from the open
set. Getting this wrong produces spurious orphan-execute reports.

---

## Float accumulation on quantities

`quantity_remaining` is decremented repeatedly as partial fills arrive. After
enough subtractions it will not be exactly zero.

```python
# Wrong — will leave phantom orders in the book forever
if order.quantity_remaining == 0:
    remove(order)

# Right
QUANTITY_EPSILON = 1e-9
if order.quantity_remaining <= QUANTITY_EPSILON:
    remove(order)
```

Define the tolerance once as a module constant and use the same value in the
reconstructor, the validator, and the matching engine. Two components
disagreeing about what counts as zero is a bug that hides for a long time.

**Prices need the same care.** Round to the tick size on ingestion so
dictionary lookups by price are exact. A price off the tick grid creates a
level that can never match.

---

## Cached aggregates drift

`PriceLevel.total_volume` is maintained incrementally rather than recomputed,
for speed. That optimization is only safe if something independently
recomputes the sum and compares.

A single missed decrement silently inflates depth for the rest of the session
and quietly changes every OFI value computed from it — with no error raised.

Issue 0010 asserts `level.total_volume == sum(o.quantity_remaining for o in level.orders)`
after every event in a replay. That assertion is what makes the optimization
legitimate.

---

## Padding conventions

A book thinner than the serialization depth needs a value for the missing
levels. The choice matters more than it appears:

| Value | Problem |
|---|---|
| `0` | collides with a legitimately zero volume |
| `-1` | survives into a feature matrix and poisons a model |
| `NaN` | distinguishable from any legal value; pandas reads it natively |

`NaN` is the choice. Downstream code must then use `skipna=True` when summing
levels, so padding drops out rather than contributing zero:

```python
bid_vol = lob_df[bid_cols].sum(axis=1, skipna=True)
```

---

## Event density determines what latency can be studied

Latency is specified in milliseconds; the simulator advances in sequence
numbers. Converting requires knowing how many events occur in that window.

On the synthetic dataset the median inter-event gap is measured in seconds, so
**100 ms of latency converts to zero events** and every latency level produces
identical results.

This is a property of the data, not a bug — but a flat latency curve presented
without that caveat reads as a finding about markets. Issues 0046, 0047, 0051,
0062, and 0065 each carry a degeneracy flag forward for exactly this reason.

`analyze_horizons.py` in the quant-researcher skill detects and reports it.

---

## Session boundaries

A multi-day capture has overnight gaps where nothing trades.

**Consequences:**
- Forward returns spanning a session boundary measure an overnight move, not
  a microstructure move
- Rolling windows spanning a boundary mix regimes
- Latency conversion across a gap advances zero events, correctly but uselessly

The gap threshold in `profile_dataset.py` flags these. The current single-session
dataset does not have them, but real data will, and the labelling rule must
state how boundary-spanning rows are handled.

---

## Related

- Balance validation: `docs/issues/phase-1/0010`
- Queue assumptions: `docs/issues/phase-3/0036`
- Latency degeneracy: `docs/issues/phase-3/0046`
- Glossary: `docs/CONTEXT.md`
