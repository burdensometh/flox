# Consecutive exceedance events

Threshold monitoring questions are about *events*, not individual flags: when
did a station start and stop exceeding a limit, how long did each episode
last, and what was the peak during it? Grouping directly by a boolean
"exceeds / does not exceed" label is wrong — it merges every exceedance in
the whole record into a single group. `flox.events` first turns the boolean
flag into run-length **event ids** along an ordered time axis, then reuses
the existing groupby reduction engine to summarize each event.

## Labeling events

{py:func}`flox.label_events` walks the time axis and assigns a positive,
1-based event id to every exceeding observation that belongs to an event;
every other observation gets the non-event sentinel `-1`.

```python
import numpy as np
from flox import label_events

time = np.arange(8)
exceeds = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)

labels, entities = label_events(exceeds, time)
labels
# array([ 1,  1, -1, -1,  2,  2,  2, -1])
```

The two runs are now distinct groups instead of one boolean group.

### Gap tolerance

By default an event may only continue across directly adjacent exceedances.
`max_gap` instead allows exceedances separated by at most a given time gap
to stay in the same event (the gap is measured between the surrounding
*observed* exceedances):

```python
label_events(exceeds, time, max_gap=3)[0]
# array([ 1,  1, -1, -1,  1,  1,  1, -1])
```

`max_gap` accepts a `numpy.timedelta64` or pandas-timedelta-like value when
the time axis is `datetime64`, and a plain number for a numeric time axis.

### Missing observations

Missing observations (`NaN`/`NaT` in the flag array, or positions given via
`missing`) follow an explicit contract through `missing_policy`:

- `"interrupt"` (default): a missing observation breaks the run.
- `"ignore"`: missing positions are transparent and the time gap spans
  across them.

### Minimum event length

Events shorter than `min_points` exceeding observations, or shorter than
`min_duration` (`end_time - start_time`), are dropped: their observations
are relabeled to `-1`. Surviving events keep their original ids (ids are not
renumbered after filtering).

## Multiple entities

Pass entity labels with `by` to segment independently per entity. Event ids
restart at 1 for each entity and events never merge across entities:

```python
by = np.array(["a", "a", "b", "b", "a", "a", "a", "b"])
label_events(np.ones(8, bool), time, by)[0]
# array([1, 1, 1, 1, 2, 2, 2, 2])
```

When `by` is a dask array you must also pass `expected_entities`, the full
set of entity labels, so factorization stays blockwise.

## Reducing each event

The event id is a real group label. {py:func}`flox.reduce_events` builds a
composite `(entity, event_id)` key and runs the ordinary
{py:func}`flox.groupby_reduce` machinery over it, so every returned
statistic aligns exactly with the returned event ids. Observations labeled
`-1` are excluded.

```python
from flox import reduce_events

values = np.array([5.0, 9.0, 0, 0, 2, 7, 3, 0])
results, pairs = reduce_events(values, labels, entities, funcs=("max", "sum"))
pairs            # [[entity_code, event_id], ...]
# array([[0, 1],
#        [0, 2]])
results["max"]   # array([9., 7.])
results["sum"]   # array([14., 12.])
```

## Event table

{py:func}`flox.find_events` assembles one aligned row per event with its
start/end time, duration and observation count, plus any requested
statistics:

```python
from flox import find_events

table = find_events(exceeds, time, values=values, stat_funcs=("max", "sum"))
table.to_dataframe()
#    event_id  entity  start_time  end_time  nobs  duration  max   sum
# 0         1       0           0         1     2         1  9.0  14.0
# 1         2       0           4         6     3         2  7.0  12.0
```

## How it scales with dask

For dask inputs `label_events` never loads the full time axis:

1. local run-length labels are assigned independently inside every block;
2. only compact per-block descriptors (the events touching a block and the
   positions of missing observations) are computed — their size is
   proportional to the number of blocks/events, not observations;
3. those descriptors are merged on the driver to give globally consistent
   ids, correctly joining one event that spans chunk boundaries;
4. a final blockwise relabel applies the tiny per-block id maps.

The time axis is validated for strict ordering from block endpoints and
in-block checks, again without materializing it. Duplicate or
non-monotonic timestamps raise `ValueError`.
