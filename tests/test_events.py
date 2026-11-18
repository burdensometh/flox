"""Tests for consecutive exceedance-event detection (flox.events)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flox.events import (
    NO_EVENT,
    EventTable,
    find_events,
    label_events,
    reduce_events,
)

from . import has_dask, requires_dask

if has_dask:
    import dask
    import dask.array as da

    dask.config.set(scheduler="sync")


# ---------------------------------------------------------------------------
# numpy labeling
# ---------------------------------------------------------------------------


def test_basic_two_runs():
    time = np.arange(8)
    flags = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    labels, entities = label_events(flags, time)
    assert labels.tolist() == [1, 1, -1, -1, 2, 2, 2, -1]
    assert entities.tolist() == [0] * 8
    assert labels.dtype == np.int64


def test_max_gap_merges_and_splits():
    time = np.arange(8)
    flags = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    # gap between t=1 and t=4 is 3: gap>=3 merges, gap<3 splits
    merged, _ = label_events(flags, time, max_gap=3)
    assert merged.tolist() == [1, 1, -1, -1, 1, 1, 1, -1]
    split, _ = label_events(flags, time, max_gap=2)
    assert split.tolist() == [1, 1, -1, -1, 2, 2, 2, -1]
    # None requires direct adjacency
    none, _ = label_events(flags, time, max_gap=None)
    assert none.tolist() == [1, 1, -1, -1, 2, 2, 2, -1]


def test_missing_interrupt_vs_ignore():
    time = np.arange(6)
    flags = np.array([1, np.nan, 1, 0, 1, 1], dtype=float)

    interrupt, _ = label_events(flags, time, missing_policy="interrupt")
    assert interrupt.tolist() == [1, NO_EVENT, 2, NO_EVENT, 3, 3]

    ignore, _ = label_events(flags, time, missing_policy="ignore")
    # NaN transparent; the observed zero at pos3 still breaks (no gap given)
    assert ignore.tolist() == [1, NO_EVENT, 1, NO_EVENT, 2, 2]

    # ignore + gap spanning both the NaN (gap 2) and the zero (gap 2)
    ignore_gap, _ = label_events(flags, time, missing_policy="ignore", max_gap=2)
    assert ignore_gap.tolist() == [1, NO_EVENT, 1, NO_EVENT, 1, 1]

    # interrupt always breaks on NaN even with a generous gap
    interrupt_gap, _ = label_events(
        flags, time, missing_policy="interrupt", max_gap=10
    )
    assert interrupt_gap.tolist() == [1, NO_EVENT, 2, NO_EVENT, 2, 2]


def test_explicit_missing_mask():
    time = np.arange(4)
    flags = np.ones(4, dtype=bool)
    missing = np.array([0, 1, 0, 0], dtype=bool)
    labels, _ = label_events(flags, time, missing=missing, missing_policy="interrupt")
    assert labels.tolist() == [1, NO_EVENT, 2, 2]
    labels_ignore, _ = label_events(
        flags, time, missing=missing, missing_policy="ignore"
    )
    assert labels_ignore.tolist() == [1, NO_EVENT, 1, 1]


def test_entities_restart_ids_and_never_merge():
    time = np.arange(8)
    flags = np.ones(8, dtype=bool)
    by = np.array(["a", "a", "b", "b", "a", "a", "a", "b"])
    labels, codes = label_events(flags, time, by)
    # a run 0,1 -> event 1; b run 2,3 -> event 1 (restarts); a 4..6 -> event2;
    # b at 7 -> event 2 (b's own count)
    assert labels.tolist() == [1, 1, 1, 1, 2, 2, 2, 2]
    assert codes.tolist() == [0, 0, 1, 1, 0, 0, 0, 1]


def test_nan_entity_is_separate_and_does_not_crash():
    time = np.arange(4)
    flags = np.ones(4, dtype=bool)
    by = np.array(["a", "a", np.nan, "a"], dtype=object)
    labels, codes = label_events(flags, time, by)
    # NaN entity gets the missing-entity sentinel and forms its own run
    assert codes.tolist() == [0, 0, -1, 0]
    # a is split by the NaN entity; the NaN cell is its own event
    assert labels.tolist() == [1, 1, 1, 2]


def test_2d_rows_independent():
    time = np.arange(4)
    flags = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=bool)
    labels, _ = label_events(flags, time)
    assert labels.tolist() == [[1, 1, -1, 2], [-1, 1, 1, -1]]


def test_datetime64_gap():
    time = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06"],
        dtype="datetime64[D]",
    )
    flags = np.ones(4, dtype=bool)
    adj, _ = label_events(flags, time)
    assert adj.tolist() == [1, 1, 1, 1]
    two_days, _ = label_events(flags, time, max_gap=np.timedelta64(2, "D"))
    assert two_days.tolist() == [1, 1, 2, 2]
    three_days, _ = label_events(flags, time, max_gap=pd.Timedelta(days=3))
    assert three_days.tolist() == [1, 1, 1, 1]


def test_min_points_and_min_duration_keep_ids():
    time = np.arange(6)
    flags = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    points, _ = label_events(flags, time, min_points=2)
    # events 1 and 3 dropped; event 2 keeps its original id
    assert points.tolist() == [-1, -1, 2, 2, -1, -1]
    duration, _ = label_events(flags, time, min_duration=1)
    assert duration.tolist() == [-1, -1, 2, 2, -1, -1]


def test_validation_duplicate_and_unsorted():
    flags = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match="Duplicate"):
        label_events(flags, np.array([0, 1, 1]))
    with pytest.raises(ValueError, match="not strictly increasing"):
        label_events(flags, np.array([0, 2, 1]))
    with pytest.raises(ValueError, match="NaT"):
        label_events(
            np.array([1, 1, 1], dtype=float),
            np.array([0, np.datetime64("NaT"), 2], dtype="datetime64[D]"),
        )
    with pytest.raises(ValueError, match="missing_policy"):
        label_events(flags, np.arange(3), missing_policy="bogus")
    with pytest.raises(ValueError, match="min_points"):
        label_events(flags, np.arange(3), min_points=0)


def test_all_non_exceeding():
    labels, _ = label_events(np.zeros(5, dtype=bool), np.arange(5))
    assert labels.tolist() == [-1] * 5


def test_time_axis_not_last_restores_shape():
    flags = np.array([[1, 0, 1], [1, 1, 0]], dtype=bool)
    time = np.arange(2)
    labels, entities = label_events(flags, time, axis=0)
    assert labels.shape == flags.shape
    assert entities.shape == flags.shape
    # column 0: [1,1] one run; column 2: [1,0] one run
    assert labels[:, 0].tolist() == [1, 1]
    assert labels[:, 2].tolist() == [1, -1]


def test_all_missing_row_and_singleton():
    flags = np.array([[np.nan, np.nan], [1.0, 1.0]])
    labels, _ = label_events(flags, np.arange(2))
    assert labels[0].tolist() == [NO_EVENT, NO_EVENT]
    assert labels[1].tolist() == [1, 1]
    single, _ = label_events(np.array([1], bool), np.array([0]))
    assert single.tolist() == [1]
    none, _ = label_events(np.array([0], bool), np.array([0]))
    assert none.tolist() == [-1]


def test_reduce_events_empty_when_all_events_filtered():
    labels, entities = label_events(
        np.array([1, 0, 1], bool), np.arange(3), min_points=2
    )
    assert labels.tolist() == [-1, -1, -1]
    results, pairs = reduce_events(
        np.arange(3, dtype=float), labels, entities, funcs="max"
    )
    assert pairs.shape == (0, 2)
    assert results["max"].shape == (0,)


@requires_dask
def test_dask_reduce_events_empty_when_all_events_filtered():
    labels, entities = label_events(
        da.from_array(np.array([1, 0, 1, 0, 1], bool), chunks=2),
        da.from_array(np.arange(5), chunks=2),
        min_points=2,
    )
    assert (labels.compute() == NO_EVENT).all()
    results, pairs = reduce_events(
        da.from_array(np.arange(5, dtype=float), chunks=2),
        labels,
        entities,
        funcs=("max", "sum"),
    )
    assert pairs.compute().shape == (0, 2)
    assert results["max"].compute().shape == (0,)
    assert results["sum"].compute().shape == (0,)


# ---------------------------------------------------------------------------
# find_events / reduce_events
# ---------------------------------------------------------------------------


def test_find_events_table_aligned():
    time = np.arange(6)
    flags = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    values = np.arange(6, dtype=float)
    table = find_events(
        flags, time, values=values, stat_funcs=("max", "sum")
    )
    assert isinstance(table, EventTable)
    assert table.event_id.tolist() == [1, 2, 3]
    assert table.start_time.tolist() == [0, 2, 5]
    assert table.end_time.tolist() == [0, 3, 5]
    assert table.nobs.tolist() == [1, 2, 1]
    assert table.duration.tolist() == [0, 1, 0]
    assert table.stats["max"].tolist() == [0.0, 3.0, 5.0]
    assert table.stats["sum"].tolist() == [0.0, 5.0, 5.0]
    df = table.to_dataframe()
    assert list(df.columns)[:6] == [
        "event_id", "entity", "start_time", "end_time", "nobs", "duration",
    ]


def test_find_events_table_uses_raw_entity_labels_but_pairs_use_codes():
    time = np.arange(8)
    flags = np.ones(8, dtype=bool)
    by = np.array(["a", "a", "b", "b", "a", "a", "a", "b"])
    values = np.arange(8, dtype=float)
    table = find_events(flags, time, by, values=values, stat_funcs=("max",))
    # human-readable table carries the raw entity labels
    assert set(np.unique(table.entity).tolist()) == {"a", "b"}

    # reduction pairs carry integer entity codes, aligned to stats
    labels, codes = label_events(flags, time, by)
    results, pairs = reduce_events(values, labels, codes, funcs=("max",))
    assert pairs.shape[1] == 2
    assert set(np.unique(pairs[:, 0]).tolist()) == {0, 1}
    # every table row maps to a matching (code, event_id) reduction row
    code_of = {"a": 0, "b": 1}
    pair_set = {tuple(p) for p in pairs.tolist()}
    for ent, eid in zip(table.entity, table.event_id):
        assert (code_of[ent], int(eid)) in pair_set


def test_reduce_events_uses_groupby_engine_numpy():
    time = np.arange(8)
    flags = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    values = np.array([5.0, 9.0, 0, 0, 2, 7, 3, 0])
    labels, entities = label_events(flags, time)
    results, pairs = reduce_events(values, labels, entities, funcs=("max", "sum", "mean"))
    assert pairs.tolist() == [[0, 1], [0, 2]]
    assert results["max"].tolist() == [9.0, 7.0]
    assert results["sum"].tolist() == [14.0, 12.0]
    assert results["mean"].tolist() == [7.0, 4.0]


def test_reduce_events_drops_sentinel_only():
    # every non-event observation must be excluded from the reductions
    time = np.arange(4)
    flags = np.array([1, 0, 1, 0], dtype=bool)
    values = np.array([10.0, 99.0, 20.0, 99.0])
    labels, entities = label_events(flags, time)
    results, pairs = reduce_events(values, labels, entities, funcs="max")
    assert pairs.tolist() == [[0, 1], [0, 2]]
    assert results["max"].tolist() == [10.0, 20.0]


# ---------------------------------------------------------------------------
# dask parity
# ---------------------------------------------------------------------------


@requires_dask
@pytest.mark.parametrize(
    "max_gap", [None, 1, 2, 3, 5],
)
@pytest.mark.parametrize("missing_policy", ["interrupt", "ignore"])
def test_dask_matches_numpy_1d(max_gap, missing_policy):
    rng = np.random.default_rng(42)
    time = np.arange(30)
    flags = rng.random(30) > 0.4
    # sprinkle some NaN missing values
    flags_f = flags.astype(float)
    flags_f[rng.random(30) < 0.2] = np.nan

    chunks = (7, 8, 8, 7)
    np_labels, np_ent = label_events(
        flags_f, time, max_gap=max_gap, missing_policy=missing_policy
    )
    d_labels, d_ent = label_events(
        da.from_array(flags_f, chunks=chunks),
        da.from_array(time, chunks=chunks),
        max_gap=max_gap,
        missing_policy=missing_policy,
    )
    np.testing.assert_array_equal(d_labels.compute(), np_labels)
    np.testing.assert_array_equal(d_ent.compute(), np_ent)


@requires_dask
def test_dask_matches_numpy_2d():
    rng = np.random.default_rng(7)
    flags = (rng.random((6, 16)) > 0.5).astype(float)
    flags[rng.random(flags.shape) < 0.15] = np.nan
    time = np.arange(16)
    np_labels, np_ent = label_events(flags, time, max_gap=2, missing_policy="ignore")
    d_labels, d_ent = label_events(
        da.from_array(flags, chunks=(3, 5)),
        da.from_array(time, chunks=5),
        max_gap=2,
        missing_policy="ignore",
    )
    np.testing.assert_array_equal(d_labels.compute(), np_labels)
    np.testing.assert_array_equal(d_ent.compute(), np_ent)


@requires_dask
def test_dask_event_spanning_many_chunks():
    # one long run split across many small chunks must remain a single event
    n = 40
    flags = np.ones(n, dtype=bool)
    time = np.arange(n)
    labels, _ = label_events(
        da.from_array(flags, chunks=3), da.from_array(time, chunks=3)
    )
    assert labels.compute().tolist() == [1] * n


@requires_dask
def test_dask_entities_match_numpy():
    rng = np.random.default_rng(3)
    n = 24
    flags = rng.random(n) > 0.4
    entities = rng.choice(["a", "b"], size=n)
    time = np.arange(n)
    np_labels, np_codes = label_events(flags, time, entities)
    d_labels, d_codes = label_events(
        da.from_array(flags, chunks=6),
        da.from_array(time, chunks=6),
        da.from_array(entities, chunks=6),
        expected_entities=["a", "b"],
    )
    np.testing.assert_array_equal(d_labels.compute(), np_labels)
    np.testing.assert_array_equal(d_codes.compute(), np_codes)


@requires_dask
def test_dask_requires_expected_entities():
    time = da.from_array(np.arange(8), chunks=4)
    flags = da.from_array(np.ones(8, bool), chunks=4)
    entities = da.from_array(np.array(["a"] * 8), chunks=4)
    with pytest.raises(ValueError, match="expected_entities"):
        label_events(flags, time, entities)


@requires_dask
@pytest.mark.parametrize(
    "bad",
    [
        np.array([0, 1, 1, 2, 3, 4, 5, 6]),       # duplicate across boundary
        np.array([0, 1, 3, 2, 4, 5, 6, 7]),       # unsorted across boundary
        np.array([0, 1, 2, 3, 3, 5, 6, 7]),       # duplicate inside second chunk
        np.array([0, 1, 2, 5, 4, 5, 6, 7]),       # unsorted inside a chunk
    ],
)
def test_dask_rejects_bad_time(bad):
    flags = da.from_array(np.ones(8, bool), chunks=4)
    time = da.from_array(bad, chunks=4)
    with pytest.raises(ValueError):
        label_events(flags, time)


@requires_dask
def test_dask_does_not_materialize_full_time_axis():
    # Only O(blocks) boundary descriptors are computed during labeling; the
    # returned relabel array itself must stay a lazy dask collection whose
    # evaluation is a separate single compute.
    n = 200
    flags = da.from_array(np.random.default_rng(1).random(n) > 0.5, chunks=20)
    time = da.from_array(np.arange(n), chunks=20)
    labels, _ = label_events(flags, time)
    assert hasattr(labels, "compute")  # lazy, not a numpy array
    # per-task data is one chunk (20), never the full 200-element axis
    for layer_chunks in labels.chunks:
        assert max(layer_chunks) <= 20
    np.testing.assert_array_equal(
        labels.compute(),
        label_events(np.asarray(flags), np.arange(n))[0],
    )


@requires_dask
def test_dask_reduce_events_matches_numpy():
    time = np.arange(20)
    flags = np.random.default_rng(11).random(20) > 0.4
    values = np.arange(20, dtype=float)
    labels, entities = label_events(flags, time)
    d_labels, d_entities = label_events(
        da.from_array(flags, chunks=6), da.from_array(time, chunks=6)
    )
    np_res, np_pairs = reduce_events(values, labels, entities, funcs=("max", "sum"))
    d_res, d_pairs = reduce_events(
        da.from_array(values, chunks=6), d_labels, d_entities, funcs=("max", "sum")
    )
    np.testing.assert_array_equal(d_pairs.compute(), np_pairs)
    np.testing.assert_array_equal(d_res["max"].compute(), np_res["max"])
    np.testing.assert_array_equal(d_res["sum"].compute(), np_res["sum"])


@requires_dask
def test_dask_find_events_matches_numpy():
    time = np.arange(18)
    flags = np.random.default_rng(5).random(18) > 0.4
    values = np.arange(18, dtype=float)
    table_np = find_events(flags, time, values=values, max_gap=2)
    table_dk = find_events(
        da.from_array(flags, chunks=5),
        da.from_array(time, chunks=5),
        values=da.from_array(values, chunks=5),
        max_gap=2,
    )
    np.testing.assert_array_equal(table_dk.event_id, table_np.event_id)
    np.testing.assert_array_equal(table_dk.start_time, table_np.start_time)
    np.testing.assert_array_equal(table_dk.nobs, table_np.nobs)
    np.testing.assert_array_equal(table_dk.stats["max"], table_np.stats["max"])


@requires_dask
@settings(deadline=None, max_examples=40)
@given(
    data=st.data(),
    n=st.integers(min_value=1, max_value=40),
    chunk=st.integers(min_value=1, max_value=12),
    gap_choice=st.sampled_from([None, 1, 2, 4]),
    policy=st.sampled_from(["interrupt", "ignore"]),
)
def test_property_dask_numpy_parity(data, n, chunk, gap_choice, policy):
    # explicit random seed drawn by hypothesis makes failures reproducible
    seed = data.draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    raw = rng.random(n)
    kind = rng.choice(["exc", "miss", "none"], size=n, p=[0.6, 0.2, 0.2])
    flags = np.full(n, np.nan)
    flags[kind == "exc"] = 1.0
    flags[kind == "none"] = 0.0
    time = np.arange(n)
    max_gap = None if gap_choice is None else gap_choice

    np_labels, np_entities = label_events(
        flags, time, max_gap=max_gap, missing_policy=policy
    )
    d_labels, d_entities = label_events(
        da.from_array(flags, chunks=chunk),
        da.from_array(time, chunks=chunk),
        max_gap=max_gap,
        missing_policy=policy,
    )
    np.testing.assert_array_equal(d_labels.compute(), np_labels)
    np.testing.assert_array_equal(d_entities.compute(), np_entities)


@requires_dask
@pytest.mark.parametrize("nrows", [2, 3, 5])
@pytest.mark.parametrize("row_chunk", [1, 2])
def test_dask_2d_multiple_rows_parity(nrows, row_chunk):
    # Regression: per-entity event ids restart per row, but event records must
    # stay globally distinct so one row's id never merges another row's event.
    rng = np.random.default_rng(101 + nrows + row_chunk)
    ntime = 37
    kind = rng.choice(["exc", "miss", "none"], size=(nrows, ntime), p=[0.6, 0.2, 0.2])
    flags = np.full((nrows, ntime), np.nan)
    flags[kind == "exc"] = 1.0
    flags[kind == "none"] = 0.0
    time = np.arange(ntime)

    np_labels, np_entities = label_events(flags, time, max_gap=2, min_points=2)
    d_labels, d_entities = label_events(
        da.from_array(flags, chunks=(row_chunk, 5)),
        da.from_array(time, chunks=5),
        max_gap=2,
        min_points=2,
    )
    np.testing.assert_array_equal(d_labels.compute(), np_labels)
    np.testing.assert_array_equal(d_entities.compute(), np_entities)

    # reductions over the 2-D events must match the flattened reference
    values = rng.normal(size=(nrows, ntime))
    np_res, np_pairs = reduce_events(values, np_labels, np_entities, funcs="max")
    d_res, d_pairs = reduce_events(
        da.from_array(values, chunks=(row_chunk, 5)), d_labels, d_entities, funcs="max"
    )
    np.testing.assert_array_equal(d_pairs.compute(), np_pairs)
    np.testing.assert_array_equal(d_res["max"].compute(), np_res["max"])
