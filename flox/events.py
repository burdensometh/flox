"""Consecutive exceedance-event detection along an ordered time axis.

This module turns a boolean "observation exceeds a threshold" flag into
*events*: maximal runs of exceeding observations that belong to the same
entity and whose consecutive *observed* timestamps are no further apart than
``max_gap``. Every retained event receives a stable integer id that can be
used directly as a group label for :func:`flox.groupby_reduce`.

The implementation is chunk-aware. Local event labels are first assigned
independently inside every dask block; only compact per-block descriptors
(the events touching a block and the positions of missing observations) are
merged on the driver, after which a blockwise relabel step makes ids
globally consistent. The full time axis is never materialized -- validation,
merging and filtering only touch data whose size is proportional to the
number of blocks/events, not the number of observations.

Contracts
---------
* ``time`` must be strictly increasing along ``axis``. Duplicate timestamps
  and non-monotonic timestamps raise ``ValueError`` (also for dask inputs,
  where only block-local values and adjacent block boundaries are inspected).
* A missing observation (NaT/NaN in ``flags`` or a True entry in ``missing``)
  either interrupts an event (``missing_policy="interrupt"``) or is
  transparent (``missing_policy="ignore"``); with ``"ignore"`` the gap is
  measured between the surrounding *observed* timestamps.
* Events with fewer than ``min_points`` exceeding observations, or shorter
  than ``min_duration`` (``end_time - start_time``), are dropped: their
  observations are relabeled to the non-event sentinel ``-1``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .aggregations import Aggregation
from .core import groupby_reduce
from .factorize import factorize_
from .xrutils import is_duck_dask_array, isnull

__all__ = [
    "EventTable",
    "find_events",
    "label_events",
    "reduce_events",
]

#: Label assigned to observations that do not belong to a retained event.
NO_EVENT = -1


@dataclass
class EventTable:
    """One row per detected event; all attributes are aligned 1-D arrays.

    Attributes
    ----------
    event_id : ndarray[int64]
        Per-entity event ids (1-based; unique together with ``entity``).
    entity : ndarray
        Entity label each event belongs to.
    start_time, end_time : ndarray
        Timestamps of the first and last exceeding observations.
    nobs : ndarray[int64]
        Number of exceeding observations in the event.
    duration : ndarray
        ``end_time - start_time`` (timedelta64 for datetime axes).
    stats : dict[str, ndarray]
        Aligned per-event reductions requested through ``stat_funcs``;
        empty when none were requested.
    """

    event_id: np.ndarray
    entity: np.ndarray
    start_time: np.ndarray
    end_time: np.ndarray
    nobs: np.ndarray
    duration: np.ndarray
    stats: dict[str, np.ndarray]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the table (including any stats) to a :class:`pandas.DataFrame`."""
        data = {
            "event_id": self.event_id,
            "entity": self.entity,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "nobs": self.nobs,
            "duration": self.duration,
        }
        data.update(self.stats)
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _strictly_increasing(time: np.ndarray) -> None:
    """Raise on NaT/NaN, duplicate or non-monotonic 1-D timestamps."""
    time = time.reshape(-1)
    if time.size == 0:
        return
    if bool(isnull(time).any()):
        raise ValueError("`time` contains NaT/NaN values; the time axis must be complete.")
    if time.size == 1:
        return
    is_datetime = time.dtype.kind == "M"
    diff = np.diff(time.astype("datetime64[ns]") if is_datetime else time)
    bad = np.flatnonzero(diff <= 0)
    if bad.size:
        i = int(bad[0])
        if diff[i] == 0:
            raise ValueError(
                f"Duplicate timestamp {time[i]!r} at positions {i} and {i + 1}; "
                "`time` must be strictly increasing."
            )
        raise ValueError(
            f"`time` is not strictly increasing: {time[i]!r} is followed by {time[i + 1]!r}."
        )


def _time_endpoints(block: np.ndarray) -> np.ndarray:
    """First/last timestamp of a 1-D block, validating strict order inside.

    Interior duplicates or unsorted values are rejected here even though only
    the endpoints are returned to the driver.
    """
    flat = block.reshape(-1)
    _strictly_increasing(flat)
    return flat if flat.size <= 2 else flat[[0, -1]]


def _time_endpoints_2d(block: np.ndarray) -> np.ndarray:
    """Endpoints of every row of a 2-D block, validating each row in full.

    Returns shape ``(n_rows, 2)`` for blocks with >=2 time points and
    ``(n_rows, 1)`` for a singleton time block, so downstream concatenation
    keeps the boundary sequence contiguous.
    """
    width = 2 if block.shape[1] > 1 else 1
    out = np.empty((block.shape[0], width), dtype=block.dtype)
    for r in range(block.shape[0]):
        row = block[r]
        _strictly_increasing(row)
        if width == 2:
            out[r, 0] = row[0]
            out[r, 1] = row[-1]
        else:
            out[r, 0] = row[0]
    return out


def _validate_time(time, ndim: int) -> None:
    """Validate strict ordering eagerly or from chunk endpoints (lazy).

    Each block kernel checks strict ordering inside its block; the driver
    then checks the boundary sequence between blocks. Only two timestamps per
    block (per row) are materialized, so validation costs O(blocks), never the
    full time axis.
    """
    if is_duck_dask_array(time):
        import dask.array

        if ndim == 1:
            endpoints = dask.array.map_blocks(
                _time_endpoints, time,
                dtype=time.dtype, meta=np.array([], dtype=time.dtype),
            ).compute()
            # blocks emit [first, last] (or one value for a singleton block);
            # their concatenation is already in time-block order
            _strictly_increasing(np.asarray(endpoints))
        else:
            endpoints = dask.array.map_blocks(
                _time_endpoints_2d, time,
                dtype=time.dtype, meta=np.empty((0, 2), dtype=time.dtype),
            ).compute()
            endpoints = np.asarray(endpoints)
            # map_blocks concatenates time blocks along axis 1 while keeping
            # whole rows along axis 0, so each output row already holds that
            # row's boundary sequence in time order (a singleton last time
            # block contributes a single value).
            flat_rows = endpoints.reshape(time.shape[0], -1)
            for r in range(time.shape[0]):
                _strictly_increasing(flat_rows[r])
    else:
        t = np.asarray(time)
        _strictly_increasing(t if t.ndim == 1 else t.reshape(-1, t.shape[-1])[0])


def _coerce_gap(value, time, name: str):
    """Normalize a gap/duration to integer nanoseconds or a numeric scalar."""
    if value is None:
        return None
    if time.dtype.kind == "M":
        if isinstance(value, np.timedelta64):
            gap = value
        elif isinstance(value, pd.Timedelta):
            gap = value.to_timedelta64()
        else:
            try:
                gap = pd.Timedelta(value).to_timedelta64()
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"`{name}` must be timedelta-like when `time` is datetime64; "
                    f"got {type(value).__name__}."
                ) from e
        return int(gap.astype("timedelta64[ns]").astype(np.int64))
    gap = np.asarray(value)
    if np.issubdtype(gap.dtype, np.timedelta64):
        raise TypeError(f"`{name}` is timedelta-like but `time` is not a datetime64 axis.")
    if gap.ndim != 0 or gap < 0:
        raise ValueError(f"`{name}` must be a non-negative scalar.")
    return gap


# ---------------------------------------------------------------------------
# Entity factorization
# ---------------------------------------------------------------------------


def _factorize_entities(by, shape: tuple[int, ...], expected_entities):
    """Return (integer codes broadcastable to shape, raw label lookup).

    Dask label arrays are factorized blockwise (the expected entity list
    makes every block use identical codes); numpy arrays are factorized
    eagerly.
    """
    if by is None:
        return np.zeros(shape, dtype=np.int64), np.array([None], dtype=object)
    if is_duck_dask_array(by):
        if expected_entities is None:
            raise ValueError(
                "`expected_entities` is required when `by` is a dask array, "
                "so entity factorization can stay blockwise."
            )
        import dask.array

        from .factorize import _lazy_factorize_wrapper

        labels_index = pd.Index(expected_entities)
        group_idx = dask.array.map_blocks(
            _lazy_factorize_wrapper,
            by,
            expected_groups=(labels_index,),
            axes=tuple(range(by.ndim)),
            fastpath=True,
            reindex=True,
            sort=True,
            meta=np.array([], dtype=np.int64),
        )
        # labels outside the expected index are coded -1 by the factorizer
        group_idx = group_idx.astype(np.int64, copy=False)
        lookup = labels_index.to_numpy()
        return group_idx, lookup

    axes = tuple(range(by.ndim))
    if expected_entities is not None:
        expected = (pd.Index(expected_entities),)
        group_idx, found, *_ = factorize_(
            (by,),
            (axes,),
            expected_groups=expected,
            reindex=True,
            sort=True,
        )
        codes = group_idx.astype(np.int64, copy=False)
        (labels_index,) = found
        lookup = (
            labels_index.to_numpy()
            if hasattr(labels_index, "to_numpy")
            else np.asarray(labels_index)
        )
        return codes, lookup

    # Eager factorization without an expected index: pd.factorize yields a
    # dense 0..k-1 coding and -1 for NaN labels (the missing-entity sentinel).
    flat = np.asarray(by).reshape(-1)
    codes_flat, uniques = pd.factorize(flat, sort=True)
    lookup = np.asarray(uniques)
    codes = codes_flat.astype(np.int64).reshape(by.shape)
    return codes, lookup


# ---------------------------------------------------------------------------
# Run-length kernel (pure numpy, operates on a 2-D (row, time) block)
# ---------------------------------------------------------------------------


def _previous_positions(mask: np.ndarray) -> np.ndarray:
    """Column index of the strictly-previous cell where ``mask`` is True."""
    nrows, n = mask.shape
    pos = np.where(mask, np.arange(n), -1)
    out = np.empty_like(pos)
    running = np.full(nrows, -1, dtype=np.int64)
    for j in range(n):
        out[:, j] = running
        running = np.maximum(running, pos[:, j])
    return out


def _label_rows(
    flags: np.ndarray,
    time: np.ndarray,
    entity_codes: np.ndarray,
    *,
    max_gap_num,
    missing_policy: str,
) -> np.ndarray:
    """Assign 1-based local event labels within a block.

    An exceeding, observed position starts a new event whenever it is the
    first exceeding position or, relative to the strictly previous
    *exceeding* position:

    * the entity changed;
    * an observed non-exceedance or (when missing values interrupt) a
      missing value lies in between -- this always splits when no
      ``max_gap`` is given;
    * the timestamp gap between the two exceedances exceeds ``max_gap``.

    With ``missing_policy="ignore"`` missing positions are transparent: the
    gap is measured straight between surrounding exceedances.
    """
    nrows, n = flags.shape
    labels = np.full((nrows, n), NO_EVENT, dtype=np.int64)
    missing = np.asarray(isnull(flags))
    exceeds = np.asarray(flags, dtype=bool) & ~missing
    if not exceeds.any():
        return labels

    # predecessor tracks: previous observed cell and previous exceedance
    prev_obs = _previous_positions(~missing)
    prev_exc = _previous_positions(exceeds)
    has_prev_exc = prev_exc >= 0
    safe_exc = np.maximum(prev_exc, 0)
    safe_obs = np.maximum(prev_obs, 0)

    if time.ndim == 1:
        t = np.broadcast_to(time, (nrows, n))
    else:
        t = time
    tnum = (
        t.astype("datetime64[ns]").astype(np.float64)
        if t.dtype.kind == "M"
        else t.astype(np.float64, copy=False)
    )
    # gap is always between consecutive exceeding observations
    exc_time = np.take_along_axis(tnum, safe_exc, axis=1)
    gap = tnum - exc_time

    row_idx = np.arange(nrows)[:, None]
    entity_change = entity_codes != entity_codes[row_idx, safe_exc]

    # Whether any missing cell lies strictly between the previous exceedance
    # and the current position: count missing cells up to i minus those up
    # to the previous exceedance (cumulative counts, fully vectorized).
    missing_cum = np.cumsum(missing, axis=1, dtype=np.int64)
    missing_at_prev_exc = np.take_along_axis(missing_cum, safe_exc, axis=1)
    missing_between = (missing_cum - missing_at_prev_exc) > 0
    observed_break = np.take_along_axis(~exceeds, safe_obs, axis=1) & np.take_along_axis(
        ~missing, safe_obs, axis=1
    )
    if max_gap_num is None:
        # no gap tolerance: observed non-exceedance always breaks; missing
        # cells break only when they interrupt
        adjacency_break = observed_break | (
            (missing_policy == "interrupt") & missing_between
        )
        gap_break = False
    else:
        # gap rule governs observed non-exceedance; interrupting missing
        # cells still break regardless of the gap
        adjacency_break = (missing_policy == "interrupt") & missing_between
        gap_break = gap > float(max_gap_num)

    starts = exceeds & (
        ~has_prev_exc | entity_change | adjacency_break | gap_break
    )
    run_number = np.cumsum(starts, axis=1, dtype=np.int64)
    labels[exceeds] = run_number[exceeds]
    return labels


# ---------------------------------------------------------------------------
# Block descriptors
# ---------------------------------------------------------------------------


def _row_descriptors(
    labels: np.ndarray,
    time_1d: np.ndarray,
    entity_lookup: np.ndarray,
    entity_codes_row: np.ndarray,
    *,
    row_index: int,
    block_index: int,
    block_offset: int,
) -> dict[str, Any]:
    """Compact boundary descriptor for the local events of one block row.

    Positions are reported as *global* column indices (local index plus the
    block's offset along the time axis), so cross-block adjacency does not
    depend on the block sizes.
    """
    events: list[dict[str, Any]] = []
    for local_id in np.unique(labels[labels > 0]):
        idx = np.flatnonzero(labels == local_id)
        first, last = int(idx[0]), int(idx[-1])
        code = int(entity_codes_row[first])
        entity = entity_lookup[code] if code >= 0 else np.nan
        events.append(
            {
                "local_id": int(local_id),
                "entity": entity,
                "start_time": time_1d[first],
                "end_time": time_1d[last],
                "nobs": int(idx.size),
                "first_pos": first + block_offset,
                "last_pos": last + block_offset,
            }
        )
    return {
        "row": row_index,
        "block": block_index,
        "events": events,
    }


def _descriptor_kernel(
    labels: np.ndarray,
    time_block: np.ndarray,
    entity_codes: np.ndarray,
    entity_lookup: np.ndarray,
    *,
    row_offset: int,
    block_index: int,
    block_offset: int,
) -> np.ndarray:
    """map_blocks kernel: object array of boundary descriptors per row."""
    nrows = labels.shape[0]
    t = time_block if time_block.ndim == 1 else time_block.reshape(nrows, -1)
    out = np.empty(nrows, dtype=object)
    for r in range(nrows):
        out[r] = _row_descriptors(
            labels[r],
            t if t.ndim == 1 else t[r],
            entity_lookup,
            entity_codes[r],
            row_index=row_offset + r,
            block_index=block_index,
            block_offset=block_offset,
        )
    return out


def _missing_positions_kernel(flags: np.ndarray, *, block_offset: int) -> np.ndarray:
    """Object array: global missing column positions within each block row."""
    nrows = flags.shape[0]
    missing = np.asarray(isnull(flags)).reshape(nrows, -1)
    out = np.empty(nrows, dtype=object)
    for r in range(nrows):
        out[r] = (np.flatnonzero(missing[r]) + block_offset).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Cross-block merge
# ---------------------------------------------------------------------------


def _entity_equal(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(np.array_equal(a, b))


def _count_missing_between(
    missing_positions: list[list[np.ndarray]], row: int, lo: int, hi: int
) -> int:
    """Number of missing cells with global positions in (lo, hi)."""
    total = 0
    for block in missing_positions:
        positions = block[row]
        total += int(np.count_nonzero((positions > lo) & (positions < hi)))
    return total


def _merge_events(
    descriptors: list[list[dict[str, Any]]],
    missing_positions: list[list[np.ndarray]],
    *,
    nrows: int,
    nblocks: int,
    max_gap_num,
    missing_policy: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[int, int]]]:
    """Merge block-local events into per-(row, entity) 1-based global events.

    Only the first local event of block ``b`` can continue the last event of
    block ``b - 1`` in the same row; everything else starts a fresh event.
    With ``max_gap=None`` the merge additionally requires direct adjacency
    of observed exceedances across the boundary; otherwise only the
    timestamp gap (and, when missing values interrupt, their absence)
    matters.
    """
    # Per-(row, entity) 1-based ids restart for each entity, but event records
    # are keyed by the globally-unique (row, entity, id) tuple so distinct
    # entities/rows can never share a record (a flat gid-indexed list would
    # collide because ids restart independently).
    next_id: dict[tuple[int, Any], int] = {}
    relabel: dict[tuple[int, int], dict[int, int]] = {}
    global_events: dict[tuple[int, Any, int], dict[str, Any]] = {}
    last_event: dict[int, dict[str, Any]] = {}

    for b in range(nblocks):
        for r in range(nrows):
            block_map: dict[int, int] = {}
            desc = descriptors[b][r]
            for k, local in enumerate(desc["events"]):
                key = (r, local["entity"])
                gid: int | None = None
                if k == 0 and b > 0 and r in last_event:
                    prev = last_event[r]
                    can_join = _entity_equal(prev["entity"], local["entity"])
                    if can_join:
                        gap = local["start_time"] - prev["end_time"]
                        gap_num = (
                            int(gap.astype("timedelta64[ns]").astype(np.int64))
                            if isinstance(gap, np.timedelta64)
                            else int(gap)
                        )
                        # Global column positions make the between-cell count
                        # independent of block boundaries and of empty blocks
                        # between the two events.
                        lo, hi = prev["last_pos"], local["first_pos"]
                        cells_between = hi - lo - 1
                        n_missing_between = _count_missing_between(
                            missing_positions, r, lo, hi
                        )
                        if max_gap_num is None:
                            # Direct adjacency: every cell strictly between the
                            # boundary exceedances must be missing (transparent
                            # under "ignore"); an observed non-exceedance always
                            # breaks the run, and under "interrupt" a missing
                            # cell breaks too.
                            if missing_policy == "ignore":
                                can_join = n_missing_between == cells_between
                            else:
                                can_join = cells_between == 0
                        else:
                            can_join = gap_num <= int(max_gap_num)
                            if can_join and missing_policy == "interrupt":
                                can_join = n_missing_between == 0
                    if can_join:
                        gid = prev["global_id"]
                        ev = global_events[(r, local["entity"], gid)]
                        ev["end_time"] = local["end_time"]
                        ev["nobs"] += local["nobs"]
                if gid is None:
                    gid = next_id.get(key, 0) + 1
                    next_id[key] = gid
                    global_events[(r, local["entity"], gid)] = {
                        "event_id": gid,
                        "row": r,
                        "entity": local["entity"],
                        "start_time": local["start_time"],
                        "end_time": local["end_time"],
                        "nobs": local["nobs"],
                    }
                block_map[local["local_id"]] = gid
                if k == len(desc["events"]) - 1:
                    last_event[r] = {
                        "global_id": gid,
                        "entity": local["entity"],
                        "end_time": local["end_time"],
                        "start_time": local["start_time"],
                        "last_pos": local["last_pos"],
                        "first_pos": local["first_pos"],
                        "block": b,
                    }
            relabel[(b, r)] = block_map
    # preserve insertion order (block/row scan) as a list of event records
    return list(global_events.values()), relabel


def _filter_events(global_events, relabel, min_points, min_duration_num):
    """Restrict relabel maps to events passing the min-length contracts."""
    if min_points is None and min_duration_num is None:
        return relabel
    keep: set[int] = set()
    for ev in global_events:
        if min_points is not None and ev["nobs"] < min_points:
            continue
        if min_duration_num is not None:
            duration = ev["end_time"] - ev["start_time"]
            d_num = (
                int(duration.astype("timedelta64[ns]").astype(np.int64))
                if isinstance(duration, np.timedelta64)
                else int(duration)
            )
            if d_num < int(min_duration_num):
                continue
        keep.add(ev["event_id"])
    return {
        key: {lid: gid for lid, gid in mapping.items() if gid in keep}
        for key, mapping in relabel.items()
    }


# ---------------------------------------------------------------------------
# Relabel application
# ---------------------------------------------------------------------------


def _relabel_kernel(labels: np.ndarray, mappings: np.ndarray) -> np.ndarray:
    """Relabel one 2-D block; ``mappings[r]`` is that row's id map."""
    out = np.full(labels.shape, NO_EVENT, dtype=np.int64)
    for r in range(labels.shape[0]):
        for local_id, gid in mappings[r].items():
            out[r][labels[r] == local_id] = gid
    return out


def _apply_relabel(
    local_labels,
    relabel: dict[tuple[int, int], dict[int, int]],
    *,
    nrows: int,
    nblocks: int,
    row_chunks: tuple[int, ...],
    time_chunks: tuple[int, ...],
):
    """Apply cross-block relabel maps eagerly or as a lazy dask layer.

    Maps are tiny (one entry per block-local event), so per-task memory is
    independent of the time-axis length.
    """
    if not is_duck_dask_array(local_labels):
        out = np.full(local_labels.shape, NO_EVENT, dtype=np.int64)
        cols = np.cumsum((0,) + time_chunks)
        for b in range(nblocks):
            sl = slice(int(cols[b]), int(cols[b + 1]))
            for r in range(nrows):
                for local_id, gid in relabel.get((b, r), {}).items():
                    out[r, sl][local_labels[r, sl] == local_id] = gid
        return out

    import dask.array
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph

    row_offsets = np.cumsum((0,) + row_chunks)
    out_name = f"event-relabel-{tokenize(local_labels, relabel)}"
    # local_labels is 2-D with blocks keyed (row_block, time_block); each
    # block holds every row of that row-block, so attach one map array per
    # block and relabel all rows inside the kernel.
    graph: dict[tuple, tuple] = {}
    for ib, rb in enumerate(row_chunks):
        for b in range(nblocks):
            mappings = np.empty(rb, dtype=object)
            for lr in range(rb):
                r = int(row_offsets[ib]) + lr
                mappings[lr] = relabel.get((b, r), {})
            graph[(out_name, ib, b)] = (
                _relabel_kernel, (local_labels.name, ib, b), mappings,
            )
    chunks = (tuple(row_chunks), time_chunks)
    hl = HighLevelGraph.from_collections(out_name, graph, dependencies=[local_labels])
    return dask.array.Array(hl, out_name, chunks, meta=np.array([], dtype=np.int64))


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _move_axis_to_end(array, axis: int):
    if axis == array.ndim - 1:
        return array, tuple(range(array.ndim))
    order = tuple(ax for ax in range(array.ndim) if ax != axis) + (axis,)
    if is_duck_dask_array(array):
        return array.transpose(order), order
    return np.asarray(array).transpose(order), order


def _fold_missing(flags: np.ndarray, missing: np.ndarray | None) -> np.ndarray:
    """Return a float copy with explicit missing positions set to NaN."""
    out = np.asarray(flags).astype(float, copy=True)
    if missing is not None:
        if missing.shape != flags.shape:
            raise ValueError("`missing` must have the same shape as `flags`.")
        out[np.asarray(missing, dtype=bool)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def label_events(
    flags,
    time,
    by=None,
    *,
    axis: int = -1,
    max_gap=None,
    min_points: int | None = None,
    min_duration=None,
    missing_policy: str = "interrupt",
    missing=None,
    expected_entities=None,
):
    """Label consecutive exceedance events along an ordered time axis.

    Parameters
    ----------
    flags : array_like or dask.array
        Boolean array marking exceedance observations. Float arrays are
        accepted: nonzero values count as exceedance, NaN values count as
        missing observations.
    time : array_like or dask.array
        Timestamps aligned with ``axis`` (1-D or broadcastable to ``flags``).
        Must be strictly increasing along ``axis``; duplicate or
        non-monotonic timestamps raise ``ValueError``.
    by : array_like or dask.array, optional
        Entity labels, aligned like the ``by`` argument of
        :func:`flox.groupby_reduce` (1-D along ``axis`` or full-shape).
        Events never span different entities. When omitted a single implicit
        entity is used.
    axis : int, optional
        Time axis of ``flags`` (default: last axis).
    max_gap : timedelta-like or float, optional
        Maximum gap between consecutive *observed* exceedances for them to
        stay in one event. ``None`` (default) requires direct adjacency of
        observed exceedances.
    min_points : int, optional
        Drop events with fewer than this many exceeding observations.
    min_duration : timedelta-like or float, optional
        Drop events whose ``end_time - start_time`` is shorter than this.
    missing_policy : {"interrupt", "ignore"}, default: "interrupt"
        Whether missing observations interrupt a run. With ``"ignore"``
        missing positions are transparent and gaps span across them.
    missing : array_like, optional
        Extra boolean mask (same shape as ``flags``) of missing observations.
    expected_entities : sequence, optional
        All possible entity labels; required when ``by`` is a dask array.
        Note that blockwise factorization of an object-dtype dask ``by``
        containing NaN labels mixed with strings is not supported upstream
        (``numpy.searchsorted`` cannot compare the mixed types); pass a
        numeric/string dtype without NaNs, or factorize eagerly.

    Returns
    -------
    labels : numpy.ndarray or dask.array
        Int64 array like ``flags``. Retained events carry positive 1-based
        ids (unique per entity); every other observation is ``-1``. Dask
        inputs return a lazy array whose graph only materializes block
        boundary descriptors, never the full time axis.
    entity_codes : numpy.ndarray or dask.array
        Factorized entity labels aligned with ``flags`` (all zeros without
        ``by``). The pair ``(entity_codes, labels)`` uniquely identifies an
        event and is the grouping key consumed by :func:`reduce_events`.
    """
    if missing_policy not in ("interrupt", "ignore"):
        raise ValueError(f"missing_policy must be 'interrupt' or 'ignore', got {missing_policy!r}.")
    if min_points is not None and (not isinstance(min_points, int) or min_points < 1):
        raise ValueError("min_points must be a positive integer.")
    if min_duration is not None and np.any(np.asarray(min_duration) < 0):
        raise ValueError("min_duration must be non-negative.")

    axis_ = axis % flags.ndim
    flags_t, order = _move_axis_to_end(flags, axis_)
    original_shape = flags_t.shape
    time_t = time if getattr(time, "ndim", 1) == 1 else _move_axis_to_end(time, axis_)[0]
    by_t = None if by is None else (
        by if getattr(by, "ndim", 1) == 1 else _move_axis_to_end(by, axis_)[0]
    )
    missing_t = None if missing is None else (
        missing if getattr(missing, "ndim", 1) == 1 else _move_axis_to_end(missing, axis_)[0]
    )

    is_dask = (
        is_duck_dask_array(flags)
        or is_duck_dask_array(time)
        or (by_t is not None and is_duck_dask_array(by_t))
    )
    gap_ref = time_t if not is_duck_dask_array(time_t) else time
    max_gap_num = _coerce_gap(max_gap, gap_ref, "max_gap")
    min_duration_num = _coerce_gap(min_duration, gap_ref, "min_duration")

    # Normalize to 2-D (row x time); both backends work on that layout.
    ntime = original_shape[-1]
    nrows = math.prod(original_shape[:-1])
    if is_dask:
        import dask.array

        flags_2d = flags_t.reshape(nrows, ntime)
        if getattr(time_t, "ndim", 1) == 1:
            time_2d = dask.array.broadcast_to(
                time_t, (nrows, ntime), chunks=((nrows,), time_t.chunks[0])
            )
        else:
            time_2d = time_t.reshape(nrows, ntime)
        by_2d = None if by_t is None else _broadcast_by(by_t, nrows, ntime, True)
        missing_2d = None if missing_t is None else missing_t.reshape(nrows, ntime)
        labels, entity_codes = _label_events_dask(
            flags_2d, time_2d, by_2d, missing_2d,
            max_gap_num=max_gap_num, min_points=min_points,
            min_duration_num=min_duration_num, missing_policy=missing_policy,
            expected_entities=expected_entities,
        )
    else:
        flags_2d = np.asarray(flags_t).reshape(nrows, ntime)
        time_2d = np.asarray(time_t)
        time_2d = (
            np.broadcast_to(time_2d, (nrows, ntime))
            if time_2d.ndim == 1
            else time_2d.reshape(nrows, ntime)
        )
        by_2d = None if by_t is None else _broadcast_by(np.asarray(by_t), nrows, ntime, False)
        missing_2d = None if missing_t is None else np.asarray(missing_t).reshape(nrows, ntime)
        labels, entity_codes = _label_events_numpy(
            flags_2d, time_2d, by_2d, missing_2d,
            max_gap_num=max_gap_num, min_points=min_points,
            min_duration_num=min_duration_num, missing_policy=missing_policy,
            expected_entities=expected_entities,
        )
    labels = labels.reshape(original_shape)
    entity_codes = entity_codes.reshape(original_shape)
    if order != tuple(range(flags.ndim)):
        # undo the time-axis-to-end transposition
        inverse = np.argsort(order).tolist()
        labels = labels.transpose(inverse)
        entity_codes = entity_codes.transpose(inverse)
    return labels, entity_codes


def _entity_code_lookup(by, expected_entities) -> dict[int, Any]:
    """Build a factorized-code -> raw-entity-label mapping for the table."""
    if by is None:
        return {0: None}
    if expected_entities is not None:
        return dict(enumerate(np.asarray(expected_entities)))
    flat = np.asarray(by).reshape(-1)
    _, uniques = pd.factorize(flat, sort=True)
    # pd.factorize puts NaN at code -1; keep a mapping only for real labels
    return dict(enumerate(np.asarray(uniques)))


def _broadcast_by(by, nrows: int, ntime: int, is_dask: bool):
    """Broadcast entity labels to the 2-D (row, time) working shape.

    A size-1 time axis is expanded here so factorization runs on the full
    broadcast shape and never mistakes a trailing column value for missing.
    """
    target = (nrows, ntime)
    if by.shape == target:
        return by
    if is_dask:
        import dask.array

        # row axis may itself be size-1; preserve sensible chunking
        row_chunks = by.chunks[0] if by.shape[0] == nrows else (nrows,)
        time_chunks = by.chunks[-1] if by.shape[-1] == ntime else (ntime,)
        return dask.array.broadcast_to(by, target, chunks=(row_chunks, time_chunks))
    # materialize the broadcast so factorization sees every entity label
    # (a (n, 1) column would otherwise expose only its first row)
    return np.broadcast_to(by, target).copy()


def _label_events_numpy(
    flags, time, by, missing, *,
    max_gap_num, min_points, min_duration_num, missing_policy, expected_entities,
):
    _validate_time(time, flags.ndim)
    flags = _fold_missing(flags, missing)
    entity_codes, lookup = _factorize_entities(by, flags.shape, expected_entities)
    entity_codes = np.broadcast_to(entity_codes, flags.shape)

    nrows = math.prod(flags.shape[:-1])
    ntime = flags.shape[-1]
    f2d = flags.reshape(nrows, ntime)
    c2d = np.broadcast_to(entity_codes, flags.shape).reshape(nrows, ntime)
    t2d = time if time.ndim == 1 else time.reshape(nrows, ntime)
    local = _label_rows(
        f2d, t2d, c2d, max_gap_num=max_gap_num, missing_policy=missing_policy
    )

    descriptors = [
        [
            _row_descriptors(
                local[r],
                time if time.ndim == 1 else t2d[r],
                lookup, c2d[r], row_index=r, block_index=0, block_offset=0,
            )
            for r in range(nrows)
        ]
    ]
    missing_positions = [
        [np.flatnonzero(np.asarray(isnull(f2d[r]))).astype(np.int64) for r in range(nrows)]
    ]
    global_events, relabel = _merge_events(
        descriptors, missing_positions,
        nrows=nrows, nblocks=1, max_gap_num=max_gap_num, missing_policy=missing_policy,
    )
    relabel = _filter_events(global_events, relabel, min_points, min_duration_num)
    labels = _apply_relabel(
        local, relabel, nrows=nrows, nblocks=1,
        row_chunks=(nrows,), time_chunks=(ntime,),
    )
    return labels, entity_codes


def _label_events_dask(
    flags, time, by, missing, *,
    max_gap_num, min_points, min_duration_num, missing_policy, expected_entities,
):
    import dask.array

    inputs = [flags]
    index_lists = [tuple(range(flags.ndim))]
    if time.ndim == flags.ndim:
        inputs.append(time)
        index_lists.append(tuple(range(flags.ndim)))
    if by is not None:
        inputs.append(by)
        index_lists.append(tuple(range(flags.ndim - by.ndim, flags.ndim)))
    if missing is not None:
        inputs.append(missing)
        index_lists.append(tuple(range(flags.ndim - missing.ndim, flags.ndim)))
    unified = dask.array.unify_chunks(
        *[item for pair in zip(inputs, index_lists) for item in pair]
    )[1]
    flags_c = unified[0]
    cursor = 1
    time_c = time
    if time.ndim == flags.ndim:
        time_c = unified[cursor]
        cursor += 1
    by_c = None
    if by is not None:
        by_c = unified[cursor]
        cursor += 1
    missing_c = unified[cursor] if missing is not None else None

    if time.ndim == 1:
        _validate_time(time, 1)
        time_c = dask.array.broadcast_to(time, flags_c.shape, chunks=flags_c.chunks)
    else:
        _validate_time(time_c, flags_c.ndim)

    entity_codes, lookup = _factorize_entities(by_c, flags_c.shape, expected_entities)
    if hasattr(entity_codes, "chunks"):
        if entity_codes.ndim == 1:
            entity_codes = dask.array.broadcast_to(
                entity_codes, flags_c.shape, chunks=flags_c.chunks
            )
        _, (entity_codes,) = dask.array.unify_chunks(
            entity_codes, tuple(range(flags_c.ndim))
        )
    else:
        entity_codes = dask.array.from_array(
            np.broadcast_to(entity_codes, flags_c.shape), chunks=flags_c.chunks
        )

    flags_c = dask.array.map_blocks(
        _fold_missing, flags_c, missing_c,
        dtype=np.float64, meta=np.array([], dtype=np.float64),
    )

    local = dask.array.map_blocks(
        lambda fl, tm, ent: _label_rows(
            fl, tm, ent, max_gap_num=max_gap_num, missing_policy=missing_policy
        ),
        flags_c, time_c, entity_codes,
        dtype=np.int64, meta=np.array([], dtype=np.int64),
    )

    # local is 2-D (row x time); chunks[:-1] is the single row-axis tuple.
    row_chunks: tuple[int, ...] = tuple(local.chunks[0])
    time_chunks: tuple[int, ...] = tuple(local.chunks[-1])
    nrows = sum(row_chunks)
    nblocks = len(time_chunks)
    row_offsets = np.cumsum((0,) + row_chunks)
    time_offsets = np.cumsum((0,) + time_chunks)

    descriptor_layers = []
    missing_layers = []
    for ib, rb in enumerate(row_chunks):
        for b in range(nblocks):
            lb = local.blocks[ib, b]
            tb = time_c.blocks[ib, b] if time_c.ndim > 1 else time_c.blocks[b]
            eb = entity_codes.blocks[ib, b]
            block_offset = int(time_offsets[b])
            descriptor_layers.append(
                dask.array.map_blocks(
                    _descriptor_kernel, lb, tb, eb, lookup,
                    row_offset=int(row_offsets[ib]), block_index=b,
                    block_offset=block_offset,
                    dtype=object, meta=np.array([], dtype=object), drop_axis=1,
                )
            )
            missing_layers.append(
                dask.array.map_blocks(
                    _missing_positions_kernel, flags_c.blocks[ib, b],
                    block_offset=block_offset,
                    dtype=object, meta=np.array([], dtype=object), drop_axis=1,
                )
            )

    # The only materialization: O(events) descriptors and O(missing points)
    # positions -- never the full time/value arrays.
    descriptor_values = [layer.compute() for layer in descriptor_layers]
    missing_values = [layer.compute() for layer in missing_layers]

    descriptors = [
        [{"row": r, "block": b, "events": []} for r in range(nrows)]
        for b in range(nblocks)
    ]
    missing_positions = [[np.array([], dtype=np.int64) for _ in range(nrows)] for _ in range(nblocks)]
    k = 0
    for ib, rb in enumerate(row_chunks):
        for b in range(nblocks):
            for lr, desc in enumerate(descriptor_values[k].tolist()):
                descriptors[b][int(row_offsets[ib]) + lr] = desc
            for lr, positions in enumerate(missing_values[k].tolist()):
                missing_positions[b][int(row_offsets[ib]) + lr] = positions
            k += 1

    global_events, relabel = _merge_events(
        descriptors, missing_positions,
        nrows=nrows, nblocks=nblocks,
        max_gap_num=max_gap_num, missing_policy=missing_policy,
    )
    relabel = _filter_events(global_events, relabel, min_points, min_duration_num)
    labels = _apply_relabel(
        local, relabel,
        nrows=nrows, nblocks=nblocks, row_chunks=row_chunks, time_chunks=time_chunks,
    )
    return labels, entity_codes


# ---------------------------------------------------------------------------
# Reductions over events
# ---------------------------------------------------------------------------


def reduce_events(
    values,
    event_labels,
    entity_codes,
    funcs=("max",),
    *,
    axis: int = -1,
    **kwargs,
):
    """Reduce the observations of each event with the existing groupby engine.

    Event ids are used *as real group labels*. A composite
    ``entity_code * stride + event_id`` key (globally unique per event) is
    factorized and passed to :func:`flox.groupby_reduce`, so every returned
    statistic aligns exactly with the returned ``(entity, event_id)`` pairs.
    Observations labeled ``-1`` are excluded. Arrays are flattened before
    reduction, which keeps one group per event (no empty per-row group cells)
    and works for reductions like ``mean`` without a fill value.

    Parameters
    ----------
    values : array_like or dask.array
        Observations to reduce, broadcastable to ``event_labels``.
    event_labels : array_like or dask.array
        Labels from :func:`label_events`.
    entity_codes : array_like or dask.array
        Entity codes from :func:`label_events`.
    funcs : str, Aggregation or sequence thereof, default: ("max",)
        Reduction(s) to compute per event.
    axis : int, optional
        Time axis (accepted for API symmetry; reductions are over the
        flattened event observations).
    **kwargs
        Extra keyword arguments for :func:`flox.groupby_reduce`.

    Returns
    -------
    results : dict[str, ndarray or dask.array]
        Reduction name -> aligned per-event reduction.
    pairs : ndarray or dask.array
        ``(n_events, 2)`` array of ``(entity_code, event_id)`` pairs, in the
        same order as every reduction array.
    """
    del axis  # reductions are over the flattened observations
    if isinstance(funcs, str | Aggregation):
        funcs = (funcs,)

    lazy = is_duck_dask_array(event_labels)
    if lazy:
        import dask.array

        labels = event_labels
        entities = entity_codes
        valid = labels >= 0
        max_id = int(dask.array.where(valid, labels, 0).max().compute())
        stride = max_id + 1
        composite = dask.array.where(valid, entities * stride + labels, -1)
        values_b = dask.array.broadcast_to(values, labels.shape) if values.shape != labels.shape else values
        flat_values = values_b.reshape(-1)
        flat_composite = composite.reshape(-1)
    else:
        labels = np.asarray(event_labels)
        entities = np.broadcast_to(np.asarray(entity_codes), labels.shape)
        values_b = np.broadcast_to(np.asarray(values), labels.shape)
        valid = labels >= 0
        stride = int(labels[valid].max()) + 1 if valid.any() else 1
        composite = np.where(valid, entities * stride + labels, -1)
        flat_values = np.asarray(values_b).reshape(-1)
        flat_composite = composite.reshape(-1)

    results: dict[str, Any] = {}
    groups_out = None
    for func in funcs:
        name = func if isinstance(func, str) else func.name
        reduced, groups = groupby_reduce(
            flat_values, flat_composite, func=func, **kwargs
        )
        results[name] = reduced
        groups_out = groups

    if is_duck_dask_array(groups_out):
        import dask.array

        groups_sized = groups_out.compute_chunk_sizes()
        keep = groups_sized >= 0
        safe_groups = dask.array.where(keep, groups_sized, 0)
        entity_part = (safe_groups // stride)[:, None]
        event_part = (safe_groups % stride)[:, None]
        pairs_all = dask.array.concatenate(
            [entity_part, event_part], axis=-1, allow_unknown_chunksizes=True
        ).compute_chunk_sizes()
        pairs = pairs_all[keep]
        results = {
            name: reduced.compute_chunk_sizes()[keep] for name, reduced in results.items()
        }
    else:
        groups_out = np.asarray(groups_out)
        keep = groups_out >= 0
        groups_out = groups_out[keep]
        pairs = np.stack([groups_out // stride, groups_out % stride], axis=-1)
        results = {name: np.asarray(reduced)[keep] for name, reduced in results.items()}
    return results, pairs


def find_events(
    flags,
    time,
    by=None,
    *,
    values=None,
    stat_funcs: Sequence[str] | None = ("max",),
    axis: int = -1,
    max_gap=None,
    min_points: int | None = None,
    min_duration=None,
    missing_policy: str = "interrupt",
    missing=None,
    expected_entities=None,
) -> EventTable:
    """Detect events and return aligned bounds, duration and statistics.

    Parameters mirror :func:`label_events`. Event labeling stays lazy for
    dask inputs; per-block descriptors and (when ``values`` is given)
    per-event reductions are then computed and assembled into one
    :class:`EventTable`.

    Parameters
    ----------
    values : array_like or dask.array, optional
        Observations for per-event reductions (e.g. the peak).
    stat_funcs : sequence of str, optional
        Reductions to include when ``values`` is given (default: peak max).
        Set to ``None`` or an empty sequence to skip.

    Returns
    -------
    EventTable
    """
    labels, entity_codes = label_events(
        flags, time, by,
        axis=axis, max_gap=max_gap, min_points=min_points, min_duration=min_duration,
        missing_policy=missing_policy, missing=missing, expected_entities=expected_entities,
    )
    if is_duck_dask_array(labels):
        labels_np = np.asarray(labels.compute())
        entity_np = np.broadcast_to(np.asarray(entity_codes.compute()), labels_np.shape)
        time_np = time.compute() if is_duck_dask_array(time) else np.asarray(time)
    else:
        labels_np = np.asarray(labels)
        entity_np = np.broadcast_to(np.asarray(entity_codes), labels_np.shape)
        time_np = np.asarray(time)

    nrows = math.prod(labels_np.shape[:-1])
    ntime = labels_np.shape[-1]
    lab2d = labels_np.reshape(nrows, ntime)
    ent2d = entity_np.reshape(nrows, ntime)

    # Map factorized entity codes back to raw labels for the human-readable
    # table. reduce_events pairs still use the integer codes by contract.
    code_to_label = _entity_code_lookup(by, expected_entities)
    stride = int(labels_np[labels_np > 0].max()) + 1 if (labels_np > 0).any() else 1

    event_ids: list[int] = []
    entity_codes_list: list[int] = []
    entities: list[Any] = []
    starts: list[Any] = []
    ends: list[Any] = []
    nobs: list[int] = []
    for r in range(nrows):
        t = time_np if time_np.ndim == 1 else time_np.reshape(nrows, ntime)[r]
        row_lab = lab2d[r]
        row_ent = ent2d[r]
        # ids restart per entity, so enumerate distinct (entity_code, id)
        # pairs rather than bare ids (two entities can share an id)
        event_mask = row_lab > 0
        composite = row_ent[event_mask].astype(np.int64) * stride + row_lab[event_mask]
        for comp in np.unique(composite):
            code = int(comp // stride)
            gid = int(comp % stride)
            idx = np.flatnonzero(event_mask & (row_ent == code) & (row_lab == gid))
            event_ids.append(gid)
            entity_codes_list.append(code)
            entities.append(code_to_label.get(code, np.nan))
            starts.append(t[idx[0]])
            ends.append(t[idx[-1]])
            nobs.append(int(idx.size))

    event_id_arr = np.asarray(event_ids, dtype=np.int64)
    entity_codes_arr = np.asarray(entity_codes_list, dtype=np.int64)
    entity_arr = np.asarray(entities)
    start_arr = np.asarray(starts)
    end_arr = np.asarray(ends)
    nobs_arr = np.asarray(nobs, dtype=np.int64)

    stats: dict[str, np.ndarray] = {}
    if values is not None and stat_funcs:
        results, pairs = reduce_events(
            values, labels_np, entity_np, funcs=tuple(stat_funcs), axis=axis
        )
        pair_to_pos = {
            (int(e), int(g)): i for i, (e, g) in enumerate(np.asarray(pairs).tolist())
        }
        order = [
            pair_to_pos[(int(e), int(g))]
            for e, g in zip(entity_codes_arr, event_id_arr)
        ]
        stats = {name: np.asarray(reduced)[order] for name, reduced in results.items()}

    return EventTable(
        event_id=event_id_arr,
        entity=entity_arr,
        start_time=start_arr,
        end_time=end_arr,
        nobs=nobs_arr,
        duration=end_arr - start_arr,
        stats=stats,
    )
