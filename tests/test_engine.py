"""Engine unit tests: causal classification, folding order and cycles."""

from app.engine import (
    BLOCKED,
    GROUNDED,
    INVALID,
    build_timeline,
    classify_sensor,
    evaluate_batch,
    sample_sensor,
)
from app.storage import BatchStore, StoredEvent


def _ev(rev, parent, start=0, end=10, value=1.0, withdrawn=False,
        batch="B", sensor="S", key=None):
    return StoredEvent(
        batch_id=batch,
        sensor_id=sensor,
        revision_id=rev,
        parent_id=parent,
        idempotency_key=key or f"k-{rev}",
        start_ns=start,
        end_ns=end,
        value=None if withdrawn else float(value),
        unit="u",
        withdrawn=withdrawn,
    )


def test_classify_root_missing_and_cycle():
    events = {
        "root": _ev("root", "ROOT"),
        "ok": _ev("ok", "root"),
        "orphan": _ev("orphan", "ghost"),
        "c1": _ev("c1", "c2"),
        "c2": _ev("c2", "c1"),
        "down": _ev("down", "c1"),
    }
    states, depths = classify_sensor(events)
    assert states["root"] == GROUNDED and depths["root"] == 0
    assert states["ok"] == GROUNDED and depths["ok"] == 1
    assert states["orphan"] == BLOCKED
    assert states["c1"] == INVALID and states["c2"] == INVALID
    assert states["down"] == INVALID


def test_self_loop_is_invalid():
    events = {"x": _ev("x", "x")}
    states, _ = classify_sensor(events)
    assert states["x"] == INVALID


def test_fold_order_depth_then_id():
    # Two incomparable children of root overlap at [20,30); deterministic
    # tie-break: lexicographically larger revision id applies last/wins.
    events = {
        "root": _ev("root", "ROOT", 0, 100, value=0),
        "a": _ev("a", "root", 20, 40, value=1),
        "b": _ev("b", "root", 10, 30, value=2),
    }
    states, depths = classify_sensor(events)
    segs = build_timeline(list(events.values()), depths)
    assert [(s.start_ns, s.end_ns, s.value, s.revision_id) for s in segs] == [
        (0, 10, 0.0, "root"),
        (10, 30, 2.0, "b"),
        (30, 40, 1.0, "a"),
        (40, 100, 0.0, "root"),
    ]


def test_descendant_beats_lexicographic_order():
    events = {
        "root": _ev("root", "ROOT", 0, 50, value=0),
        "z": _ev("z", "root", 0, 50, value=1),
        "a": _ev("a", "z", 0, 50, value=2),  # deeper -> wins regardless of id
    }
    states, depths = classify_sensor(events)
    segs = build_timeline(list(events.values()), depths)
    assert len(segs) == 1 and segs[0].value == 2.0 and segs[0].revision_id == "a"


def test_blocked_events_excluded_from_timeline_and_hash_stable():
    store = BatchStore()
    for e in [_ev("root", "ROOT", 0, 20, value=7.0), _ev("lost", "ghost", 0, 5, value=9.0)]:
        store.events[e.revision_id] = e
        store.idempotency[e.idempotency_key] = "x"
    doc1 = evaluate_batch("B", store)
    doc2 = evaluate_batch("B", store)
    assert doc1["timeline_hash"] == doc2["timeline_hash"]
    sensor = doc1["sensors"][0]
    assert len(sensor["segments"]) == 1
    assert sensor["segments"][0]["value"] == 7.0
    assert doc1["revisions_blocked"] == 1
    assert doc1["complete"] is False


def test_sample_sensor_point_states_and_chain():
    events = {
        "root": _ev("root", "ROOT", 0, 100, value=1.0),
        "child": _ev("child", "root", 10, 20, value=2.0),
        "w": _ev("w", "child", 12, 18, withdrawn=True),
    }
    at = lambda ts: sample_sensor("B", "S", events, ts)  # noqa: E731
    out = at(12)
    assert out["status"] == "withdrawn"
    assert out["value"] is None and out["unit"] is None
    assert out["revision_id"] == "w"
    assert out["parent_chain"] == ["w", "child", "root"]
    out = at(11)
    assert out["status"] == "calibrated" and out["value"] == 2.0
    assert out["revision_id"] == "child"
    out = at(50)
    assert out["status"] == "calibrated" and out["value"] == 1.0
    assert out["parent_chain"] == ["root"]
    out = at(100)  # half-open: end boundary is not covered
    assert out["status"] == "uncovered"
    assert out["revision_id"] is None
    assert out["parent_chain"] == [] and out["missing_ancestors"] == []


def test_sample_sensor_blocked_feature_and_missing_union():
    events = {
        "a": _ev("a", "missingA", 0, 100),
        "b": _ev("b", "a", 0, 100),  # deeper blocked revision wins
        "c": _ev("c", "missingC", 0, 100),
    }
    out = sample_sensor("B", "S", events, 50)
    assert out["status"] == "blocked"
    assert out["revision_id"] == "b"
    assert out["parent_chain"] == ["b", "a"]
    assert out["missing_ancestors"] == ["missingA", "missingC"]


def test_sample_sensor_blocked_tie_breaks_on_revision_id():
    events = {"a": _ev("a", "ghost", 0, 10), "b": _ev("b", "ghost", 0, 10)}
    out = sample_sensor("B", "S", events, 5)
    assert out["status"] == "blocked"
    assert out["revision_id"] == "b"  # same depth: larger id wins, as in the fold
    assert out["missing_ancestors"] == ["ghost"]


def test_sample_sensor_ignores_invalid_cycles():
    events = {"c1": _ev("c1", "c2", 0, 10), "c2": _ev("c2", "c1", 0, 10)}
    out = sample_sensor("B", "S", events, 5)
    assert out["status"] == "uncovered"
    assert out["revision_id"] is None
