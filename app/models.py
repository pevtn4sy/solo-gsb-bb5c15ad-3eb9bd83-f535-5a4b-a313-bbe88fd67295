"""Pydantic schemas for the calibration timeline service.

Time coordinates are nanoseconds since an arbitrary batch-local epoch.
Intervals are half-open ``[start_ns, end_ns)`` so that abutting fragments do
not share any sample instant.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A revision root uses this literal as its (missing) parent.
ROOT_PARENT = "ROOT"

VersionId = Annotated[str, Field(min_length=1, max_length=200)]
BatchId = Annotated[str, Field(min_length=1, max_length=200)]
SensorId = Annotated[str, Field(min_length=1, max_length=200)]
IdemKey = Annotated[str, Field(min_length=1, max_length=200)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeasurementEvent(StrictModel):
    """One measurement-fragment revision delivered to the ingest endpoint."""

    batch_id: BatchId
    sensor_id: SensorId
    revision_id: VersionId
    # Parent revision for this sensor; the literal "ROOT" marks a causal root.
    parent_id: VersionId
    # Idempotency key; the same event payload is accepted exactly once in
    # effect and every repeated submission is acknowledged as a duplicate.
    idempotency_key: IdemKey
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=1)
    # Calibration value for the fragment; required unless the fragment is
    # withdrawn (withdrawn fragments erase instead of writing a value).
    value: Optional[float] = None
    unit: Optional[str] = Field(default=None, max_length=40)
    withdrawn: bool = False

    @model_validator(mode="after")
    def _check_interval_and_value(self) -> "MeasurementEvent":
        if self.revision_id == ROOT_PARENT:
            raise ValueError(f"revision_id may not be the reserved value {ROOT_PARENT!r}")
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be strictly greater than start_ns")
        if not self.withdrawn and self.value is None:
            raise ValueError("value is required when withdrawn is false")
        return self


class TimelineSegment(StrictModel):
    start_ns: int
    end_ns: int
    state: Literal["calibrated", "withdrawn"]
    # Populated for calibrated segments; null on withdrawn (erased) spans.
    value: Optional[float] = None
    unit: Optional[str] = None
    revision_id: str


class Gap(StrictModel):
    """A missing revision that blocks one or more known children."""

    batch_id: str
    sensor_id: str
    missing_revision_id: str
    awaited_by: list[str]
    # Revision that referenced the missing parent (may repeat across gaps).
    referenced_from: str


class SensorTimeline(StrictModel):
    sensor_id: str
    grounded: bool
    segments: list[TimelineSegment]
    segments_hash: str
    covered_start_ns: Optional[int] = None
    covered_end_ns: Optional[int] = None


class BatchResult(StrictModel):
    batch_id: str
    sensors: list[SensorTimeline]
    gaps: list[Gap]
    revisions_total: int
    revisions_grounded: int
    revisions_blocked: int
    revisions_invalid: int
    # True when every known revision is grounded (no gaps, no cycles).
    complete: bool
    timeline_hash: str


class IngestResponse(StrictModel):
    status: Literal["accepted", "duplicate", "conflict"]
    idempotency_key: str
    batch_id: str
    sensor_id: str
    revision_id: str
    result: BatchResult


class SampleResponse(StrictModel):
    """Effective state of one sensor at a single nanosecond instant."""

    batch_id: str
    sensor_id: str
    timestamp_ns: int
    status: Literal["calibrated", "withdrawn", "uncovered", "blocked"]
    # Effective value and unit; populated only when status is "calibrated".
    value: Optional[float] = None
    unit: Optional[str] = None
    # Winning revision (for "blocked": the featured blocked revision);
    # null only when status is "uncovered".
    revision_id: Optional[str] = None
    # Lineage from the winning revision (first element) back along parent_id
    # to the root revision (last element, its parent is the literal "ROOT").
    # For "blocked" it is the known prefix ending before the missing ancestor.
    parent_chain: list[str] = Field(default_factory=list)
    # Missing ancestor revision ids to backfill; populated for "blocked".
    missing_ancestors: list[str] = Field(default_factory=list)


class ResetResponse(StrictModel):
    status: Literal["reset"]
    batches_removed: int
    events_removed: int


class HealthResponse(StrictModel):
    status: Literal["ok"]
    storage: Literal["memory"]
    version: str
