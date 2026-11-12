from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class IdentityStatus(str, Enum):
    KNOWN = "Known"
    UNKNOWN = "Unknown"
    UNRESOLVED = "Unresolved"


@dataclass(frozen=True)
class PersonRecord:
    track_id: str
    identity_status: IdentityStatus
    name: str = "Not identified"
    upper_clothing: str = "upper clothing not determined"
    lower_clothing: str = "lower clothing not determined"
    carried_items: str = "items not determined"
    activity: str = "performing an undetermined activity"
    position: str = "an undetermined position"
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        if not self.track_id.strip():
            raise ValueError("track_id cannot be empty")
        if self.identity_status is IdentityStatus.KNOWN and (
            not self.name.strip() or self.name == "Not identified"
        ):
            raise ValueError("Known people require a database name")
        if self.identity_status is not IdentityStatus.KNOWN and self.name != "Not identified":
            raise ValueError("Unknown and unresolved people must use 'Not identified'")

    def identity_phrase(self, person_number: int) -> str:
        if self.identity_status is IdentityStatus.KNOWN:
            return f"{self.name} (known person)"
        if self.identity_status is IdentityStatus.UNKNOWN:
            return f"Unknown person {person_number}"
        return f"Person {person_number} (identity unresolved)"

    def description(self, person_number: int) -> str:
        subject = self.identity_phrase(person_number)
        visible_clothing = []
        if self.upper_clothing != "upper clothing not determined":
            visible_clothing.append(self.upper_clothing)
        if self.lower_clothing != "lower clothing not determined":
            visible_clothing.append(self.lower_clothing)

        clauses = []
        if visible_clothing:
            clothing = (
                visible_clothing[0] if len(visible_clothing) == 1
                else f"{visible_clothing[0]} and {visible_clothing[1]}"
            )
            clauses.append(f"is wearing {clothing}")
        if self.carried_items != "items not determined":
            if self.carried_items == "no carried items":
                clauses.append("has no visible carried items")
            else:
                clauses.append(f"is carrying {self.carried_items}")
        if self.activity != "performing an undetermined activity":
            clauses.append(f"is {self.activity}")
        if self.position != "an undetermined position":
            clauses.append(f"is at {self.position}")

        if not clauses:
            return f"{subject} is visible."
        if len(clauses) == 1:
            return f"{subject} {clauses[0]}."
        return f"{subject} {', '.join(clauses[:-1])}, and {clauses[-1]}."


@dataclass(frozen=True)
class SurveillanceEvent:
    event_id: str
    event_type: str
    date_time_with_timezone: str
    location: str
    source_id: str
    people: tuple[PersonRecord, ...] = field(default_factory=tuple)
    specific_observed_reason: str = "Not determined"
    first_detected_time: str = "Not determined"
    observed_duration: str = "Not determined"
    status: str = "Not determined"
    snapshot: str = "Not available"
    video_clip: str = "Not available"

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id cannot be empty")
        ids = [person.track_id for person in self.people]
        if len(ids) != len(set(ids)):
            raise ValueError("Every visible person must have a unique track_id")

    @property
    def sorted_people(self) -> tuple[PersonRecord, ...]:
        def key(person: PersonRecord):
            value = person.track_id
            return (0, int(value)) if value.isdigit() else (1, value.casefold())
        return tuple(sorted(self.people, key=key))

    @property
    def counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in IdentityStatus}
        for person in self.people:
            counts[person.identity_status.value] += 1
        result = {
            "total": len(self.people),
            "known": counts[IdentityStatus.KNOWN.value],
            "unknown": counts[IdentityStatus.UNKNOWN.value],
            "unresolved": counts[IdentityStatus.UNRESOLVED.value],
        }
        assert result["total"] == result["known"] + result["unknown"] + result["unresolved"]
        return result

    def as_dict(self) -> dict:
        return asdict(self)
