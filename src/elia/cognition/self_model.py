from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


MAX_SELF_MODEL_ENTRIES = 64
MAX_BLIND_SPOTS = 32
MAX_CONSTRAINTS = 32
MAX_TASK_HISTORY = 32


@dataclass
class SelfModel:
    """Bounded operational self-description; not a model of subjective experience."""

    current_task: str = "idle"
    known_capabilities: dict[str, float] = field(default_factory=dict)
    blind_spots: set[str] = field(default_factory=set)
    domains: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    known_constraints: list[str] = field(default_factory=list)
    recent_history: list[str] = field(default_factory=list)
    task_counts: dict[str, int] = field(default_factory=dict)

    def set_capabilities(self, capabilities: dict[str, float]) -> None:
        self.known_capabilities = self._bounded_scores(capabilities, "capability")

    def set_constraints(self, constraints: list[str]) -> None:
        if not isinstance(constraints, list) or len(constraints) > MAX_CONSTRAINTS:
            raise ValueError(f"at most {MAX_CONSTRAINTS} constraints are allowed")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 160 for item in constraints):
            raise ValueError("constraints must contain 1 to 160 characters")
        self.known_constraints = list(dict.fromkeys(item.strip() for item in constraints))

    def register_domain(self, domain: str, confidence: float) -> None:
        self._set_bounded_score(self.domains, domain, confidence, "domain confidence")
        self._set_bounded_score(self.confidence, domain, confidence, "confidence")

    def add_blind_spot(self, item: str) -> None:
        if not isinstance(item, str) or not item.strip() or len(item) > 160:
            raise ValueError("blind spot must contain 1 to 160 characters")
        item = item.strip()
        self.blind_spots.add(item)
        if len(self.blind_spots) > MAX_BLIND_SPOTS:
            self.blind_spots.remove(sorted(self.blind_spots)[0])

    def update_task(self, task: str) -> None:
        if not isinstance(task, str) or not task.strip() or len(task) > 120:
            raise ValueError("task must be 1 to 120 characters")
        task = task.strip()
        self.current_task = task
        self.recent_history.append(task)
        self.recent_history = self.recent_history[-MAX_TASK_HISTORY:]
        self.task_counts[task] = self.task_counts.get(task, 0) + 1
        if len(self.task_counts) > MAX_SELF_MODEL_ENTRIES:
            del self.task_counts[sorted(self.task_counts)[0]]

    def update_confidence(self, domain: str, value: float) -> None:
        self._set_bounded_score(self.confidence, domain, value, "confidence")

    @staticmethod
    def _set_bounded_score(target: dict[str, float], key: str, value: float, label: str) -> None:
        if not isinstance(key, str) or not key.strip() or len(key) > 120:
            raise ValueError(f"{label} key must be 1 to 120 characters")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number between 0 and 1")
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{label} must be a finite number between 0 and 1")
        target.pop(key, None)
        target[key] = score
        if len(target) > MAX_SELF_MODEL_ENTRIES:
            del target[next(iter(target))]

    @classmethod
    def _bounded_scores(cls, scores: dict[str, float], label: str) -> dict[str, float]:
        if len(scores) > MAX_SELF_MODEL_ENTRIES:
            raise ValueError(f"at most {MAX_SELF_MODEL_ENTRIES} {label} scores are allowed")
        result: dict[str, float] = {}
        for key, value in scores.items():
            cls._set_bounded_score(result, key, value, label)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "model_kind": "bounded_operational_self_model",
            "current_task": self.current_task,
            "known_capabilities": dict(self.known_capabilities),
            "blind_spots": sorted(self.blind_spots),
            "domains": dict(self.domains),
            "confidence": dict(self.confidence),
            "known_constraints": list(self.known_constraints),
            "recent_history": list(self.recent_history),
            "task_counts": dict(self.task_counts),
            "metrics": {
                "capability_count": len(self.known_capabilities),
                "domain_count": len(self.domains),
                "confidence_count": len(self.confidence),
                "blind_spot_count": len(self.blind_spots),
                "task_history_size": len(self.recent_history),
                "task_history_capacity": MAX_TASK_HISTORY,
                "entry_capacity": MAX_SELF_MODEL_ENTRIES,
                "constraint_count": len(self.known_constraints),
                "constraint_capacity": MAX_CONSTRAINTS,
                "blind_spot_capacity": MAX_BLIND_SPOTS,
                "subjective_experience_claimed": False,
            },
            "limitations": [
                "configured capability scores are not benchmark results",
                "confidence values are operational estimates, not calibrated probabilities",
                "does not represent subjective experience or consciousness",
            ],
        }


__all__ = ["SelfModel"]
