from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MAX_ACTIVE_ITEMS = 64
MAX_PRIORITY_ITEMS = 64
MAX_DECISION_HISTORY = 16
MAX_TRACE_HISTORY = 32


@dataclass
class ActiveWorkspace:
    """Bounded transient foreground context for the functional processing layers."""

    task: str = "idle"
    active_evidence: list[str] = field(default_factory=list)
    active_hypotheses: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    attention_priorities: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    recent_decisions: list[str] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    update_count: int = 0
    eviction_count: int = 0

    def update(
        self,
        *,
        task: str | None = None,
        evidence: list[str] | None = None,
        hypotheses: list[str] | None = None,
        questions: list[str] | None = None,
        priorities: dict[str, float] | None = None,
        confidence: dict[str, float] | None = None,
        decision: str | None = None,
        trace: str | None = None,
    ) -> dict[str, Any]:
        self.update_count += 1
        if task is not None:
            self.task = task
        if evidence is not None:
            self.active_evidence = self._bounded_unique(evidence)
        if hypotheses is not None:
            self.active_hypotheses = self._bounded_unique(hypotheses)
        if questions is not None:
            self.unresolved_questions = self._bounded_unique(questions)
        if priorities is not None:
            self._merge_bounded(self.attention_priorities, priorities)
        if confidence is not None:
            self._merge_bounded(self.confidence, confidence)
        if decision is not None:
            self.recent_decisions.append(decision)
            self._trim(self.recent_decisions, MAX_DECISION_HISTORY)
        if trace is not None:
            self.reasoning_trace.append(trace)
            self._trim(self.reasoning_trace, MAX_TRACE_HISTORY)
        return self.summary()

    def _bounded_unique(self, values: list[str]) -> list[str]:
        if not isinstance(values, list) or len(values) > MAX_ACTIVE_ITEMS:
            raise ValueError(f"workspace fields allow at most {MAX_ACTIVE_ITEMS} items")
        unique = list(dict.fromkeys(values))
        return unique

    def _merge_bounded(self, target: dict[str, float], values: dict[str, float]) -> None:
        if not isinstance(values, dict) or len(values) > MAX_PRIORITY_ITEMS:
            raise ValueError(f"workspace fields allow at most {MAX_PRIORITY_ITEMS} items")
        for key, value in values.items():
            target.pop(key, None)
            target[key] = value
        overflow = max(0, len(target) - MAX_PRIORITY_ITEMS)
        if overflow:
            self.eviction_count += overflow
            for key in list(target)[:overflow]:
                del target[key]

    @staticmethod
    def _trim(values: list[str], capacity: int) -> None:
        if len(values) > capacity:
            del values[:-capacity]

    def priority_snapshot(self) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        for item in self.active_evidence:
            snapshot[item] = max(snapshot.get(item, 0.0), 0.65)
        for item in self.active_hypotheses:
            snapshot[item] = max(snapshot.get(item, 0.0), 0.8)
        for item in self.unresolved_questions:
            snapshot[item] = max(snapshot.get(item, 0.0), 0.9)
        if self.task:
            snapshot[self.task] = max(snapshot.get(self.task, 0.0), 1.0)
        for key, value in self.attention_priorities.items():
            snapshot[key] = float(value)
        if not snapshot:
            return {self.task: 1.0} if self.task else {}
        return dict(sorted(snapshot.items(), key=lambda item: (-float(item[1]), str(item[0]))))

    def select_focus(self) -> dict[str, Any]:
        priority_snapshot = self.priority_snapshot()
        focus = next(iter(priority_snapshot), self.task)
        focus_candidates = [
            {"item": key, "priority": float(value)}
            for key, value in priority_snapshot.items()
        ]
        return {
            "focus": focus,
            "task": self.task,
            "active_evidence_count": len(self.active_evidence),
            "active_hypothesis_count": len(self.active_hypotheses),
            "unresolved_questions": list(self.unresolved_questions),
            "attention_priorities": dict(self.attention_priorities),
            "priority_snapshot": priority_snapshot,
            "focus_candidates": focus_candidates,
        }

    def summary(self) -> dict[str, Any]:
        focus = self.select_focus()
        return {
            "task": self.task,
            "active_evidence": list(self.active_evidence),
            "active_hypotheses": list(self.active_hypotheses),
            "unresolved_questions": list(self.unresolved_questions),
            "attention_priorities": dict(self.attention_priorities),
            "priority_snapshot": focus["priority_snapshot"],
            "focus_candidates": focus["focus_candidates"],
            "confidence": dict(self.confidence),
            "recent_decisions": list(self.recent_decisions),
            "reasoning_trace": list(self.reasoning_trace),
            "metrics": {
                "updates": self.update_count,
                "evictions": self.eviction_count,
                "active_item_capacity": MAX_ACTIVE_ITEMS,
                "priority_capacity": MAX_PRIORITY_ITEMS,
                "decision_history_capacity": MAX_DECISION_HISTORY,
                "trace_history_capacity": MAX_TRACE_HISTORY,
            },
        }


__all__ = ["ActiveWorkspace"]
