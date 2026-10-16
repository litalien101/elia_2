from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


MAX_MODE_HISTORY = 32


@dataclass
class StateController:
    """Selects a bounded operational mode without replacing the underlying planner."""

    active_mode: str = "idle"
    valid_modes: tuple[str, ...] = (
        "idle",
        "active_research",
        "evidence_scan",
        "reflective_review",
        "memory_consolidation",
        "low_power",
        "caution_mode",
        "task_completion",
    )
    mode_history: list[str] = field(default_factory=list)
    decision_count: int = 0
    mode_counts: dict[str, int] = field(default_factory=dict)

    def set_mode(self, mode: str) -> str:
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported mode: {mode}")
        self.active_mode = mode
        self.mode_history.append(mode)
        self.mode_history = self.mode_history[-MAX_MODE_HISTORY:]
        self.decision_count += 1
        self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1
        return mode

    def decide_mode(
        self,
        *,
        evidence_count: int = 0,
        hypothesis_count: int = 0,
        unresolved_questions: int = 0,
        contradictions: int = 0,
        confidence_low: bool = False,
        memory_pressure: float = 0.0,
    ) -> str:
        for name, value in (
            ("evidence_count", evidence_count),
            ("hypothesis_count", hypothesis_count),
            ("unresolved_questions", unresolved_questions),
            ("contradictions", contradictions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(memory_pressure, bool) or not isinstance(memory_pressure, (int, float)):
            raise ValueError("memory_pressure must be a finite number between 0 and 1")
        memory_pressure = float(memory_pressure)
        if not math.isfinite(memory_pressure) or not 0 <= memory_pressure <= 1:
            raise ValueError("memory_pressure must be a finite number between 0 and 1")
        if not isinstance(confidence_low, bool):
            raise ValueError("confidence_low must be a boolean")

        if contradictions > 0:
            return self.set_mode("reflective_review")
        if confidence_low or unresolved_questions > 0:
            return self.set_mode("evidence_scan")
        if evidence_count > 0 and hypothesis_count > 0:
            return self.set_mode("active_research")
        if memory_pressure > 0.75:
            return self.set_mode("memory_consolidation")
        if hypothesis_count == 0 and evidence_count == 0:
            return self.set_mode("idle")
        if confidence_low and memory_pressure > 0.5:
            return self.set_mode("low_power")
        return self.set_mode("task_completion")

    def summary(self) -> dict[str, Any]:
        return {
            "active_mode": self.active_mode,
            "valid_modes": list(self.valid_modes),
            "mode_history": list(self.mode_history),
            "metrics": {
                "decisions": self.decision_count,
                "mode_counts": dict(self.mode_counts),
                "history_size": len(self.mode_history),
                "history_capacity": MAX_MODE_HISTORY,
            },
        }


__all__ = ["StateController"]
