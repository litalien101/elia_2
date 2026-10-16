from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .reflection import ReflectiveAgent
from .self_model import SelfModel
from .state_controller import StateController
from .workspace import ActiveWorkspace


@dataclass
class ConsciousnessController:
    """Coordinates the active workspace, self-model, reflection, and mode control.

    This is intentionally bounded and additive: it does not overwrite the base
    reasoning engine or planner, but gives the system an explicit attention and
    metacognitive layer for foreground focus, state transitions, and reflection.
    """

    workspace: ActiveWorkspace = field(default_factory=ActiveWorkspace)
    self_model: SelfModel = field(default_factory=SelfModel)
    reflection: ReflectiveAgent = field(default_factory=ReflectiveAgent)
    state_controller: StateController = field(default_factory=StateController)
    consolidator: Any | None = None
    last_focus: dict[str, Any] = field(default_factory=dict)
    last_report: dict[str, Any] | None = None
    decision_count: int = 0
    cycle_count: int = 0
    attention_budget: int = 4
    layer_order: tuple[str, ...] = (
        "workspace",
        "self_model",
        "state_controller",
        "reflection",
        "consolidation",
    )

    def metacognitive_cycle(self, *, task: str | None = None) -> dict[str, Any]:
        self.cycle_count += 1
        focus = self.workspace.select_focus()
        snapshot = focus.get("priority_snapshot", {})
        recommendation = str((self.last_report or {}).get("recommendation", "continue_with_current_policy"))
        risk_flags = list((self.last_report or {}).get("risk_flags", []))
        contradiction_count = len((self.last_report or {}).get("contradictions", []))
        unresolved = len(self.workspace.unresolved_questions)
        evidence_load = len(self.workspace.active_evidence)
        hypothesis_load = len(self.workspace.active_hypotheses)
        confidence_values = list(self.self_model.confidence.values()) or [0.0]
        overall_confidence = sum(float(value) for value in confidence_values) / len(confidence_values)
        high_priority = [
            {"item": key, "priority": float(value)}
            for key, value in sorted(snapshot.items(), key=lambda item: (-float(item[1]), str(item[0])))
            if float(value) >= 0.75
        ][0:8]
        memory_pressure = min(
            1.0,
            (evidence_load + unresolved + hypothesis_load) / 12.0,
        )
        if self.consolidator is not None and self.consolidator.last_summary is not None:
            memory_pressure = min(1.0, memory_pressure + float(self.consolidator.last_summary.compression_ratio))
        if risk_flags or contradiction_count:
            loop_status = "review"
        elif unresolved > 0:
            loop_status = "collect"
        elif memory_pressure > 0.7:
            loop_status = "reframe"
        else:
            loop_status = "steady"
        revisit_queue = [
            item["item"]
            for item in high_priority
            if item["item"] not in (task or self.workspace.task)
        ]
        if not revisit_queue and (risk_flags or unresolved):
            revisit_queue = [task or self.workspace.task]
        return {
            "cycle": self.cycle_count,
            "task": task or self.workspace.task,
            "active_mode": self.state_controller.active_mode,
            "status": loop_status,
            "recommendation": recommendation,
            "overall_confidence": round(float(overall_confidence), 4),
            "memory_pressure": round(float(memory_pressure), 4),
            "risk_flags": risk_flags,
            "contradictions": contradiction_count,
            "revisit_queue": revisit_queue,
            "high_priority_items": high_priority,
            "decision_priority": (
                recommendation if recommendation != "continue_with_current_policy" else "continue_current_policy"
            ),
        }

    def pulse(self, *, task: str | None = None) -> dict[str, Any]:
        focus = self.workspace.select_focus()
        snapshot = focus.get("priority_snapshot", {})
        attention_stack = [
            {"item": key, "priority": float(value)}
            for key, value in sorted(snapshot.items(), key=lambda item: (-float(item[1]), str(item[0])))[: self.attention_budget]
        ]
        recommendation = str((self.last_report or {}).get("recommendation", "continue_with_current_policy"))
        recent_risk_count = len((self.last_report or {}).get("risk_flags", []))
        focus_confidence = max(
            (self.workspace.confidence.values() or [0.0]),
            default=0.0,
            key=float,
        )
        unresolved = len(self.workspace.unresolved_questions)
        evidence_load = len(self.workspace.active_evidence)
        health_score = 1.0
        health_score -= min(0.25, unresolved * 0.04)
        health_score -= min(0.2, evidence_load * 0.01)
        health_score -= min(0.2, recent_risk_count * 0.12)
        health_score -= max(0.0, (1.0 - float(focus_confidence)) * 0.2)
        health_score = max(0.0, min(1.0, health_score))
        if health_score >= 0.8:
            health_status = "stable"
        elif health_score >= 0.55:
            health_status = "watch"
        else:
            health_status = "critical"
        metacognitive_loop = self.metacognitive_cycle(task=task)
        return {
            "task": task or self.workspace.task,
            "mode": self.state_controller.active_mode,
            "attention_stack": attention_stack,
            "layer_order": list(self.layer_order),
            "active_layers": [
                layer for layer in self.layer_order if layer in {"workspace", "self_model", "state_controller", "reflection"}
            ],
            "recommendation": recommendation,
            "focus": focus,
            "metacognitive_load": round(min(1.0, (unresolved + evidence_load + recent_risk_count) / 12), 4),
            "attention_quality": round(float(focus_confidence), 4),
            "priority_snapshot": snapshot,
            "metacognitive_loop": metacognitive_loop,
            "functional_health": {
                "score": round(health_score, 4),
                "status": health_status,
                "risk_flags": recent_risk_count,
                "unresolved_questions": unresolved,
                "active_evidence": evidence_load,
            },
        }

    def update(
        self,
        *,
        task: str = "idle",
        evidence: list[str] | None = None,
        hypotheses: list[str] | None = None,
        questions: list[str] | None = None,
        priorities: dict[str, float] | None = None,
        confidence: dict[str, float] | None = None,
        decision: str | None = None,
        trace: str | None = None,
        uncertainty: dict[str, float] | None = None,
        conflict: dict[str, Any] | None = None,
        pending: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.workspace.update(
            task=task,
            evidence=evidence or [],
            hypotheses=hypotheses or [],
            questions=questions or [],
            priorities=priorities,
            confidence=confidence,
            decision=decision,
            trace=trace,
        )
        self.self_model.update_task(task)
        if isinstance(evidence, list):
            self.self_model.update_confidence("active_evidence", min(1.0, len(evidence) / 8))
        if confidence:
            for key, value in confidence.items():
                self.self_model.update_confidence(str(key), float(value))
        if uncertainty or conflict or pending:
            report = self.reflection.evaluate(
                evidence_count=len(evidence or []),
                hypothesis_count=len(hypotheses or []),
                conflict=conflict,
                pending=pending,
                uncertainty=uncertainty,
            )
            self.last_report = report.summary()
        mode = self.state_controller.decide_mode(
            evidence_count=len(evidence or []),
            hypothesis_count=len(hypotheses or []),
            unresolved_questions=len(questions or []),
            contradictions=1 if conflict and conflict.get("supports", 0) and conflict.get("challenges", 0) else 0,
            confidence_low=bool(confidence and min(confidence.values()) < 0.6),
            memory_pressure=0.0,
        )
        self.decision_count += 1
        self.last_focus = self.workspace.select_focus()
        return self.summary(mode=mode)

    def focus(self) -> dict[str, Any]:
        focus = self.workspace.select_focus()
        self.last_focus = focus
        return focus

    def summarize(self) -> dict[str, Any]:
        return self.summary()

    def summary(self, mode: str | None = None) -> dict[str, Any]:
        focus = self.workspace.select_focus()
        active_mode = mode or self.state_controller.active_mode
        last_report = self.last_report or {}
        reflection_recommendation = str(last_report.get("recommendation", "continue_with_current_policy"))
        planner_behavior_changed = reflection_recommendation not in {
            "continue_with_current_policy",
        }
        persistence_changed = False
        if self.consolidator is not None and self.consolidator.last_summary is not None:
            persistence_changed = float(self.consolidator.last_summary.compression_ratio) > 0.0
        pulse = self.pulse(task=self.workspace.task)
        base = {
            "controller_kind": "bounded_functional_consciousness",
            "active_mode": active_mode,
            "focus": focus,
            "workspace": self.workspace.summary(),
            "self_model": self.self_model.summary(),
            "state_controller": self.state_controller.summary(),
            "reflection": self.reflection.summary(),
            "decision_count": self.decision_count,
            "last_focus": self.last_focus or focus,
            "last_report": self.last_report,
            "planner_behavior_changed": planner_behavior_changed,
            "persistent_memory_changed_by_layers": persistence_changed,
            "global_workspace": pulse,
            "functional_health": pulse["functional_health"],
        }
        if self.consolidator is not None:
            base["consolidation"] = {
                **self.consolidator.status(),
                "last_summary": (
                    self.consolidator.last_summary.summary()
                    if self.consolidator.last_summary
                    else None
                ),
            }
        return base

    def status(self) -> dict[str, Any]:
        return self.summary()


__all__ = ["ConsciousnessController"]
