from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


MAX_REFLECTION_ENTRIES = 64
MAX_REFLECTION_HISTORY = 32


@dataclass
class ReflectionReport:
    contradictions: list[str] = field(default_factory=list)
    uncertain_claims: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    recommendation: str = "continue_with_current_policy"
    confidence_delta: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "contradictions": list(self.contradictions),
            "uncertain_claims": list(self.uncertain_claims),
            "missing_evidence": list(self.missing_evidence),
            "risk_flags": list(self.risk_flags),
            "recommendation": self.recommendation,
            "confidence_delta": self.confidence_delta,
        }


class ReflectiveAgent:
    """Checks claim quality, contradiction, and evidence sufficiency without changing behavior directly."""

    def __init__(self) -> None:
        self.last_report: ReflectionReport | None = None
        self.evaluation_count = 0
        self.risk_counts: dict[str, int] = {}
        self.recent_recommendations: list[str] = []

    def evaluate(
        self,
        *,
        evidence_count: int = 0,
        hypothesis_count: int = 0,
        conflict: dict[str, Any] | None = None,
        pending: dict[str, Any] | None = None,
        uncertainty: dict[str, float] | None = None,
    ) -> ReflectionReport:
        for name, value in (
            ("evidence_count", evidence_count),
            ("hypothesis_count", hypothesis_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if conflict is not None and not isinstance(conflict, dict):
            raise ValueError("conflict must be an object when supplied")
        if pending is not None and not isinstance(pending, dict):
            raise ValueError("pending must be an object when supplied")
        if uncertainty is not None:
            if len(uncertainty) > MAX_REFLECTION_ENTRIES:
                raise ValueError(f"uncertainty supports at most {MAX_REFLECTION_ENTRIES} entries")
            for key, value in uncertainty.items():
                if not isinstance(key, str) or not key.strip() or len(key) > 120:
                    raise ValueError("uncertainty keys must be 1 to 120 characters")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("uncertainty values must be finite numbers between 0 and 1")
                if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                    raise ValueError("uncertainty values must be finite numbers between 0 and 1")
        contradiction = []
        uncertain_claims = []
        missing_evidence = []
        risk_flags = []
        recommendation = "continue_with_current_policy"
        confidence_delta = 0.0

        if conflict is not None:
            supports = int(conflict.get("supports", 0) or 0)
            challenges = int(conflict.get("challenges", 0) or 0)
            if supports and challenges:
                contradiction.append(f"conflicting_evidence:{conflict.get('id', 'unknown')}")
                risk_flags.append("conflicting_evidence")
                recommendation = "review_conflicting_evidence"
                confidence_delta -= 0.15

        if pending is not None:
            missing_evidence.append(f"pending_prediction:{pending.get('id', 'unknown')}")
            risk_flags.append("pending_verification")
            if recommendation == "continue_with_current_policy":
                recommendation = "verify_pending_prediction"

        if evidence_count == 0 and hypothesis_count > 0:
            missing_evidence.append("no_grounded_observations")
            risk_flags.append("under_supported_hypothesis")
            recommendation = "request_consent_based_evidence"
            confidence_delta -= 0.1

        if uncertainty:
            for key, value in sorted(uncertainty.items(), key=lambda item: float(item[1])):
                if float(value) > 0.7:
                    uncertain_claims.append(str(key))
                    risk_flags.append("high_uncertainty")
                    confidence_delta -= 0.05

        if hypothesis_count >= 4 and evidence_count <= 2:
            risk_flags.append("evidence_too_sparse")
            recommendation = "gather_more_supporting_evidence"
            confidence_delta -= 0.1

        report = ReflectionReport(
            contradictions=contradiction,
            uncertain_claims=uncertain_claims,
            missing_evidence=missing_evidence,
            risk_flags=risk_flags,
            recommendation=recommendation,
            confidence_delta=confidence_delta,
        )
        self.last_report = report
        self.evaluation_count += 1
        for risk in report.risk_flags:
            self.risk_counts[risk] = self.risk_counts.get(risk, 0) + 1
        self.recent_recommendations.append(recommendation)
        self.recent_recommendations = self.recent_recommendations[-MAX_REFLECTION_HISTORY:]
        return report

    def summary(self) -> dict[str, Any]:
        return {
            "last_report": self.last_report.summary() if self.last_report else None,
            "metrics": {
                "evaluations": self.evaluation_count,
                "risk_counts": dict(self.risk_counts),
                "history_size": len(self.recent_recommendations),
                "history_capacity": MAX_REFLECTION_HISTORY,
            },
            "recent_recommendations": list(self.recent_recommendations),
            "effects": "reports_only_no_planner_or_memory_mutation",
        }


__all__ = ["ReflectiveAgent", "ReflectionReport"]
