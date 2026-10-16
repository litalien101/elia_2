from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    action: str
    target_type: str
    target_id: str | None
    title: str
    rationale: str
    proposed_step: str
    priority: int


class TaskPlanner:
    """Choose the next bounded action from a small evidence-driven policy."""

    def __init__(self, min_validation_cases: int = 10) -> None:
        self.min_validation_cases = min_validation_cases

    @staticmethod
    def _risk_band(priority: int) -> str:
        if priority >= 80:
            return "high"
        if priority >= 60:
            return "moderate"
        if priority >= 30:
            return "low"
        return "minimal"

    def _decision_score(
        self,
        *,
        evidence_count: int,
        hypothesis_count: int,
        conflict: dict[str, Any] | None,
        pending: dict[str, Any] | None,
        unresolved_candidate: dict[str, Any] | None,
        base_priority: int,
    ) -> int:
        score = base_priority
        score += min(12, evidence_count * 2)
        score += min(10, hypothesis_count * 3)
        if conflict is not None:
            score += min(20, int(conflict.get("supports", 0) or 0) * 4)
            score += min(20, int(conflict.get("challenges", 0) or 0) * 5)
        if pending is not None:
            score += min(15, max(0, int(pending.get("age_days", 0) or 0)) * 4)
        if unresolved_candidate is not None:
            score += min(15, int(unresolved_candidate.get("resolved_predictions", 0) or 0) * 3)
        return min(score, 100)

    def plan(
        self,
        *,
        evidence_count: int = 0,
        hypothesis_count: int = 0,
        conflict: dict[str, Any] | None = None,
        pending: dict[str, Any] | None = None,
        unresolved_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolution_mode = "single_step_priority"
        if conflict is not None:
            support_count = int(conflict.get("supports", 0) or 0)
            challenge_count = int(conflict.get("challenges", 0) or 0)
            priority = max(91, 55 + support_count + challenge_count * 2)
            recommendation = PlanStep(
                action="review_conflicting_evidence",
                target_type="hypothesis",
                target_id=str(conflict.get("id") or ""),
                title="Review evidence that disagrees",
                rationale=(
                    f"This claim has {support_count} supporting and {challenge_count} challenging "
                    "observations, so the current evidence is unstable."
                ),
                proposed_step=(
                    "Compare the linked source observations and decide whether the claim needs "
                    "narrower conditions, new tests, or a stronger evidence threshold."
                ),
                priority=priority,
            )
            plan_steps = [
                {
                    "action": "collect_conflict_context",
                    "title": "List the supporting and challenging observations",
                    "rationale": "Separate the provenance points before changing the claim.",
                    "priority": 95,
                },
                {
                    "action": "resolve_evidence_disagreement",
                    "title": "Decide whether the claim should narrow, be rejected, or be tested more",
                    "rationale": "Conflicting evidence must be adjudicated before any recommendation is trusted.",
                    "priority": 90,
                },
                {
                    "action": "schedule_new_verification",
                    "title": "Design a follow-up test or decision rule",
                    "rationale": "Only after disagreement is handled should the next test be allowed.",
                    "priority": 75,
                },
            ]
            if pending is not None:
                plan_steps.extend([
                    {
                        "action": "inspect_pending_prediction",
                        "title": "Review the unresolved prediction that still lacks a final outcome",
                        "rationale": "A pending result can amplify the disagreement unless it is resolved explicitly.",
                        "priority": 84,
                    },
                    {
                        "action": "verify_or_mark_unknown",
                        "title": "Attach a verified outcome or a justified unknown to the pending prediction",
                        "rationale": "Do not conflate unresolved status with evidence for either side.",
                        "priority": 80,
                    },
                    {
                        "action": "reassess_conflict_after_verification",
                        "title": "Re-run the evidence comparison after the pending case is resolved",
                        "rationale": "The final claim should be judged against the complete, verified record.",
                        "priority": 78,
                    },
                ])
            resolution_mode = "multi_step_evidence_resolution"
        elif pending is not None:
            resolution_mode = "pending_verification_sequence"
            priority = 80 + max(0, int(pending.get("age_days", 0) or 0))
            recommendation = PlanStep(
                action="verify_pending_prediction",
                target_type="prediction",
                target_id=str(pending.get("id") or ""),
                title="Check a prediction that has no recorded outcome",
                rationale=(
                    f"The prediction concerns “{pending.get('context', 'this case')}” and has not "
                    "been marked as known or unknown."
                ),
                proposed_step=(
                    "Review the prediction and record a verified yes/no outcome, or mark it "
                    "unknown. ELIA will not infer an outcome from silence."
                ),
                priority=priority,
            )
            plan_steps = [
                {
                    "action": "inspect_pending_prediction",
                    "title": "Open the pending prediction and its context",
                    "rationale": "A missing outcome cannot be treated as evidence."
                },
                {
                    "action": "verify_or_mark_unknown",
                    "title": "Attach a verified outcome or a justified unknown",
                    "rationale": "The system must record a real status, not silence.",
                },
            ]
        elif unresolved_candidate is not None and unresolved_candidate.get("status") == "candidate":
            resolution_mode = "candidate_validation_sequence"
            resolved_predictions = int(unresolved_candidate.get("resolved_predictions", 0) or 0)
            priority = min(95, 70 + max(0, resolved_predictions - 1) * 5)
            recommendation = PlanStep(
                action="design_prospective_test",
                target_type="hypothesis",
                target_id=str(unresolved_candidate.get("id") or ""),
                title="Design a fresh test for a candidate TRACE",
                rationale=(
                    f"The candidate has {resolved_predictions} resolved prediction(s); promotion "
                    f"requires {self.min_validation_cases} and better performance than its captured "
                    "baseline."
                ),
                proposed_step=(
                    "Choose a genuinely new, observable situation, register the predicted "
                    "probability before its outcome, and later attach a verification source."
                ),
                priority=priority,
            )
            plan_steps = [
                {
                    "action": "define_test_condition",
                    "title": "State the new scenario and expected outcome",
                    "rationale": "Candidate promotion requires a genuinely new observable case.",
                },
                {
                    "action": "register_prediction_before_outcome",
                    "title": "Record a probability before verification",
                    "rationale": "This preserves the test's prospective calibration.",
                },
                {
                    "action": "evaluate_and_promote",
                    "title": "Compare against baseline and then decide promotion",
                    "rationale": "Promotion requires enough resolved predictions and better Brier performance.",
                },
            ]
        elif unresolved_candidate is not None:
            resolution_mode = "counterexample_search_sequence"
            priority = 60 + int(bool(unresolved_candidate.get("claim"))) * 5
            recommendation = PlanStep(
                action="seek_counterexample",
                target_type="hypothesis",
                target_id=str(unresolved_candidate.get("id") or ""),
                title="Try to disprove a validated TRACE",
                rationale=(
                    f"“{unresolved_candidate.get('claim', 'This claim')}” is currently validated "
                    "under the prototype's evidence threshold; it may still be wrong outside "
                    "tested cases."
                ),
                proposed_step=(
                    "Find a new case where the claim could fail, record it prospectively, and "
                    "verify the result before treating the claim as reliable."
                ),
                priority=priority,
            )
            plan_steps = [
                {
                    "action": "search_for_counterexample",
                    "title": "Find a plausible case that could falsify the claim",
                    "rationale": "Validated claims must still be stress-tested under novel conditions.",
                },
                {
                    "action": "register_prospective_test",
                    "title": "Pre-register the prediction in the new case",
                    "rationale": "This keeps the next step evidence-based and bounded.",
                },
                {
                    "action": "verify_outcome",
                    "title": "Compare the real outcome against the prediction",
                    "rationale": "Only after verification should the claim be considered more or less reliable.",
                },
            ]
        elif evidence_count > 0:
            resolution_mode = "evidence_to_hypothesis_sequence"
            priority = 40 + min(10, evidence_count)
            recommendation = PlanStep(
                action="form_testable_hypothesis",
                target_type="evidence",
                target_id=None,
                title="Turn an observation into a testable question",
                rationale=(
                    f"There are {evidence_count} source-linked observation(s), but no active "
                    "hypothesis to evaluate."
                ),
                proposed_step=(
                    "A person should select an observation and write a narrow claim with a "
                    "measurable expected outcome; ELIA will not invent it as fact."
                ),
                priority=priority,
            )
            plan_steps = [
                {
                    "action": "select_observation",
                    "title": "Choose the most relevant observation or episode",
                    "rationale": "Generate a testable question only from a grounded observation.",
                },
                {
                    "action": "write_hypothesis",
                    "title": "Turn it into a narrow claim with expected outcome",
                    "rationale": "The claim must be explicit and measurable.",
                },
                {
                    "action": "register_verification_plan",
                    "title": "Define how it will be checked later",
                    "rationale": "Evidence-driven testing needs an explicit outcome path.",
                },
            ]
        else:
            resolution_mode = "consent_request_sequence"
            recommendation = PlanStep(
                action="request_consent_based_evidence",
                target_type="none",
                target_id=None,
                title="Gather an explicitly shared source before reasoning further",
                rationale="There are no source-linked observations or hypotheses to assess.",
                proposed_step=(
                    "Add an experience only if the user explicitly chooses to store it; "
                    "otherwise there is not enough evidence to make a grounded recommendation."
                ),
                priority=10,
            )
            plan_steps = [
                {
                    "action": "collect_consent",
                    "title": "Request explicit consent before storing an experience",
                    "rationale": "Local evidence must be consented and traceable.",
                }
            ]

        decision_score = self._decision_score(
            evidence_count=evidence_count,
            hypothesis_count=hypothesis_count,
            conflict=conflict,
            pending=pending,
            unresolved_candidate=unresolved_candidate,
            base_priority=recommendation.priority,
        )

        return {
            "action": recommendation.action,
            "target_type": recommendation.target_type,
            "target_id": recommendation.target_id,
            "title": recommendation.title,
            "rationale": recommendation.rationale,
            "proposed_step": recommendation.proposed_step,
            "priority": recommendation.priority,
            "priority_score": recommendation.priority,
            "decision_score": decision_score,
            "risk_level": self._risk_band(recommendation.priority),
            "resolution_mode": resolution_mode,
            "resolution_path": plan_steps,
            "plan_steps": plan_steps,
            "basis": {
                "observations": evidence_count,
                "hypotheses": hypothesis_count,
                "decision_policy": "bounded_evidence_priority_rules_v3",
            },
            "requires_human_approval": True,
            "executed": False,
            "limitations": (
                "This is a transparent rule-based next-step selection, not independent "
                "conscious thought or general-purpose reasoning."
            ),
        }


__all__ = ["PlanStep", "TaskPlanner"]
