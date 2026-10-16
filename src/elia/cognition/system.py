from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from .circuit import SpecialistCircuit
from .consolidation import BackgroundConsolidator
from .consciousness import ConsciousnessController
from .core import SharedCognitiveCore, detect_hardware_profile, SparseLogisticReadout
from .reflection import ReflectiveAgent
from .replay import BalancedReplayBuffer, ReplayExample
from .router import SpecialistRouter
from .self_model import SelfModel
from .state_controller import StateController
from .workspace import ActiveWorkspace


MAX_SPECIALISTS = 64
MAX_REPLAY_STEPS_PER_UPDATE = 128
MAX_PREDICTION_CACHE_ENTRIES = 2_048


@dataclass(frozen=True)
class StoredExperience:
    example_id: str
    specialist_id: str
    text: str
    label: int
    sequence: int


class CognitiveSystem:
    """Coordinates the shared core, sparse substrate, circuits, router and replay."""

    def __init__(
        self,
        unit_count: int = 10_000,
        fanout: int = 16,
        replay_capacity_per_specialist: int = 128,
        replay_per_specialist: int = 16,
        replay_enabled: bool = True,
        specialists_enabled: bool = True,
        hardware_profile: dict[str, object] | None = None,
    ) -> None:
        if not 1 <= replay_per_specialist <= replay_capacity_per_specialist:
            raise ValueError("replay_per_specialist must be within the replay capacity")
        self.hardware_profile = dict(detect_hardware_profile() if hardware_profile is None else hardware_profile)
        self.core = SharedCognitiveCore(
            unit_count=unit_count,
            max_fanout=fanout,
            hardware_profile=self.hardware_profile,
        )
        self.router = SpecialistRouter()
        self.replay = BalancedReplayBuffer(replay_capacity_per_specialist)
        self.circuits: dict[str, SpecialistCircuit] = {}
        self.workspace = ActiveWorkspace()
        self.self_model = SelfModel()
        self.self_model.set_constraints([
            "local_cpu_only",
            "bounded_reasoning",
            "provenance_preserved",
            "no_unverified_memory_mutation",
        ])
        self.reflection = ReflectiveAgent()
        self.state_controller = StateController()
        self.consolidator = BackgroundConsolidator()
        self.consciousness = ConsciousnessController(
            workspace=self.workspace,
            self_model=self.self_model,
            reflection=self.reflection,
            state_controller=self.state_controller,
            consolidator=self.consolidator,
        )
        self.replay_per_specialist = replay_per_specialist
        self.replay_enabled = replay_enabled
        self.specialists_enabled = specialists_enabled
        self._sequence = 0
        self._lock = threading.RLock()
        self._prediction_cache_limit = max(512, min(4096, self.hardware_profile["cpu_count"] * 256))
        self._prediction_cache: OrderedDict[tuple[str, str | None], dict[str, object]] = OrderedDict()

    def _combined_probability(self, features: dict[int, float], specialist_id: str) -> float:
        circuit = self.circuits.get(specialist_id) if self.specialists_enabled else None
        if circuit and circuit.examples_seen:
            return circuit.probability(features)
        return self.core.readout.probability(features)

    def _update_functional_state(
        self,
        *,
        task: str,
        evidence: list[str] | None = None,
        hypotheses: list[str] | None = None,
        questions: list[str] | None = None,
        priorities: dict[str, float] | None = None,
        confidence: dict[str, float] | None = None,
        decision: str | None = None,
        trace: str | None = None,
        uncertainty: dict[str, float] | None = None,
        conflict: dict[str, object] | None = None,
        pending: dict[str, object] | None = None,
    ) -> None:
        self.consciousness.update(
            task=task,
            evidence=evidence,
            hypotheses=hypotheses,
            questions=questions,
            priorities=priorities,
            confidence=confidence,
            decision=decision,
            trace=trace,
            uncertainty=uncertainty,
            conflict=conflict,
            pending=pending,
        )

    def update_functional_state(
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
        conflict: dict[str, object] | None = None,
        pending: dict[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            self.consciousness.update(
                task=task,
                evidence=evidence,
                hypotheses=hypotheses,
                questions=questions,
                priorities=priorities,
                confidence=confidence,
                decision=decision,
                trace=trace,
                uncertainty=uncertainty,
                conflict=conflict,
                pending=pending,
            )
            return self.functional_status()

    def predict(self, text: str, specialist_id: str | None = None) -> dict[str, object]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > 16_000:
            raise ValueError("text exceeds 16000 characters")
        cache_key = (text, specialist_id)
        with self._lock:
            cached = self._prediction_cache.get(cache_key)
            if cached is not None:
                self._prediction_cache.move_to_end(cache_key)
                return dict(cached)
            direct_features = self.core.combined_features(text)
            features = self.core.substrate.activate(direct_features)
            routing = self.router.route(direct_features)
            selected = specialist_id or routing["specialist_id"]
            if specialist_id is not None:
                if specialist_id not in self.circuits:
                    result = {
                        "decision": "abstain",
                        "reason": "specialist_unknown",
                        "specialist_id": None,
                        "routing": routing,
                    }
                    self._update_functional_state(
                        task="prediction",
                        evidence=[text],
                        hypotheses=[str(specialist_id)],
                        priorities={text: 0.6},
                        confidence={text: 0.0},
                        decision="abstain_specialist_unknown",
                        trace=f"prediction_abstain:{specialist_id}",
                        questions=[f"unknown_specialist:{specialist_id}"],
                        uncertainty={str(specialist_id): 1.0},
                    )
                    self._prediction_cache[cache_key] = result
                    self._prediction_cache.move_to_end(cache_key)
                    if len(self._prediction_cache) > self._prediction_cache_limit:
                        self._prediction_cache.popitem(last=False)
                    return dict(result)
                mode = "explicit"
            else:
                mode = str(routing["mode"])
            if selected is None:
                result = {
                    "decision": "abstain",
                    "reason": "novel_or_ambiguous_route",
                    "specialist_id": None,
                    "routing": routing,
                    "active_units": len(features),
                }
                self._update_functional_state(
                    task="prediction",
                    evidence=[text],
                    hypotheses=[],
                    priorities={text: 0.5},
                    confidence={text: 0.0},
                    decision="abstain_route",
                    trace="prediction_abstain:ambiguous_route",
                    questions=["ambiguous_route"],
                    uncertainty={"route_choice": 1.0},
                )
                self._prediction_cache[cache_key] = result
                self._prediction_cache.move_to_end(cache_key)
                if len(self._prediction_cache) > self._prediction_cache_limit:
                    self._prediction_cache.popitem(last=False)
                return dict(result)
            probability = self._combined_probability(features, str(selected))
            result = {
                "decision": "predict",
                "specialist_id": selected,
                "routing_mode": mode,
                "probability": round(probability, 6),
                "predicted_label": int(probability >= 0.5),
                "routing": routing,
                "active_units": len(features),
            }
            self._update_functional_state(
                task="prediction",
                evidence=[text],
                hypotheses=[str(selected)],
                priorities={text: round(float(probability), 6)},
                confidence={text: round(float(probability), 6), "prediction": float(probability)},
                decision=f"predict:{selected}",
                trace=f"prediction_{selected}:{round(float(probability), 6)}",
                uncertainty={"prediction_confidence": 1.0 - float(probability)},
            )
            self._prediction_cache[cache_key] = result
            self._prediction_cache.move_to_end(cache_key)
            if len(self._prediction_cache) > self._prediction_cache_limit:
                self._prediction_cache.popitem(last=False)
            return dict(result)

    def observe(
        self,
        example_id: str,
        specialist_id: str,
        text: str,
        label: int,
        sequence: int | None = None,
    ) -> dict[str, object]:
        if not example_id or len(example_id) > 128:
            raise ValueError("example_id must be between 1 and 128 characters")
        if not specialist_id or len(specialist_id) > 80:
            raise ValueError("specialist_id must be between 1 and 80 characters")
        if not isinstance(text, str) or not text.strip() or len(text) > 16_000:
            raise ValueError("text must contain 1 to 16000 characters")
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("label must be the integer 0 or 1")

        with self._lock:
            self._prediction_cache.clear()
            if specialist_id not in self.circuits and len(self.circuits) >= MAX_SPECIALISTS:
                raise ValueError(f"At most {MAX_SPECIALISTS} specialist circuits are supported")
            if self.replay.contains(example_id):
                raise ValueError("example_id has already been learned")
            self.circuits.setdefault(specialist_id, SpecialistCircuit(specialist_id))
            self._sequence = max(self._sequence + 1, sequence or 0)
            current_sequence = sequence if sequence is not None else self._sequence

            direct_features = self.core.combined_features(text)
            features = self.core.substrate.activate(direct_features)
            before = self._combined_probability(features, specialist_id)
            surprise = abs(label - before)
            example = ReplayExample(
                example_id=example_id,
                specialist_id=specialist_id,
                text=text,
                label=label,
                surprise=surprise,
                sequence=current_sequence,
            )
            self.replay.add(example)
            self.router.learn(specialist_id, direct_features)
            self.core.learn_associations(features, surprise)

            if self.replay_enabled:
                batch = self.replay.balanced_batch(
                    min(self.replay_per_specialist, MAX_REPLAY_STEPS_PER_UPDATE),
                    include_id=example_id,
                    max_total=MAX_REPLAY_STEPS_PER_UPDATE,
                )
                if all(item.example_id != example_id for item in batch):
                    batch = batch[:MAX_REPLAY_STEPS_PER_UPDATE - 1] + [example]
            else:
                batch = [example]
            replay_updates = 0
            for replay_item in batch:
                replay_features = self.core.encode(replay_item.text)
                self.core.readout.update(replay_features, replay_item.label, learning_rate_scale=0.65)
                if self.specialists_enabled:
                    self.circuits[replay_item.specialist_id].learn(replay_features, replay_item.label)
                replay_updates += 1

            after = self._combined_probability(features, specialist_id)
            self._update_functional_state(
                task="learning",
                evidence=[text],
                hypotheses=[specialist_id],
                priorities={text: round(float(after), 6)},
                confidence={text: round(float(after), 6), specialist_id: float(after)},
                decision=f"observe:{specialist_id}",
                trace=f"observe_{specialist_id}:{round(float(after), 6)}",
                uncertainty={specialist_id: float(abs(label - before))},
            )
            return {
                "example_id": example_id,
                "specialist_id": specialist_id,
                "pre_update_probability": round(before, 6),
                "post_update_probability": round(after, 6),
                "surprise": round(surprise, 6),
                "replay_updates": replay_updates,
                "retained_in_replay": self.replay.contains(example_id),
                "active_units": len(features),
                "synapses": self.core.substrate.synapse_count,
            }

    def rebuild(self, experiences: Iterable[StoredExperience]) -> None:
        """Relearn from retained source-linked records, e.g. after a source deletion."""
        with self._lock:
            self._prediction_cache.clear()
            self.core.reset()
            self.router.clear()
            self.replay.clear()
            self.circuits.clear()
            self._sequence = 0
            for experience in experiences:
                self.observe(
                    experience.example_id,
                    experience.specialist_id,
                    experience.text,
                    experience.label,
                    experience.sequence,
                )

    def functional_controller(self) -> dict[str, object]:
        with self._lock:
            return self.consciousness.summary()

    def functional_status(self) -> dict[str, object]:
        with self._lock:
            consciousness = self.consciousness.summary()
            return {
                "workspace": self.workspace.summary(),
                "self_model": self.self_model.summary(),
                "state_controller": self.state_controller.summary(),
                "reflection": self.reflection.summary(),
                "consciousness": consciousness,
                "consciousness_controller": consciousness,
                "global_workspace": consciousness["global_workspace"],
                "active_mode": consciousness["active_mode"],
                "focus": consciousness["focus"],
                "attention_stack": consciousness["global_workspace"]["attention_stack"],
                "functional_health": consciousness["functional_health"],
                "consolidation": {
                    **self.consolidator.status(),
                    "last_summary": (
                        self.consolidator.last_summary.summary()
                        if self.consolidator.last_summary
                        else None
                    ),
                },
                "planner_behavior_changed": consciousness["planner_behavior_changed"],
                "persistent_memory_changed_by_layers": consciousness["persistent_memory_changed_by_layers"],
                "metacognitive_loop": consciousness["global_workspace"]["metacognitive_loop"],
            }

    def functional_focus(self) -> dict[str, object]:
        with self._lock:
            return self.consciousness.focus()

    def enqueue_consolidation(self) -> dict[str, object]:
        with self._lock:
            return self.consolidator.enqueue(
                evidence=self.workspace.active_evidence,
                hypotheses=self.workspace.active_hypotheses,
            )

    def run_consolidation_once(self) -> dict[str, object]:
        with self._lock:
            previous_mode = self.state_controller.active_mode
            self.state_controller.set_mode("memory_consolidation")
            try:
                report = self.consolidator.run_one()
                self.self_model.update_task("memory_consolidation")
                return report
            finally:
                self.state_controller.set_mode(previous_mode)

    def status(self) -> dict[str, object]:
        with self._lock:
            functional = self.functional_status()
            return {
                "architecture": {
                    "shared_core": "sparse_compositional_language_features_and_online_readout",
                    "substrate": "bounded_plastic_associative_units",
                    "specialists": "per_domain_sparse_logistic_circuits",
                    "router": "prototype_similarity_and_bounded_exemplar_abstention",
                    "replay": "surprise_prioritized_balanced_per_specialist",
                },
                "routing_thresholds": {
                    "minimum_similarity": self.router.minimum_similarity,
                    "minimum_margin": self.router.minimum_margin,
                    "minimum_example_similarity": self.router.minimum_example_similarity,
                    "exemplar_capacity_per_specialist": self.router.exemplar_capacity,
                },
                "neurons": self.core.substrate.unit_count,
                "max_fanout": self.core.substrate.max_fanout,
                "synapses": self.core.substrate.synapse_count,
                "synapse_capacity": self.core.substrate.synapse_capacity,
                "specialist_count": len(self.circuits),
                "specialists": [
                    {"id": key, "updates": circuit.examples_seen,
                     "parameter_count": len(circuit.readout.weights)}
                    for key, circuit in sorted(self.circuits.items())
                ],
                "shared_parameter_count": len(self.core.readout.weights),
                "replay_size": self.replay.size,
                "replay_capacity_per_specialist": self.replay.capacity_per_specialist,
                "replay_per_specialist": self.replay_per_specialist,
                "replay_enabled": self.replay_enabled,
                "specialists_enabled": self.specialists_enabled,
                "replay_counts": self.replay.specialist_counts,
                "hardware_profile": self.hardware_profile,
                "hardware_optimized": True,
                **functional,
                "functional_layers": functional,
                "cpu_only": True,
                "external_model_required": False,
            }
