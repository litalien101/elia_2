import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elia.cognition import CognitiveSystem, ConsciousnessController, TaskPlanner
from elia.cognition.benchmark import (
    run_benchmark,
    run_generalization_benchmark,
    run_sweep,
    run_symbolic_composition_benchmark,
    run_unseen_template_benchmark,
    run_unseen_template_sweep,
)
from elia.cognition.core import SharedCognitiveCore, stable_unit
from elia.cognition.consolidation import BackgroundConsolidator
from elia.cognition.plasticity import CompetitivePlasticSubstrate
from elia.cognition.plasticity_benchmark import run_plasticity_benchmark
from elia.cognition.reflection import ReflectiveAgent
from elia.cognition.replay import BalancedReplayBuffer, ReplayExample
from elia.cognition.router import SpecialistRouter
from elia.cognition.self_model import SelfModel
from elia.cognition.state_controller import StateController
from elia.cognition.substrate import SparseNeuronSubstrate
from elia.cognition.system import CognitiveSystem
from elia.cognition.workspace import ActiveWorkspace


class SparseSubstrateTests(unittest.TestCase):
    def test_associations_activate_sparse_neighbours_with_bounded_fanout(self):
        substrate = SparseNeuronSubstrate(unit_count=128, max_fanout=2)
        substrate.learn_associations([3, 9, 15], surprise=1.0)
        self.assertLessEqual(substrate.synapse_count, 3 * 2)
        self.assertLessEqual(substrate.synapse_count, substrate.synapse_capacity)
        activated = substrate.activate({3: 1.0})
        self.assertIn(3, activated)
        self.assertTrue(set(activated) & {9, 15})
        substrate.clear()
        self.assertEqual(substrate.synapse_count, 0)

    def test_competitive_plasticity_is_bounded_and_reproducible(self):
        report = run_plasticity_benchmark(seed=19)
        repeated = run_plasticity_benchmark(seed=19)
        self.assertEqual(report, repeated)
        self.assertEqual(report["benchmark"], "elia_competitive_plasticity_synthetic_v2")
        self.assertGreater(report["mean_accuracy_change"], 0.0)
        self.assertTrue(report["association_recall_all_seeds"])
        self.assertTrue(report["association_disabled_recall_empty_all_seeds"])
        self.assertTrue(all(
            row["synapses"] <= row["synapse_capacity"]
            for row in report["runs"]
        ))
        self.assertTrue(all(row["observations"] == 320 for row in report["runs"]))
        self.assertIn("does not measure", report["warning"])

    def test_competitive_plasticity_association_fanout_is_bounded(self):
        substrate = CompetitivePlasticSubstrate(
            unit_count=12,
            feature_count=32,
            top_k=3,
            max_fanout=2,
            associative_learning=False,
            seed=12,
        )
        for source in range(6):
            for target in range(6, 12):
                substrate.learn_association({source: 1.0}, {target: 1.0})
        self.assertLessEqual(substrate.synapse_count, 12 * 2)
        self.assertEqual(substrate.status()["associative_learning"], False)

    def test_competitive_plasticity_validates_inputs_and_resets_deterministically(self):
        substrate = CompetitivePlasticSubstrate(
            unit_count=16,
            feature_count=32,
            top_k=3,
            max_fanout=2,
            associative_learning=False,
            seed=7,
        )
        with self.assertRaises(ValueError):
            substrate.encode({32: 1.0})
        with self.assertRaises(ValueError):
            substrate.observe({1: float("nan")})

        activity = substrate.observe({1: 1.0, 2: 0.5})
        self.assertEqual(len(activity), 3)
        substrate.clear(seed=7)
        fresh = CompetitivePlasticSubstrate(
            unit_count=16,
            feature_count=32,
            top_k=3,
            max_fanout=2,
            associative_learning=False,
            seed=7,
        )
        self.assertEqual(substrate.encode({1: 1.0}), fresh.encode({1: 1.0}))
        self.assertEqual(substrate.observations, 0)

    def test_core_encoding_is_stable_and_sparse(self):
        core = SharedCognitiveCore(unit_count=1024, max_fanout=8, max_features=32)
        first = core.token_features("Practical project troubleshooting")
        second = core.token_features("PRACTICAL project troubleshooting")
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 32)

    def test_core_marks_bounded_negation_scope_and_resets_at_contrast(self):
        core = SharedCognitiveCore()
        features = core.token_features("The service is not at all helpful")
        helpful = stable_unit("negated:helpful", core.substrate.unit_count)
        useful = stable_unit("negated:useful", core.substrate.unit_count)

        self.assertIn(helpful, features)
        self.assertNotIn(
            helpful,
            core.token_features("The service is not only helpful but useful"),
        )
        self.assertNotIn(
            useful,
            core.token_features("The service is not bad but useful"),
        )

    def test_core_generalizes_negation_to_held_out_modifier_phrasings(self):
        core = SharedCognitiveCore()
        training = (
            ("The service is helpful", 1),
            ("The service is not helpful", 0),
            ("The answer is useful", 1),
            ("The answer is not useful", 0),
        )
        for _ in range(180):
            for text, label in training:
                core.readout.update(core.encode(text), label)

        positive = core.readout.probability(core.encode("The service is helpful"))
        held_out = (
            core.readout.probability(core.encode("The service is not really helpful")),
            core.readout.probability(core.encode("The service is not at all helpful")),
        )

        self.assertGreater(positive, 0.6)
        self.assertTrue(all(probability < 0.4 for probability in held_out))

    def test_combined_features_give_relation_signals_extra_weight(self):
        core = SharedCognitiveCore()
        token_only = core.token_features("Alice helps Bob")
        combined = core.combined_features("Alice helps Bob")
        relation_only = core.relation_features("Alice helps Bob")
        self.assertTrue(combined)
        common_units = set(relation_only) & set(token_only)
        self.assertTrue(common_units)
        self.assertTrue(any(combined[unit] > token_only[unit] for unit in common_units))
        self.assertTrue(set(relation_only) & set(combined))

    def test_feature_order_is_stable_across_python_hash_seeds(self):
        script = (
            "from elia.cognition.core import SharedCognitiveCore; "
            "print(sorted(SharedCognitiveCore().token_features("
            "'Practical project troubleshooting repeats project')))"
        )
        outputs = []
        for hash_seed in ("1", "2"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            outputs.append(subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            ).stdout.strip())
        self.assertEqual(outputs[0], outputs[1])


class PlannerPriorityTests(unittest.TestCase):
    def test_planner_reports_risk_and_priority_metadata(self):
        planner = TaskPlanner()
        recommendation = planner.plan(
            evidence_count=7,
            hypothesis_count=2,
            conflict={"id": "h1", "supports": 3, "challenges": 4},
        )
        self.assertEqual(recommendation["action"], "review_conflicting_evidence")
        self.assertIn("priority_score", recommendation)
        self.assertIn("risk_level", recommendation)
        self.assertGreater(recommendation["priority_score"], 90)
        self.assertEqual(recommendation["risk_level"], "high")

    def test_planner_keeps_multi_step_conflict_resolution_explicit(self):
        planner = TaskPlanner()
        recommendation = planner.plan(
            evidence_count=4,
            hypothesis_count=1,
            conflict={"id": "h2", "supports": 2, "challenges": 3},
            pending={"id": "p1", "context": "support graph update", "age_days": 2},
        )
        self.assertEqual(recommendation["action"], "review_conflicting_evidence")
        self.assertEqual(recommendation["resolution_mode"], "multi_step_evidence_resolution")
        self.assertIn("inspect_pending_prediction", [step["action"] for step in recommendation["plan_steps"]])
        self.assertIn("reassess_conflict_after_verification", [step["action"] for step in recommendation["plan_steps"]])
        self.assertGreater(recommendation["decision_score"], recommendation["priority_score"])


class SymbolicCompositionTests(unittest.TestCase):
    def test_bounded_symbolic_composition_benchmark_stays_high_accuracy_and_bounded(self):
        report = run_symbolic_composition_benchmark(seed=7)
        self.assertEqual(report["benchmark"], "elia_symbolic_composition_bounded_v1")
        self.assertGreaterEqual(report["mean_accuracy"], 0.9)
        self.assertLessEqual(report["max_work_used"], 10_000)
        self.assertEqual(report["status"], "pass")


class RoutingAndReplayTests(unittest.TestCase):
    def test_router_selects_known_domain_and_abstains_on_novelty(self):
        core = SharedCognitiveCore(unit_count=1024)
        router = SpecialistRouter()
        router.learn("graphics", core.token_features("graphics rendering shader"))
        router.learn("database", core.token_features("database query storage"))
        selected = router.route(core.token_features("graphics rendering pipeline"))
        unknown = router.route(core.token_features("quantum fermentation astronomy"))
        self.assertEqual(selected["specialist_id"], "graphics")
        self.assertEqual(selected["mode"], "inferred")
        self.assertIsNone(unknown["specialist_id"])
        self.assertEqual(unknown["mode"], "abstain")

    def test_router_abstains_on_blended_shift_outside_retained_examples(self):
        core = SharedCognitiveCore(unit_count=1024)
        router = SpecialistRouter(minimum_example_similarity=0.45)
        router.learn("graphics", core.token_features("graphics rendering shader compile pipeline"))
        router.learn("database", core.token_features("database query storage transaction tables"))
        known = router.route(core.token_features("graphics rendering shader pipeline error"))
        shifted = router.route(core.token_features(
            "graphics database shader transaction cloud orchestration protocol"
        ))

        self.assertEqual(known["specialist_id"], "graphics")
        self.assertGreaterEqual(known["top_example_similarity"], 0.45)
        self.assertIsNone(shifted["specialist_id"])
        self.assertEqual(shifted["mode"], "abstain")
        self.assertEqual(shifted["reason"], "outside_retained_examples")
        self.assertLess(
            shifted["top_example_similarity"],
            shifted["minimum_example_similarity"],
        )

    def test_router_exemplars_are_bounded_and_clear_with_profiles(self):
        core = SharedCognitiveCore(unit_count=1024)
        router = SpecialistRouter(exemplar_capacity=2)
        for text in ("graphics first", "graphics second", "graphics third"):
            router.learn("graphics", core.token_features(text))

        self.assertEqual(len(router._exemplars["graphics"]), 2)
        router.clear()
        self.assertEqual(router.specialist_ids, ())
        self.assertEqual(dict(router._exemplars), {})

    def test_replay_is_bounded_prioritizes_surprise_and_keeps_specialists_balanced(self):
        replay = BalancedReplayBuffer(capacity_per_specialist=2)
        replay.add(ReplayExample("a0", "a", "alpha zero", 0, 0.2, 1))
        replay.add(ReplayExample("a1", "a", "alpha one", 1, 0.3, 2))
        replay.add(ReplayExample("a2", "a", "alpha surprise", 1, 0.9, 3))
        replay.add(ReplayExample("b0", "b", "beta zero", 0, 0.6, 4))
        rows = replay.balanced_batch(2)
        self.assertEqual(replay.size, 3)
        self.assertIn("a2", {row.example_id for row in rows})
        self.assertEqual({row.specialist_id for row in rows}, {"a", "b"})
        self.assertTrue(all(any(row.specialist_id == key for row in rows) for key in ("a", "b")))

    def test_replay_batch_respects_global_limit_and_includes_current_example(self):
        replay = BalancedReplayBuffer(capacity_per_specialist=8)
        for specialist_id in ("a", "b", "c"):
            for index in range(4):
                replay.add(ReplayExample(
                    f"{specialist_id}{index}",
                    specialist_id,
                    f"{specialist_id} example {index}",
                    index % 2,
                    0.5,
                    index,
                ))
        rows = replay.balanced_batch(8, include_id="c3", max_total=5)
        self.assertEqual(len(rows), 5)
        self.assertIn("c3", {row.example_id for row in rows})


class CognitiveSystemTests(unittest.TestCase):
    def test_prediction_reuses_computed_text_features_for_routing(self):
        system = CognitiveSystem(unit_count=1024, fanout=8)
        system.observe("example", "support", "helpful troubleshooting steps", 1)

        with patch.object(
            system.core,
            "token_features",
            wraps=system.core.token_features,
        ) as encode:
            result = system.predict("helpful troubleshooting steps", "support")

        self.assertEqual(result["decision"], "predict")
        encode.assert_called_once_with("helpful troubleshooting steps")

    def test_specialists_learn_conflicting_labels_without_collapsing(self):
        system = CognitiveSystem(
            unit_count=1024,
            fanout=8,
            replay_capacity_per_specialist=32,
            replay_per_specialist=8,
        )
        for index in range(20):
            system.observe(f"positive-{index}", "alpha", "shared signal alpha pattern", 1)
            system.observe(f"negative-{index}", "beta", "shared signal alpha pattern", 0)
        alpha = system.predict("shared signal alpha pattern", "alpha")
        beta = system.predict("shared signal alpha pattern", "beta")
        self.assertEqual(alpha["decision"], "predict")
        self.assertEqual(beta["decision"], "predict")
        self.assertGreater(alpha["probability"], beta["probability"])
        self.assertEqual(system.predict("shared signal alpha pattern")["decision"], "abstain")
        self.assertLessEqual(system.status()["synapses"], system.status()["synapse_capacity"])
        self.assertEqual(
            system.status()["architecture"]["router"],
            "prototype_similarity_and_bounded_exemplar_abstention",
        )
        self.assertEqual(system.status()["routing_thresholds"]["exemplar_capacity_per_specialist"], 128)

    def test_relation_features_capture_action_and_quality_cues(self):
        core = SharedCognitiveCore(unit_count=1024)
        features = core.token_features("Dana helps Omar and is helpful")
        self.assertIn(stable_unit("rel:help:subject:dana", core.substrate.unit_count), features)
        self.assertIn(stable_unit("rel:help:object:omar", core.substrate.unit_count), features)
        self.assertIn(stable_unit("qual:is_helpful", core.substrate.unit_count), features)

    def test_task_planner_prioritizes_conflicts_before_pending_work(self):
        planner = TaskPlanner(min_validation_cases=7)
        conflict = planner.plan(
            evidence_count=3,
            hypothesis_count=2,
            conflict={"id": "h-1", "supports": 2, "challenges": 1},
            pending={"id": "p-1", "context": "follow-up question"},
            unresolved_candidate={"id": "h-2", "status": "candidate", "resolved_predictions": 2, "claim": "A test claim"},
        )
        self.assertEqual(conflict["action"], "review_conflicting_evidence")
        next_step = planner.plan(
            evidence_count=3,
            hypothesis_count=2,
            pending={"id": "p-2", "context": "needs verification"},
        )
        self.assertEqual(next_step["action"], "verify_pending_prediction")


class FunctionalLayerTests(unittest.TestCase):
    def test_workspace_caps_active_context_and_reports_evictions(self):
        workspace = ActiveWorkspace()
        for index in range(80):
            workspace.update(
                task="prediction",
                evidence=[f"evidence-{index}"],
                priorities={f"item-{index}": index / 100},
                confidence={f"item-{index}": index / 100},
                decision=f"decision-{index}",
                trace=f"trace-{index}",
            )

        status = workspace.summary()
        self.assertEqual(len(status["attention_priorities"]), 64)
        self.assertEqual(len(status["confidence"]), 64)
        self.assertEqual(len(status["recent_decisions"]), 16)
        self.assertEqual(len(status["reasoning_trace"]), 32)
        self.assertEqual(status["metrics"]["updates"], 80)
        self.assertEqual(status["metrics"]["evictions"], 32)

    def test_self_model_limits_scores_and_exposes_operational_metrics(self):
        model = SelfModel()
        model.update_task("prediction")
        model.update_task("prediction")
        model.update_confidence("support", 0.7)
        self.assertEqual(model.summary()["task_counts"], {"prediction": 2})
        self.assertEqual(model.summary()["metrics"]["subjective_experience_claimed"], False)
        with self.assertRaises(ValueError):
            model.update_confidence("support", float("nan"))
        with self.assertRaises(ValueError):
            model.set_capabilities({f"capability-{index}": 0.5 for index in range(65)})

    def test_reflection_reports_risk_without_changing_operational_state(self):
        reflection = ReflectiveAgent()
        report = reflection.evaluate(
            evidence_count=1,
            hypothesis_count=1,
            conflict={"id": "h-1", "supports": 1, "challenges": 1},
        )
        self.assertEqual(report.recommendation, "review_conflicting_evidence")
        self.assertEqual(reflection.summary()["metrics"]["evaluations"], 1)
        self.assertEqual(reflection.summary()["effects"], "reports_only_no_planner_or_memory_mutation")
        with self.assertRaises(ValueError):
            reflection.evaluate(uncertainty={"bad": float("nan")})

    def test_state_controller_records_bounded_mode_metrics_and_validates_inputs(self):
        controller = StateController()
        for _ in range(40):
            controller.decide_mode(contradictions=1)
        self.assertEqual(controller.active_mode, "reflective_review")
        self.assertEqual(len(controller.mode_history), 32)
        self.assertEqual(controller.summary()["metrics"]["decisions"], 40)
        with self.assertRaises(ValueError):
            controller.decide_mode(memory_pressure=float("inf"))

    def test_consciousness_controller_tracks_recursive_metacognition_loop(self):
        controller = ConsciousnessController()
        controller.update(
            task="evidence_scan",
            evidence=["evidence-alpha"],
            hypotheses=["h-1", "h-2"],
            questions=["missing_support"],
            confidence={"evidence-alpha": 0.72, "h-1": 0.49},
            uncertainty={"h-1": 0.8},
            conflict={"id": "h-1", "supports": 1, "challenges": 1},
            pending={"id": "p-1", "context": "needs validation"},
        )
        status = controller.summary()
        self.assertIn("metacognitive_loop", status["global_workspace"])
        loop = status["global_workspace"]["metacognitive_loop"]
        self.assertIn(loop["status"], {"review", "collect", "reframe", "steady"})
        self.assertGreaterEqual(loop["cycle"], 1)
        self.assertIn("decision_priority", loop)
        self.assertTrue(loop["revisit_queue"])

    def test_consciousness_controller_tracks_focus_and_active_mode(self):
        system = CognitiveSystem()
        system.predict("This answer is helpful and supported")
        status = system.functional_controller()
        self.assertEqual(status["controller_kind"], "bounded_functional_consciousness")
        self.assertIn("active_mode", status)
        self.assertIn("focus", status)
        self.assertIn("workspace", status)
        self.assertIn("self_model", status)
        self.assertIn("global_workspace", status)
        self.assertIn("attention_stack", status["global_workspace"])
        self.assertIn("priority_snapshot", status["global_workspace"])
        self.assertIn("functional_health", status)
        self.assertIn("score", status["functional_health"])
        system.consciousness.update(
            task="reflection",
            evidence=["evidence"],
            hypotheses=["hypothesis"],
            conflict={"id": "h-1", "supports": 1, "challenges": 1},
            uncertainty={"x": 0.9},
        )
        self.assertTrue(system.functional_status()["planner_behavior_changed"])

    def test_consolidation_queue_is_bounded_review_only_and_does_not_persist(self):
        consolidator = BackgroundConsolidator()
        for _ in range(8):
            consolidator.enqueue(evidence=["same evidence"], hypotheses=["same evidence"])
        with self.assertRaises(ValueError):
            consolidator.enqueue(evidence=["queue overflow"])
        report = consolidator.run_one()
        self.assertEqual(report["clusters"], ["same evidence"])
        self.assertFalse(report["persisted"])
        self.assertTrue(report["requires_human_review"])
        self.assertEqual(consolidator.status()["queue_depth"], 7)
        self.assertEqual(consolidator.status()["queue_rejections"], 1)
        with self.assertRaises(ValueError):
            consolidator.compress(evidence=["evidence"], retention_threshold=1.5)

    def test_benchmark_reports_retention_routing_and_limits(self):
        report = run_benchmark(seed=77, examples_per_class=3)
        self.assertEqual(report["benchmark"], "elia_synthetic_continual_learning_v1")
        self.assertIn("not a natural-language", report["warning"])
        self.assertEqual(len(report["results"]), 3)
        self.assertTrue(all(row["sample_size"] > 0 for row in report["results"]))
        self.assertTrue(all(0 <= row["routing_accuracy"] <= 1 for row in report["results"]))
        self.assertTrue(all(row["mean_forgetting"] >= 0 for row in report["results"]))

    def test_benchmark_sweep_reports_seed_variability(self):
        report = run_sweep(first_seed=77, seeds=2, examples_per_class=2)
        self.assertEqual(report["seeds"], 2)
        self.assertEqual(len(report["per_seed_results"]), 2)
        for row in report["results"]:
            self.assertEqual(row["seed_runs"], 2)
            self.assertGreaterEqual(row["accuracy_order_sd"], 0)

    def test_generalization_benchmark_status_checks_accuracy_and_both_classes(self):
        report = run_generalization_benchmark(seed=2026)
        self.assertEqual(report["benchmark"], "elia_help_relation_unseen_entity_pairs_v4")
        self.assertEqual(
            report["generalization_scope"],
            "unseen_entity_names_and_pairs_with_training_seen_templates",
        )
        self.assertTrue(report["entity_names_disjoint"])
        self.assertTrue(
            set(report["training_entity_names"]).isdisjoint(report["held_out_entity_names"])
        )
        self.assertEqual(
            len(report["held_out_entity_names"]),
            len(set(report["held_out_entity_names"])),
        )
        self.assertEqual(report["held_out_examples"], 230)
        self.assertEqual(report["held_out_positive_examples"], 115)
        self.assertEqual(report["held_out_negative_examples"], 115)
        self.assertFalse(report["unseen_templates"])
        self.assertIn("templates seen during training", report["warning"])
        self.assertTrue(report["class_balance_pass"])
        self.assertGreater(report["positive_accuracy"], 0.5)
        self.assertGreater(report["negative_accuracy"], 0.5)
        self.assertEqual(
            report["status"],
            "pass" if report["accuracy"] >= 0.8 and report["class_balance_pass"]
            else "needs_more_generalization",
        )

    def test_unseen_template_benchmark_reports_transfer_and_seed_sensitivity(self):
        report = run_unseen_template_benchmark(seed=2026)
        self.assertEqual(report["benchmark"], "elia_help_relation_unseen_templates_v1")
        self.assertEqual(report["generalization_scope"], "unseen_entity_names_and_templates")
        self.assertTrue(report["unseen_templates"])
        self.assertTrue(report["entity_names_disjoint"])
        self.assertTrue(
            set(report["training_templates"]).isdisjoint(report["held_out_templates"])
        )
        self.assertEqual(report["held_out_examples"], 230)
        self.assertEqual(report["held_out_positive_examples"], 115)
        self.assertEqual(report["held_out_negative_examples"], 115)
        self.assertEqual(len(report["template_accuracy"]), 10)
        self.assertEqual(report["status"], "pass")
        self.assertGreater(report["negative_accuracy"], 0.5)

        order_sensitive = run_unseen_template_benchmark(seed=2028)
        self.assertEqual(order_sensitive["status"], "needs_more_generalization")
        self.assertGreater(order_sensitive["positive_accuracy"], 0.5)
        self.assertLessEqual(order_sensitive["negative_accuracy"], 0.5)

    def test_unseen_template_sweep_reports_pass_rate_and_variability(self):
        report = run_unseen_template_sweep(first_seed=2026, seeds=3)
        self.assertEqual(report["benchmark"], "elia_help_relation_unseen_templates_sweep_v1")
        self.assertEqual(report["seeds"], 3)
        self.assertEqual(report["pass_count"], 2)
        self.assertAlmostEqual(report["pass_rate"], 2 / 3, places=6)
        lower, upper = report["pass_rate_wilson_95_interval"]
        self.assertLessEqual(lower, report["pass_rate"])
        self.assertGreaterEqual(upper, report["pass_rate"])
        self.assertEqual(
            report["interval_scope"],
            "sampled_training_order_pass_rate_for_this_fixed_synthetic_benchmark",
        )
        self.assertEqual(
            [row["status"] for row in report["seed_results"]],
            ["pass", "pass", "needs_more_generalization"],
        )
        self.assertGreater(report["accuracy_order_sd"], 0)
        with self.assertRaises(ValueError):
            run_unseen_template_sweep(seeds=1)


if __name__ == "__main__":
    unittest.main()
