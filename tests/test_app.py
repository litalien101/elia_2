import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elia.app import (
    ApiError,
    ReasoningCache,
    benchmark_memory_compression,
    compress_memory,
    create_episode,
    create_hypothesis,
    create_observation,
    create_prediction,
    create_reasoning_fact,
    create_reasoning_rule,
    create_reviewed_language_facts,
    dashboard_data,
    deliberation_history,
    delete_episode,
    evaluate,
    initialize,
    make_server,
    record_outcome,
    run_sleep_cycle,
    run_deliberation,
    reason_about,
    session,
)
from elia.cognition import CognitiveSystem
from elia.cognition.core import _token_feature_signature


class LearningLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "elia.sqlite3"
        initialize(self.db_path)

    def tearDown(self):
        self.temp.cleanup()

    def episode_and_observation(self, content="The user chose concise troubleshooting steps."):
        with session(self.db_path) as db:
            episode = create_episode(db, {
                "title": "Explicit preference",
                "source_kind": "user_note",
                "content": content,
                "consented": True,
            })
            observation = create_observation(db, {
                "episode_id": episode["id"],
                "statement": "The user explicitly chose concise troubleshooting steps.",
            })
            return episode["id"], observation["id"]

    def hypothesis(self, observation_id):
        with session(self.db_path) as db:
            return create_hypothesis(db, {
                "claim": "For technical support, the user will prefer concise steps.",
                "scope": ["technical support"],
                "conditions": ["The request is troubleshooting"],
                "expected_outcome": "The user explicitly chooses a concise answer",
                "evidence": [{"observation_id": observation_id, "stance": "supports"}],
            })

    def test_memory_compression_reduces_duplicate_evidence_into_fragments(self):
        with session(self.db_path) as db:
            episode_a = create_episode(db, {
                "title": "Repeated preference",
                "source_kind": "chat",
                "content": "The user likes concise instructions.",
                "consented": True,
            })
            episode_b = create_episode(db, {
                "title": "Repeated preference confirmation",
                "source_kind": "chat",
                "content": "The user likes concise instructions.",
                "consented": True,
            })
            observation_a = create_observation(db, {
                "episode_id": episode_a["id"],
                "statement": "The user likes concise instructions.",
            })
            observation_b = create_observation(db, {
                "episode_id": episode_b["id"],
                "statement": "The user likes concise instructions.",
            })
            create_hypothesis(db, {
                "claim": "The user likes concise instructions.",
                "scope": ["support"],
                "conditions": ["The user is asked for help"],
                "expected_outcome": "The answer is concise",
                "evidence": [
                    {"observation_id": observation_a["id"], "stance": "supports"},
                    {"observation_id": observation_b["id"], "stance": "supports"},
                ],
            })
            report = compress_memory(db, strategy="evidence_preserving", max_fragments=8)
            self.assertEqual(report["coverage_method"], "legacy_or_unbenchmarked_memory_compression")
            latest_unbenchmarked = dashboard_data(db)["latest_memory_compression"]
            self.assertEqual(
                latest_unbenchmarked["coverage_method"],
                "legacy_or_unbenchmarked_memory_compression",
            )
            self.assertIsNone(latest_unbenchmarked["task_coverage"])
            benchmark = benchmark_memory_compression(
                db,
                strategy="evidence_preserving",
                max_fragments=8,
                retention_threshold=0.5,
            )
            self.assertEqual(report["strategy"], "evidence_preserving")
            self.assertGreater(report["original_units"], report["retained_units"])
            self.assertGreater(report["reduction_ratio"], 0)
            self.assertGreaterEqual(report["fragments"][0]["record_count"], 2)
            self.assertIn("Merged", report["fragments"][0]["summary"])
            self.assertGreaterEqual(benchmark["benchmark"]["coverage"], 0.5)
            self.assertEqual(benchmark["benchmark"]["retention_threshold"], 0.5)
            self.assertIn("task_preservation", benchmark["benchmark"])
            self.assertIn("retention_gate", benchmark["benchmark"])
            self.assertEqual(
                benchmark["benchmark"]["memory_key_coverage"]["method"],
                "in_sample_normalized_memory_key_coverage_unkeyed_records_lost",
            )
            self.assertEqual(
                benchmark["task_coverage"],
                benchmark["benchmark"]["memory_key_coverage"]["coverage"],
            )
            self.assertEqual(benchmark["held_out_coverage"], benchmark["task_coverage"])
            persisted = db.execute(
                "SELECT preserved_claims, summary FROM memory_compressions WHERE id = ?",
                (benchmark["id"],),
            ).fetchone()
            self.assertIsNotNone(persisted)
            self.assertGreaterEqual(float(persisted["preserved_claims"]), 0.5)
            self.assertIn("task coverage", persisted["summary"])

            for invalid_threshold in (
                1.5, 10**400, float("nan"), float("inf"), float("-inf"), True,
            ):
                with self.subTest(retention_threshold=invalid_threshold):
                    with self.assertRaises(ApiError):
                        benchmark_memory_compression(db, retention_threshold=invalid_threshold)
            for invalid_max_fragments in (True, 1.5, 0):
                with self.subTest(max_fragments=invalid_max_fragments):
                    with self.assertRaises(ApiError):
                        compress_memory(db, max_fragments=invalid_max_fragments)

    def test_memory_compression_empty_store_passes_retention_gate(self):
        with session(self.db_path) as db:
            result = benchmark_memory_compression(db, retention_threshold=0.8)
            self.assertEqual(result["benchmark"]["coverage"], 1.0)
            self.assertTrue(result["benchmark"]["retention_gate"]["accepted"])
            self.assertEqual(result["benchmark"]["memory_key_coverage"]["coverage"], 1.0)
            self.assertEqual(result["task_coverage"], 1.0)
            row = db.execute(
                "SELECT retention_threshold, retention_gate_accepted, held_out_coverage FROM memory_compressions WHERE id = ?",
                (result["id"],),
            ).fetchone()
            self.assertEqual(float(row["retention_threshold"]), 0.8)
            self.assertEqual(int(row["retention_gate_accepted"]), 1)
            self.assertEqual(float(row["held_out_coverage"]), 1.0)

    def test_memory_compression_reports_in_sample_coverage_and_rejects_loss(self):
        with session(self.db_path) as db:
            for text in ("The first distinct observation.", "A second unrelated observation."):
                episode = create_episode(db, {
                    "title": text,
                    "source_kind": "test",
                    "content": text,
                    "consented": True,
                })
                create_observation(db, {
                    "episode_id": episode["id"],
                    "statement": text,
                })

            result = benchmark_memory_compression(
                db,
                max_fragments=1,
                retention_threshold=0.75,
            )

            self.assertEqual(result["task_coverage"], 0.5)
            self.assertFalse(result["retention_gate"]["accepted"])
            self.assertEqual(
                result["coverage_method"],
                "in_sample_normalized_memory_key_coverage_unkeyed_records_lost",
            )
            self.assertEqual(result["benchmark"]["memory_key_coverage"]["evaluated_keys"], 2)

    def test_memory_benchmark_counts_unkeyed_records_as_lost(self):
        with session(self.db_path) as db:
            for index, text in enumerate(("A useful observation.", "💡")):
                episode = create_episode(db, {
                    "title": f"Memory {index}",
                    "source_kind": "test",
                    "content": text,
                    "consented": True,
                })
                create_observation(db, {
                    "episode_id": episode["id"],
                    "statement": text,
                })

            result = benchmark_memory_compression(
                db,
                retention_threshold=0.75,
            )

            coverage = result["benchmark"]["memory_key_coverage"]
            self.assertEqual(coverage["evaluated_keys"], 1)
            self.assertEqual(coverage["unkeyed_memory_units"], 1)
            self.assertEqual(coverage["evaluated_units"], 2)
            self.assertEqual(coverage["coverage"], 0.5)
            self.assertFalse(result["retention_gate"]["accepted"])

    def test_memory_compression_preserves_unicode_memory_keys(self):
        with session(self.db_path) as db:
            texts = ("Mañana, ayuda.", "MAN\u0303ANA ayuda!", "你好，朋友。")
            for index, text in enumerate(texts):
                episode = create_episode(db, {
                    "title": f"Unicode note {index}",
                    "source_kind": "test",
                    "content": text,
                    "consented": True,
                })
                create_observation(db, {
                    "episode_id": episode["id"],
                    "statement": text,
                })

            report = compress_memory(db)

            self.assertEqual(len(report["fragments"]), 2)
            fragments_by_key = {fragment["memory_key"]: fragment for fragment in report["fragments"]}
            self.assertEqual(fragments_by_key["mañana ayuda"]["record_count"], 2)
            self.assertEqual(fragments_by_key["你好 朋友"]["record_count"], 1)
            self.assertEqual(report["original_units"], 3)

    def test_memory_compression_persists_dashboard_threshold_status(self):
        with session(self.db_path) as db:
            episode = create_episode(db, {
                "title": "Thresholded memory",
                "source_kind": "chat",
                "content": "The user likes concise instructions.",
                "consented": True,
            })
            observation = create_observation(db, {
                "episode_id": episode["id"],
                "statement": "The user likes concise instructions.",
            })
            create_hypothesis(db, {
                "claim": "The user likes concise instructions.",
                "scope": ["support"],
                "conditions": ["The user is asked for help"],
                "expected_outcome": "The answer is concise",
                "evidence": [{"observation_id": observation["id"], "stance": "supports"}],
            })
            result = benchmark_memory_compression(db, retention_threshold=0.5)
            dashboard = dashboard_data(db)
            latest = dashboard["latest_memory_compression"]
            self.assertEqual(float(latest["retention_threshold"]), 0.5)
            self.assertEqual(int(latest["retention_gate_accepted"]), 1)
            self.assertEqual(float(latest["held_out_coverage"]), float(result["held_out_coverage"]))
            self.assertEqual(float(latest["task_coverage"]), float(result["task_coverage"]))
            self.assertEqual(
                latest["coverage_method"],
                "in_sample_normalized_memory_key_coverage_unkeyed_records_lost",
            )

    def test_reasoning_cache_updates_when_new_evidence_arrives(self):
        with session(self.db_path) as db:
            episode = create_episode(db, {
                "title": "New evidence",
                "source_kind": "chat",
                "content": "A user note about support quality.",
                "consented": True,
            })
            observation = create_observation(db, {
                "episode_id": episode["id"],
                "statement": "A note about support quality.",
            })
            cache = ReasoningCache()
            before = reason_about(db, "help(alice, bob)", cache)
            self.assertEqual(before["state"], "unknown")
            create_reasoning_fact(db, {
                "fact": "help(alice, bob)",
                "observation_id": observation["id"],
            })
            after = reason_about(db, "help(alice, bob)", cache)
            self.assertEqual(after["state"], "supported")

    def test_prediction_cache_invalidates_after_learning_new_example(self):
        system = CognitiveSystem()
        text = "graphics rendering pipeline is fast and reliable"
        system.observe("learned_example_1", "graphics", text, 1)
        before = system.predict(text, "graphics")
        system.observe("learned_example_2", "graphics", text, 0)
        after = system.predict(text, "graphics")
        self.assertNotEqual(before["probability"], after["probability"])

    def test_consent_is_required_and_source_offsets_are_checked(self):
        with session(self.db_path) as db:
            with self.assertRaises(ApiError):
                create_episode(db, {
                    "title": "No consent", "source_kind": "chat", "content": "text", "consented": False
                })
        episode_id, _ = self.episode_and_observation("short")
        with session(self.db_path) as db:
            with self.assertRaises(ApiError):
                create_observation(db, {
                    "episode_id": episode_id, "statement": "outside source",
                    "evidence_start": 1, "evidence_end": 100,
                })

    def test_database_schema_version_is_initialized(self):
        with session(self.db_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 10)
            self.assertIsNotNone(db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sleep_runs'"
            ).fetchone())
            self.assertIsNotNone(db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'thinking_runs'"
            ).fetchone())
            self.assertIsNotNone(db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'reasoning_rules'"
            ).fetchone())
            self.assertIsNotNone(db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memory_compressions'"
            ).fetchone())

    def test_version_three_database_migrates_sleep_schema(self):
        with session(self.db_path) as db:
            db.execute("DROP TABLE sleep_runs")
            db.execute("DROP TABLE thinking_runs")
            db.execute("PRAGMA user_version = 3")
        initialize(self.db_path)
        with session(self.db_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 10)
            self.assertIsNotNone(db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sleep_runs'"
            ).fetchone())
            self.assertIsNotNone(db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'thinking_runs'"
            ).fetchone())

    def test_version_five_database_migrates_reasoning_schema(self):
        with session(self.db_path) as db:
            db.execute("DROP TABLE reasoning_rules")
            db.execute("DROP TABLE reasoning_facts")
            db.execute("PRAGMA user_version = 5")
        initialize(self.db_path)
        with session(self.db_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 10)
            for table in ("reasoning_facts", "reasoning_rules"):
                self.assertIsNotNone(db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone())
            self.assertIsNotNone(db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memory_compressions'"
            ).fetchone())

    def test_version_eight_memory_reports_migrate_as_legacy_unbenchmarked(self):
        legacy_path = Path(self.temp.name) / "legacy.sqlite3"
        legacy_db = sqlite3.connect(legacy_path)
        try:
            db = legacy_db
            db.execute(
                "CREATE TABLE memory_compressions ("
                "id TEXT PRIMARY KEY, strategy TEXT NOT NULL, original_units INTEGER NOT NULL, "
                "retained_units INTEGER NOT NULL, reduction_ratio REAL NOT NULL, "
                "preserved_claims REAL NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL, "
                "retention_threshold REAL NOT NULL DEFAULT 0.8, "
                "retention_gate_accepted INTEGER NOT NULL DEFAULT 1, "
                "held_out_coverage REAL NOT NULL DEFAULT 1.0)"
            )
            db.execute(
                "INSERT INTO memory_compressions "
                "(id, strategy, original_units, retained_units, reduction_ratio, preserved_claims, summary, created_at) "
                "VALUES ('legacy', 'test', 1, 1, 0, 1, 'old report', '2026-01-01')"
            )
            db.execute("PRAGMA user_version = 8")
            db.commit()
        finally:
            legacy_db.close()

        initialize(legacy_path)

        with session(legacy_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 10)
            self.assertEqual(
                db.execute(
                    "SELECT coverage_method, summary FROM memory_compressions WHERE id = 'legacy'"
                ).fetchone()["coverage_method"],
                "legacy_or_unbenchmarked_memory_compression",
            )
            self.assertEqual(
                db.execute(
                    "SELECT summary FROM memory_compressions WHERE id = 'legacy'"
                ).fetchone()["summary"],
                "Legacy compression report; retention coverage method unavailable.",
            )

    def test_version_nine_legacy_benchmark_summaries_are_neutralized(self):
        legacy_path = Path(self.temp.name) / "version-nine.sqlite3"
        legacy_db = sqlite3.connect(legacy_path)
        try:
            legacy_db.execute(
                "CREATE TABLE memory_compressions ("
                "id TEXT PRIMARY KEY, strategy TEXT NOT NULL, original_units INTEGER NOT NULL, "
                "retained_units INTEGER NOT NULL, reduction_ratio REAL NOT NULL, "
                "preserved_claims REAL NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL, "
                "retention_threshold REAL NOT NULL DEFAULT 0.8, "
                "retention_gate_accepted INTEGER NOT NULL DEFAULT 1, "
                "held_out_coverage REAL NOT NULL DEFAULT 1.0, coverage_method TEXT NOT NULL)"
            )
            legacy_db.execute(
                "INSERT INTO memory_compressions VALUES "
                "('legacy', 'test', 1, 1, 0, 1, "
                "'test retained 100.0% task coverage after compression; gate accepted.', "
                "'2026-01-01', 0.8, 1, 1.0, ?)",
                ("legacy_or_unbenchmarked_memory_compression",),
            )
            legacy_db.execute("PRAGMA user_version = 9")
            legacy_db.commit()
        finally:
            legacy_db.close()

        initialize(legacy_path)

        with session(legacy_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 10)
            self.assertEqual(
                db.execute(
                    "SELECT summary FROM memory_compressions WHERE id = 'legacy'"
                ).fetchone()["summary"],
                "Legacy compression report; retention coverage method unavailable.",
            )

    def test_deliberation_prioritizes_conflict_then_pending_evidence_and_records_history(self):
        _, support_id = self.episode_and_observation()
        _, challenge_id = self.episode_and_observation(
            "The user rejected concise troubleshooting steps."
        )
        with session(self.db_path) as db:
            hypothesis = create_hypothesis(db, {
                "claim": "A testable claim with mixed evidence",
                "scope": ["technical support"],
                "expected_outcome": "The user chooses concise instructions",
                "evidence": [
                    {"observation_id": support_id, "stance": "supports"},
                    {"observation_id": challenge_id, "stance": "challenges"},
                ],
            })
            prediction = create_prediction(db, {
                "hypothesis_id": hypothesis["id"],
                "context": "A pending case",
                "probability": 0.7,
            })
            first = run_deliberation(db)
            self.assertEqual(first["recommendation"]["action"], "review_conflicting_evidence")
            self.assertEqual(first["recommendation"]["target_id"], hypothesis["id"])
            self.assertTrue(first["requires_human_approval"])
            self.assertFalse(first["executed"])

            db.execute("DELETE FROM hypothesis_evidence WHERE hypothesis_id = ?", (hypothesis["id"],))
            second = run_deliberation(db)
            self.assertEqual(second["recommendation"]["action"], "verify_pending_prediction")
            self.assertEqual(second["recommendation"]["target_id"], prediction["id"])
            self.assertEqual(deliberation_history(db)["latest"]["id"], second["id"])

    def test_deliberation_without_evidence_does_not_invent_a_claim(self):
        with session(self.db_path) as db:
            thought = run_deliberation(db)
        self.assertEqual(
            thought["recommendation"]["action"],
            "request_consent_based_evidence",
        )
        self.assertIsNone(thought["recommendation"]["target_id"])
        self.assertFalse(thought["executed"])

    def test_reasoning_entries_are_provenance_linked_and_deleted_with_their_source(self):
        episode_id, observation_id = self.episode_and_observation()
        with session(self.db_path) as db:
            create_reasoning_fact(db, {
                "fact": "parent(alice, bob)",
                "observation_id": observation_id,
            })
            create_reasoning_rule(db, {
                "rule": "parent(?x, ?y) -> ancestor(?x, ?y)",
                "observation_id": observation_id,
            })
            with self.assertRaises(ApiError):
                create_reasoning_rule(db, {
                    "rule": "parent(?x, ?y) -> ancestor(?x, ?y)",
                    "observation_id": observation_id,
                })
            result = reason_about(db, "ancestor(alice, ?person)")
            self.assertEqual(result["state"], "supported")
            self.assertEqual(result["answers"][0]["bindings"], {"?person": "bob"})
            self.assertEqual(result["answers"][0]["proof"]["kind"], "rule")
            self.assertEqual(
                result["answers"][0]["proof"]["source_observation_id"],
                observation_id,
            )
            self.assertEqual(reason_about(db, "ancestor(cara, ?person)")["state"], "unknown")

            delete_episode(db, episode_id)
            records = db.execute(
                "SELECT (SELECT COUNT(*) FROM reasoning_facts), "
                "(SELECT COUNT(*) FROM reasoning_rules)"
            ).fetchone()
            self.assertEqual(tuple(records), (0, 0))

    def test_reasoning_cache_reuses_closure_and_invalidates_after_new_fact(self):
        _, observation_id = self.episode_and_observation()
        cache = ReasoningCache()
        with session(self.db_path) as db:
            create_reasoning_fact(db, {
                "fact": "parent(alice, bob)",
                "observation_id": observation_id,
            })
        with session(self.db_path) as db:
            first = reason_about(db, "parent(alice, ?child)", cache)
            built_engine = cache._engine
            second = reason_about(db, "parent(alice, ?child)", cache)
            self.assertIs(cache._engine, built_engine)
            self.assertEqual(first["work_used"], second["work_used"])
            create_reasoning_fact(db, {
                "fact": "parent(alice, cara)",
                "observation_id": observation_id,
            })
            changed = reason_about(db, "parent(alice, ?child)", cache)
            self.assertIsNot(cache._engine, built_engine)
            self.assertEqual(len(changed["answers"]), 2)

    def test_reviewed_language_facts_require_exact_source_span_and_confirmation(self):
        episode_id, _ = self.episode_and_observation("Alice likes Bob.")
        with session(self.db_path) as db:
            observation = create_observation(db, {
                "episode_id": episode_id,
                "statement": "Alice likes Bob.",
            })
            proposal = {
                "fact": "like(alice, bob)",
                "start": 0,
                "end": 16,
                "text": "Alice likes Bob.",
            }
            with self.assertRaises(ApiError):
                create_reviewed_language_facts(db, {
                    "observation_id": observation["id"],
                    "proposals": [proposal],
                })
            with self.assertRaises(ApiError):
                create_reviewed_language_facts(db, {
                    "observation_id": observation["id"],
                    "reviewed": True,
                    "proposals": [{**proposal, "start": 1}],
                })
            with self.assertRaises(ApiError):
                create_reviewed_language_facts(db, {
                    "observation_id": observation["id"],
                    "reviewed": True,
                    "proposals": [{**proposal, "fact": "trust(alice, bob)"}],
                })
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM reasoning_facts").fetchone()[0],
                0,
            )

            saved = create_reviewed_language_facts(db, {
                "observation_id": observation["id"],
                "reviewed": True,
                "proposals": [proposal],
            })
            self.assertEqual(saved["count"], 1)
            self.assertEqual(saved["facts"][0]["fact"], proposal["fact"])
            self.assertEqual(saved["facts"][0]["observation_id"], observation["id"])
            with self.assertRaises(ApiError):
                create_reviewed_language_facts(db, {
                    "observation_id": observation["id"],
                    "reviewed": True,
                    "proposals": [proposal],
                })

    def test_sleep_cycle_persists_evaluation_and_reports_contradictions(self):
        _, supporting_observation_id = self.episode_and_observation()
        _, challenging_observation_id = self.episode_and_observation(
            "The user rejected concise troubleshooting steps."
        )
        with session(self.db_path) as db:
            hypothesis = create_hypothesis(db, {
                "claim": "For technical support, the user will prefer concise steps.",
                "scope": ["technical support"],
                "expected_outcome": "The user explicitly chooses a concise answer",
                "evidence": [
                    {"observation_id": supporting_observation_id, "stance": "supports"},
                    {"observation_id": challenging_observation_id, "stance": "challenges"},
                ],
            })
            for index in range(10):
                prediction = create_prediction(db, {
                    "hypothesis_id": hypothesis["id"],
                    "context": f"Sleep cycle test {index}",
                    "probability": 0.9,
                })
                record_outcome(db, {
                    "prediction_id": prediction["id"],
                    "observed": True,
                    "verification_source": "Test fixture outcome",
                })
            report = run_sleep_cycle(db)
            persisted = db.execute(
                "SELECT * FROM sleep_runs WHERE id = ?", (report["id"],)
            ).fetchone()
            evaluation_count = db.execute(
                "SELECT COUNT(*) FROM evaluation_runs WHERE hypothesis_id = ?",
                (hypothesis["id"],),
            ).fetchone()[0]

        self.assertEqual(report["evaluated_predictions"], 10)
        self.assertEqual(report["hypotheses_evaluated"], 1)
        self.assertEqual(report["status_changes"], 1)
        self.assertEqual(report["validated_hypotheses"], 1)
        self.assertEqual(report["challenged_hypotheses"], 0)
        self.assertEqual(report["contradictory_hypotheses"], 1)
        self.assertIsNotNone(report["predictor_brier"])
        self.assertIsNotNone(report["baseline_brier"])
        self.assertIsNotNone(persisted)
        self.assertEqual(evaluation_count, 1)

    def test_empty_sleep_cycle_is_recorded_without_fake_metrics(self):
        with session(self.db_path) as db:
            report = run_sleep_cycle(db)
        self.assertEqual(report["evaluated_predictions"], 0)
        self.assertEqual(report["hypotheses_evaluated"], 0)
        self.assertIsNone(report["predictor_brier"])
        self.assertIsNone(report["baseline_brier"])

    def test_hypothesis_requires_provenance_and_prediction_captures_baseline(self):
        _, observation_id = self.episode_and_observation()
        hypothesis = self.hypothesis(observation_id)
        with session(self.db_path) as db:
            prediction = create_prediction(db, {
                "hypothesis_id": hypothesis["id"], "context": "New technical request",
                "probability": 0.8,
            })
            self.assertEqual(prediction["baseline_probability"], 0.5)
            with self.assertRaises(ApiError):
                create_prediction(db, {
                    "hypothesis_id": hypothesis["id"], "context": " new technical request ",
                    "probability": 0.7,
                })
            with self.assertRaises(ApiError):
                create_hypothesis(db, {
                    "claim": "Unsupported claim", "scope": ["general"],
                    "expected_outcome": "anything", "evidence": [],
                })

    def test_prediction_score_is_immutable_and_unknown_is_not_scored(self):
        _, observation_id = self.episode_and_observation()
        hypothesis = self.hypothesis(observation_id)
        with session(self.db_path) as db:
            prediction = create_prediction(db, {
                "hypothesis_id": hypothesis["id"], "context": "Unknown outcome case",
                "probability": 0.9,
            })
            record_outcome(db, {"prediction_id": prediction["id"], "observed": None})
            result = evaluate(db, hypothesis["id"])
            self.assertEqual(result["results"], [])
            with self.assertRaises(ApiError):
                record_outcome(db, {"prediction_id": prediction["id"], "observed": True})

    def test_promotion_requires_ten_cases_and_beats_captured_baseline(self):
        _, observation_id = self.episode_and_observation()
        hypothesis = self.hypothesis(observation_id)
        with session(self.db_path) as db:
            for index in range(10):
                prediction = create_prediction(db, {
                    "hypothesis_id": hypothesis["id"], "context": f"Distinct prospective case {index}",
                    "probability": 0.9,
                })
                record_outcome(db, {
                    "prediction_id": prediction["id"], "observed": True,
                    "verification_source": "User explicitly confirmed",
                })
            result = evaluate(db, hypothesis["id"])
            self.assertEqual(result["results"][0]["status"], "validated")
            row = db.execute("SELECT status FROM hypotheses WHERE id = ?", (hypothesis["id"],)).fetchone()
            self.assertEqual(row["status"], "validated")

    def test_hypothesis_that_loses_to_baseline_is_challenged(self):
        _, observation_id = self.episode_and_observation()
        hypothesis = self.hypothesis(observation_id)
        with session(self.db_path) as db:
            for index in range(10):
                prediction = create_prediction(db, {
                    "hypothesis_id": hypothesis["id"], "context": f"Distinct negative case {index}",
                    "probability": 0.9,
                })
                record_outcome(db, {
                    "prediction_id": prediction["id"], "observed": False,
                    "verification_source": "Explicit negative result",
                })
            result = evaluate(db, hypothesis["id"])
            self.assertEqual(result["results"][0]["status"], "challenged")
            with self.assertRaises(ApiError):
                create_prediction(db, {
                    "hypothesis_id": hypothesis["id"], "context": "After challenge",
                    "probability": 0.5,
                })

    def test_episode_deletion_cascades_and_invalidates_derived_hypothesis(self):
        episode_id, observation_id = self.episode_and_observation()
        remaining_episode_id, remaining_observation_id = self.episode_and_observation("Independent source remains.")
        with session(self.db_path) as db:
            hypothesis = create_hypothesis(db, {
                "claim": "For technical support, the user will prefer concise steps.",
                "scope": ["technical support"],
                "conditions": ["The request is troubleshooting"],
                "expected_outcome": "The user explicitly chooses a concise answer",
                "evidence": [
                    {"observation_id": observation_id, "stance": "supports"},
                    {"observation_id": remaining_observation_id, "stance": "supports"},
                ],
            })
        with session(self.db_path) as db:
            prediction = create_prediction(db, {
                "hypothesis_id": hypothesis["id"], "context": "Potentially derived context",
                "probability": 0.5,
            })
            record_outcome(db, {
                "prediction_id": prediction["id"], "observed": True,
                "verification_source": "Explicit feedback",
            })
            evaluate(db, hypothesis["id"])
            run_sleep_cycle(db)
            run_deliberation(db)
        with session(self.db_path) as db:
            delete_episode(db, episode_id)
            self.assertIsNone(db.execute(
                "SELECT id FROM hypotheses WHERE id = ?", (hypothesis["id"],)
            ).fetchone())
            self.assertEqual(db.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 1)
            self.assertIsNotNone(db.execute(
                "SELECT id FROM episodes WHERE id = ?", (remaining_episode_id,)
            ).fetchone())
            self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sleep_runs").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM thinking_runs").fetchone()[0], 0)

    def test_episode_deletion_invalidates_aggregate_memory_compression_reports(self):
        episode_id, _ = self.episode_and_observation()
        with session(self.db_path) as db:
            benchmark_memory_compression(db)
            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM memory_compressions"
            ).fetchone()[0], 1)

            delete_episode(db, episode_id)

            self.assertEqual(db.execute(
                "SELECT COUNT(*) FROM memory_compressions"
            ).fetchone()[0], 0)
            self.assertIsNone(dashboard_data(db)["latest_memory_compression"])

    def test_duplicate_evidence_link_is_rejected(self):
        _, observation_id = self.episode_and_observation()
        with session(self.db_path) as db:
            with self.assertRaises(ApiError):
                create_hypothesis(db, {
                    "claim": "Repeated evidence is not independent",
                    "scope": ["test"],
                    "expected_outcome": "A result",
                    "evidence": [
                        {"observation_id": observation_id, "stance": "supports"},
                        {"observation_id": observation_id, "stance": "supports"},
                    ],
                })


class HttpApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = make_server(Path(self.temp.name) / "api.sqlite3", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urlopen(request, timeout=3) as response:
                body = response.read()
                if response.headers.get_content_type() == "application/json":
                    body = json.loads(body)
                return response.status, body
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_memory_benchmark_rejects_non_finite_thresholds(self):
        for threshold in (float("nan"), float("inf"), float("-inf"), 10**400):
            with self.subTest(retention_threshold=threshold):
                status, result = self.request(
                    "POST",
                    "/api/v1/memory/benchmark",
                    {"retention_threshold": threshold},
                )
                self.assertEqual(status, 400)
                self.assertIn("retention_threshold", result["error"])

    def test_memory_compress_api_marks_report_unbenchmarked(self):
        status, report = self.request(
            "POST",
            "/api/v1/memory/compress",
            {"max_fragments": 4},
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            report["coverage_method"],
            "legacy_or_unbenchmarked_memory_compression",
        )
        status, history = self.request("GET", "/api/v1/memory")
        self.assertEqual(status, 200)
        self.assertEqual(
            history["compressions"][0]["coverage_method"],
            "legacy_or_unbenchmarked_memory_compression",
        )
        status, dashboard = self.request("GET", "/api/v1/dashboard")
        self.assertEqual(status, 200)
        self.assertIsNone(
            dashboard["latest_memory_compression"]["task_coverage"]
        )

    def test_functional_layer_api_is_additive_and_consolidation_is_review_only(self):
        status, layers = self.request("GET", "/api/v1/functional/status")
        self.assertEqual(status, 200)
        self.assertEqual(layers["workspace"]["task"], "idle")
        self.assertIn("global_workspace", layers)
        self.assertIn("attention_stack", layers["global_workspace"])
        self.assertIn("priority_snapshot", layers["global_workspace"])
        self.assertIn("functional_health", layers)
        self.assertIn("score", layers["functional_health"])
        self.assertFalse(layers["planner_behavior_changed"])
        self.assertFalse(layers["persistent_memory_changed_by_layers"])

        status, updated = self.request(
            "POST",
            "/api/v1/functional/update",
            {
                "task": "source_review",
                "evidence": ["source A", "source B"],
                "hypotheses": ["supports decision"],
                "questions": ["is the source sufficient?"],
                "priorities": {"supports decision": 0.9},
                "confidence": {"source_review": 0.8},
                "uncertainty": {"source_review": 0.85},
                "conflict": {"id": "c-1", "supports": 1, "challenges": 1},
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["workspace"]["task"], "source_review")
        self.assertIn("priority_snapshot", updated["global_workspace"])
        self.assertTrue(updated["planner_behavior_changed"])

        status, prediction = self.request("POST", "/api/v1/cognition/predict", {
            "text": "a transient request without a trained route",
        })
        self.assertEqual(status, 200)
        self.assertEqual(prediction["decision"], "abstain")
        status, focus = self.request("GET", "/api/v1/functional/focus")
        self.assertEqual(status, 200)
        self.assertEqual(focus["active_evidence_count"], 1)

        status, queued = self.request(
            "POST", "/api/v1/functional/consolidation/enqueue", {}
        )
        self.assertEqual(status, 202)
        self.assertEqual(queued["queued_items"], 1)
        status, report = self.request(
            "POST", "/api/v1/functional/consolidation/run", {}
        )
        self.assertEqual(status, 200)
        self.assertFalse(report["persisted"])
        self.assertTrue(report["requires_human_review"])
        status, empty_run = self.request(
            "POST", "/api/v1/functional/consolidation/run", {}
        )
        self.assertEqual(status, 409)
        self.assertIn("empty", empty_run["error"])

    def test_health_dashboard_and_validation_error(self):
        status, health = self.request("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["cpu_only"])
        status, result = self.request("POST", "/api/v1/episodes", {
            "title": "Missing consent", "source_kind": "user_note",
            "content": "should not be stored", "consented": False,
        })
        self.assertEqual(status, 400)
        self.assertIn("consent", result["error"].lower())
        status, dashboard = self.request("GET", "/api/v1/dashboard")
        self.assertEqual(status, 200)
        self.assertIn("functional", dashboard)
        self.assertIn("global_workspace", dashboard["functional"])
        self.assertIn("attention_stack", dashboard["functional"]["global_workspace"])
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"ELIA", page if isinstance(page, bytes) else str(page).encode())
        status, sleep = self.request("POST", "/api/v1/sleep", {})
        self.assertEqual(status, 200)
        self.assertEqual(sleep["evaluated_predictions"], 0)
        status, sleep_history = self.request("GET", "/api/v1/sleep")
        self.assertEqual(status, 200)
        self.assertEqual(sleep_history["latest"]["id"], sleep["id"])

    def test_source_deletion_forgets_its_specialist_and_retains_other_learning(self):
        source_specs = (
            ("deleted-skill", "deleted specialist", "amber device maintenance"),
            ("retained-skill", "retained specialist", "violet garden planning"),
        )
        sources = {}
        for specialist_id, title, phrase in source_specs:
            status, episode = self.request("POST", "/api/v1/episodes", {
                "title": title,
                "source_kind": "user_note",
                "content": f"Consented example: {phrase}.",
                "consented": True,
            })
            self.assertEqual(status, 201)
            sources[specialist_id] = episode["id"]
            for index in range(3):
                status, observation = self.request("POST", "/api/v1/observations", {
                    "episode_id": episode["id"],
                    "statement": f"{phrase} case {index}",
                })
                self.assertEqual(status, 201)
                status, learned = self.request("POST", "/api/v1/cognition/experiences", {
                    "episode_id": episode["id"],
                    "observation_id": observation["id"],
                    "specialist_id": specialist_id,
                    "label": True,
                    "verification_source": "Synthetic source-deletion test label",
                    "consented": True,
                })
                self.assertEqual(status, 201)

        status, before_deleted = self.request("POST", "/api/v1/cognition/predict", {
            "text": "amber device maintenance",
            "specialist_id": "deleted-skill",
        })
        self.assertEqual(status, 200)
        self.assertEqual(before_deleted["decision"], "predict")
        status, before_retained = self.request("POST", "/api/v1/cognition/predict", {
            "text": "violet garden planning",
            "specialist_id": "retained-skill",
        })
        self.assertEqual(status, 200)
        self.assertEqual(before_retained["decision"], "predict")

        status, counterfactual = self.request("POST", "/api/v1/cognition/counterfactual", {
            "episode_id": sources["deleted-skill"],
            "text": "amber device maintenance",
            "specialist_id": "deleted-skill",
        })
        self.assertEqual(status, 200)
        self.assertEqual(counterfactual["withheld_experience_count"], 3)
        self.assertEqual(counterfactual["retained_experience_count"], 3)
        self.assertEqual(
            counterfactual["complete_memory_prediction"]["decision"], "predict"
        )
        self.assertEqual(counterfactual["complete_memory_prediction"], before_deleted)
        self.assertEqual(
            counterfactual["without_selected_source_prediction"]["decision"], "abstain"
        )
        self.assertTrue(counterfactual["prediction_changed"])
        self.assertIn("not proof", counterfactual["interpretation_boundary"])
        status, missing_counterfactual = self.request(
            "POST", "/api/v1/cognition/counterfactual", {
                "episode_id": "missing-episode",
                "text": "amber device maintenance",
            }
        )
        self.assertEqual(status, 404)

        status, deletion = self.request(
            "DELETE", f"/api/v1/episodes/{sources['deleted-skill']}"
        )
        self.assertEqual(status, 200)
        status, after_deleted = self.request("POST", "/api/v1/cognition/predict", {
            "text": "amber device maintenance",
            "specialist_id": "deleted-skill",
        })
        self.assertEqual(status, 200)
        self.assertEqual(after_deleted["decision"], "abstain")
        self.assertEqual(after_deleted["reason"], "specialist_unknown")
        self.assertEqual(after_deleted, counterfactual["without_selected_source_prediction"])
        status, after_retained = self.request("POST", "/api/v1/cognition/predict", {
            "text": "violet garden planning",
            "specialist_id": "retained-skill",
        })
        self.assertEqual(status, 200)
        self.assertEqual(after_retained["decision"], "predict")
        self.assertEqual(after_retained["predicted_label"], before_retained["predicted_label"])

    def test_reasoning_counterfactual_withholds_one_source_and_preserves_alternative_proof(self):
        source_data = (
            ("primary relation", "Alice is linked to Bob.", "linked(alice, bob)", None),
            (
                "inference rule",
                "A linked pair is an associate pair.",
                None,
                "linked(?x, ?y) -> associate(?x, ?y)",
            ),
            ("alternate relation", "Alice is also linked to Bob.", "linked(alice, bob)", None),
            (
                "alternate inference rule",
                "A linked pair is also an associate pair.",
                None,
                "linked(?x, ?y) -> associate(?x, ?y)",
            ),
            ("unrelated source", "Charlie owns a lamp.", "owns(charlie, lamp)", None),
            (
                "contradictory source",
                "Alice is not associated with Bob.",
                "not associate(alice, bob)",
                None,
            ),
        )
        source_ids = {}
        observation_ids = {}
        for title, statement, fact_text, rule_text in source_data:
            status, episode = self.request("POST", "/api/v1/episodes", {
                "title": title,
                "source_kind": "user_note",
                "content": statement,
                "consented": True,
            })
            self.assertEqual(status, 201)
            status, observation = self.request("POST", "/api/v1/observations", {
                "episode_id": episode["id"],
                "statement": statement,
            })
            self.assertEqual(status, 201)
            source_ids[title] = episode["id"]
            observation_ids[title] = observation["id"]
            if fact_text:
                status, record = self.request("POST", "/api/v1/reasoning/facts", {
                    "fact": fact_text,
                    "observation_id": observation["id"],
                })
                self.assertEqual(status, 201)
            if rule_text:
                status, record = self.request("POST", "/api/v1/reasoning/rules", {
                    "rule": rule_text,
                    "observation_id": observation["id"],
                })
                self.assertEqual(status, 201)

        status, primary_withheld = self.request(
            "POST", "/api/v1/reasoning/counterfactual", {
                "query": "associate(alice, bob)",
                "episode_id": source_ids["primary relation"],
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(primary_withheld["with_source"]["state"], "conflict")
        self.assertEqual(primary_withheld["without_source"]["state"], "conflict")
        self.assertFalse(primary_withheld["state_changed"])
        self.assertEqual(primary_withheld["withheld_fact_count"], 1)
        self.assertEqual(
            primary_withheld["without_source"]["answers"][0]["proof"]["premises"][0]
            ["proof"]["source_observation_id"],
            observation_ids["alternate relation"],
        )

        status, rule_withheld = self.request(
            "POST", "/api/v1/reasoning/counterfactual", {
                "query": "associate(alice, bob)",
                "episode_id": source_ids["inference rule"],
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(rule_withheld["with_source"]["state"], "conflict")
        self.assertEqual(rule_withheld["without_source"]["state"], "conflict")
        self.assertFalse(rule_withheld["state_changed"])
        self.assertEqual(rule_withheld["withheld_rule_count"], 1)
        self.assertEqual(
            rule_withheld["without_source"]["answers"][0]["proof"][
                "source_observation_id"
            ],
            observation_ids["alternate inference rule"],
        )
        status, audit = self.request("POST", "/api/v1/reasoning/resilience", {
            "query": "associate(alice, bob)",
        })
        self.assertEqual(status, 200)
        self.assertEqual(audit["baseline"]["state"], "conflict")
        self.assertEqual(audit["source_count"], 6)
        self.assertEqual(audit["pivotal_source_episode_ids"], [])
        self.assertEqual(len(audit["redundant_source_episode_ids"]), 6)
        self.assertEqual(
            audit["single_source_resilience"], "robust_to_any_one_source_withholding"
        )
        source_audit = {
            row["episode_id"]: row for row in audit["sources"]
        }
        self.assertEqual(
            source_audit[source_ids["contradictory source"]]["result_without_source"],
            "supported",
        )
        self.assertEqual(
            source_audit[source_ids["unrelated source"]]["result_without_source"],
            "conflict",
        )
        self.assertIn("does not test all combinations", audit["interpretation_boundary"])
        status, failure_sets = self.request(
            "POST", "/api/v1/reasoning/failure-sets", {
                "query": "associate(alice, bob)",
                "evaluation_budget": 30,
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(failure_sets["baseline"]["state"], "conflict")
        self.assertTrue(failure_sets["complete"])
        self.assertEqual(failure_sets["pairs_checked"], 15)
        self.assertEqual(failure_sets["evaluations_performed"], 22)
        self.assertEqual(failure_sets["pairs_total"], 15)
        failure_sets_by_ids = {
            tuple(row["episode_ids"]): row
            for row in failure_sets["minimal_pair_failure_sets"]
        }
        duplicate_facts = tuple(sorted([
            source_ids["primary relation"],
            source_ids["alternate relation"],
        ]))
        duplicate_rules = tuple(sorted([
            source_ids["inference rule"],
            source_ids["alternate inference rule"],
        ]))
        self.assertEqual(set(failure_sets_by_ids), {duplicate_facts, duplicate_rules})
        self.assertTrue(all(
            row["minimal"] for row in failure_sets_by_ids.values()
        ))
        self.assertTrue(all(
            failure_sets_by_ids[pair]["bindings_lost"] == [{}]
            for pair in (duplicate_facts, duplicate_rules)
        ))
        self.assertTrue(all(
            pair["state"] == "refuted" and pair["minimal_failure_set"]
            for pair in failure_sets["pair_evaluations"]
            if tuple(pair["episode_ids"]) in {duplicate_facts, duplicate_rules}
        ))
        self.assertTrue(any(
            set(pair["episode_ids"]) == {
                source_ids["primary relation"],
                source_ids["contradictory source"],
            } and pair["state"] == "supported" and not pair["minimal_failure_set"]
            for pair in failure_sets["pair_evaluations"]
        ))
        self.assertTrue(any(
            set(pair["episode_ids"]) == {
                source_ids["unrelated source"],
                source_ids["primary relation"],
            } and pair["state"] == "conflict"
            for pair in failure_sets["pair_evaluations"]
        ))
        self.assertGreaterEqual(failure_sets["elapsed_ms"], 0)
        surviving_pairs = [
            pair for pair in failure_sets["pair_evaluations"]
            if pair["alternative_answers"]
        ]
        self.assertTrue(surviving_pairs)
        surviving_proof_sources = {
            source_id
            for pair in surviving_pairs
            for answer in pair["alternative_answers"]
            for source_id in answer["proof_source_observation_ids"]
        }
        self.assertIn(observation_ids["alternate relation"], surviving_proof_sources)
        self.assertIn(observation_ids["alternate inference rule"], surviving_proof_sources)
        status, incomplete_audit = self.request(
            "POST", "/api/v1/reasoning/failure-sets", {
                "query": "associate(alice, bob)",
                "evaluation_budget": 1,
            }
        )
        self.assertEqual(status, 200)
        self.assertFalse(incomplete_audit["complete"])
        self.assertEqual(incomplete_audit["stopped_reason"], "evaluation_budget")
        self.assertEqual(incomplete_audit["pairs_checked"], 0)
        status, invalid_budget = self.request(
            "POST", "/api/v1/reasoning/failure-sets", {
                "query": "associate(alice, bob)",
                "evaluation_budget": True,
            }
        )
        self.assertEqual(status, 400)
        status, missing = self.request("POST", "/api/v1/reasoning/counterfactual", {
            "query": "associate(alice, bob)",
            "episode_id": "missing-source",
        })
        self.assertEqual(status, 404)
        status, thought = self.request("POST", "/api/v1/think", {})
        self.assertEqual(status, 200)
        self.assertFalse(thought["executed"])
        status, thought_history = self.request("GET", "/api/v1/think")
        self.assertEqual(status, 200)
        self.assertEqual(thought_history["latest"]["id"], thought["id"])

    def test_language_interpretation_proposes_facts_without_storing_them(self):
        status, result = self.request("POST", "/api/v1/language/interpret", {
            "text": "Alice is not parent of Bob.",
        })
        self.assertEqual(status, 200)
        self.assertEqual(result["proposals"][0]["fact"], "not parent(alice, bob)")
        self.assertTrue(result["requires_human_review"])
        self.assertFalse(result["stored"])

        status, records = self.request("GET", "/api/v1/reasoning")
        self.assertEqual(status, 200)
        self.assertEqual(records["facts"], [])
        self.assertEqual(records["rules"], [])

        status, error = self.request("POST", "/api/v1/language/interpret", {
            "text": "x" * 4_001,
        })
        self.assertEqual(status, 400)
        self.assertIn("4000", error["error"])

    def test_http_review_flow_persists_only_selected_language_proposals(self):
        _, episode = self.request("POST", "/api/v1/episodes", {
            "title": "Language review source",
            "source_kind": "test_fixture",
            "content": "Alice likes Bob and helps Cara.",
            "consented": True,
        })
        _, observation = self.request("POST", "/api/v1/observations", {
            "episode_id": episode["id"],
            "statement": "Alice likes Bob and helps Cara.",
        })
        _, interpretation = self.request("POST", "/api/v1/language/interpret", {
            "text": "Alice likes Bob and helps Cara.",
        })
        status, saved = self.request("POST", "/api/v1/reasoning/facts/reviewed-language", {
            "observation_id": observation["id"],
            "reviewed": True,
            "proposals": [interpretation["proposals"][1]],
        })

        self.assertEqual(status, 201)
        self.assertEqual(saved["count"], 1)
        self.assertEqual(saved["facts"][0]["fact"], "help(alice, cara)")
        self.assertEqual(saved["facts"][0]["observation_id"], observation["id"])
        status, records = self.request("GET", "/api/v1/reasoning")
        self.assertEqual(status, 200)
        self.assertEqual([row["fact"] for row in records["facts"]], ["help(alice, cara)"])

    def test_complete_http_learning_lifecycle(self):
        status, episode = self.request("POST", "/api/v1/episodes", {
            "title": "Test source", "source_kind": "test_fixture",
            "content": "A deliberately explicit test statement.", "consented": True,
        })
        self.assertEqual(status, 201)
        status, observation = self.request("POST", "/api/v1/observations", {
            "episode_id": episode["id"], "statement": "The test statement was explicitly recorded.",
        })
        self.assertEqual(status, 201)
        status, cognition = self.request("POST", "/api/v1/cognition/experiences", {
            "episode_id": episode["id"],
            "observation_id": observation["id"],
            "specialist_id": "project-support",
            "label": True,
            "verification_source": "Explicit user-confirmed outcome",
            "consented": True,
        })
        self.assertEqual(status, 201)
        self.assertEqual(cognition["specialist_id"], "project-support")
        status, cognitive_status = self.request("GET", "/api/v1/cognition/status")
        self.assertEqual(status, 200)
        self.assertEqual(cognitive_status["stored_experiences"], 1)
        self.assertEqual(cognitive_status["neurons"], 10_000)
        self.assertGreater(cognitive_status["synapses"], 0)
        self.assertEqual(cognitive_status["replay_size"], 1)
        status, cognitive_prediction = self.request("POST", "/api/v1/cognition/predict", {
            "text": "The test statement was explicitly recorded.",
            "specialist_id": "project-support",
        })
        self.assertEqual(status, 200)
        self.assertEqual(cognitive_prediction["decision"], "predict")
        status, duplicate = self.request("POST", "/api/v1/cognition/experiences", {
            "episode_id": episode["id"],
            "observation_id": observation["id"],
            "specialist_id": "project-support",
            "label": True,
            "verification_source": "duplicate",
            "consented": True,
        })
        self.assertEqual(status, 409)
        status, source = self.request("GET", f"/api/v1/episodes/{episode['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(source["episode"]["content"], "A deliberately explicit test statement.")
        status, hypothesis = self.request("POST", "/api/v1/hypotheses", {
            "claim": "Test hypothesis", "scope": ["test"],
            "expected_outcome": "Explicit positive test outcome",
            "evidence": [{"observation_id": observation["id"], "stance": "supports"}],
        })
        self.assertEqual(status, 201)
        status, hypotheses = self.request("GET", "/api/v1/hypotheses")
        self.assertEqual(status, 200)
        self.assertEqual(hypotheses["hypotheses"][0]["evidence"][0]["statement"],
                         "The test statement was explicitly recorded.")
        status, prediction = self.request("POST", "/api/v1/predictions", {
            "hypothesis_id": hypothesis["id"], "context": "Test request",
            "probability": 0.8,
        })
        self.assertEqual(status, 201)
        status, outcome = self.request("POST", "/api/v1/outcomes", {
            "prediction_id": prediction["id"], "observed": True,
            "verification_source": "Test fixture",
        })
        self.assertEqual(status, 201)
        status, report = self.request("POST", "/api/v1/evaluations", {})
        self.assertEqual(status, 200)
        self.assertEqual(report["results"][0]["sample_size"], 1)
        self.assertEqual(report["results"][0]["status"], "candidate")
        status, reports = self.request("GET", "/api/v1/evaluations")
        self.assertEqual(status, 200)
        self.assertEqual(len(reports["evaluations"]), 1)
        status, sleep = self.request("POST", "/api/v1/sleep", {})
        self.assertEqual(status, 200)
        self.assertEqual(sleep["evaluated_predictions"], 1)
        self.assertEqual(sleep["hypotheses_evaluated"], 1)
        self.assertEqual(sleep["contradictory_hypotheses"], 0)
        status, sleep_history = self.request("GET", "/api/v1/sleep")
        self.assertEqual(status, 200)
        self.assertEqual(sleep_history["latest"]["id"], sleep["id"])
        status, fact = self.request("POST", "/api/v1/reasoning/facts", {
            "fact": "parent(alice, bob)",
            "observation_id": observation["id"],
        })
        self.assertEqual(status, 201)
        status, rule = self.request("POST", "/api/v1/reasoning/rules", {
            "rule": "parent(?x, ?y) -> ancestor(?x, ?y)",
            "observation_id": observation["id"],
        })
        self.assertEqual(status, 201)
        status, answer = self.request("POST", "/api/v1/reasoning/query", {
            "query": "ancestor(alice, ?person)",
        })
        self.assertEqual(status, 200)
        self.assertEqual(answer["state"], "supported")
        self.assertEqual(answer["answers"][0]["bindings"], {"?person": "bob"})
        status, natural_answer = self.request("POST", "/api/v1/language/query", {
            "question": "Who is an ancestor of Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(natural_answer["answer"], "yes")
        self.assertEqual(natural_answer["query"], "ancestor(?who, bob)")
        self.assertEqual(
            natural_answer["reasoning"]["answers"][0]["bindings"], {"?who": "alice"}
        )
        self.assertEqual(
            natural_answer["reasoning"]["answers"][0]["proof"]["rule_id"], rule["id"]
        )
        status, known_object_fact = self.request("POST", "/api/v1/reasoning/facts", {
            "fact": "help(alice, cara)",
            "observation_id": observation["id"],
        })
        self.assertEqual(status, 201)
        status, object_answer = self.request("POST", "/api/v1/language/query", {
            "question": "Who does Alice help?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(object_answer["answer"], "yes")
        self.assertEqual(object_answer["query"], "help(alice, ?who)")
        self.assertEqual(
            object_answer["reasoning"]["answers"][0]["bindings"], {"?who": "cara"}
        )
        status, why_answer = self.request("POST", "/api/v1/language/query", {
            "question": "Why does Alice help Cara?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(why_answer["answer"], "evidence_found")
        self.assertEqual(
            why_answer["evidence_trace"]["proof"]["kind"], "source_fact"
        )
        self.assertIn("not an explanation", why_answer["explanation_boundary"])
        status, why_unknown = self.request("POST", "/api/v1/language/query", {
            "question": "Why does Alice help Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(why_unknown["answer"], "unknown")
        self.assertIsNone(why_unknown["evidence_trace"])
        status, unknown_answer = self.request("POST", "/api/v1/language/query", {
            "question": "Does Alice help Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(unknown_answer["answer"], "unknown")
        status, negative_help_fact = self.request("POST", "/api/v1/reasoning/facts", {
            "fact": "not help(alice, bob)",
            "observation_id": observation["id"],
        })
        self.assertEqual(status, 201)
        status, why_refuted = self.request("POST", "/api/v1/language/query", {
            "question": "Why does Alice help Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(why_refuted["answer"], "claim_refuted")
        self.assertEqual(
            why_refuted["evidence_trace"]["counterproof"]["literal"],
            "not help(alice, bob)",
        )
        status, why_no = self.request("POST", "/api/v1/language/query", {
            "question": "Why does Alice not help Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(why_no["answer"], "evidence_found")
        self.assertEqual(
            why_no["evidence_trace"]["proof"]["literal"], "not help(alice, bob)"
        )
        status, no_known_helper = self.request("POST", "/api/v1/language/query", {
            "question": "Who helps Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(no_known_helper["answer"], "unknown")
        self.assertEqual(no_known_helper["reasoning"]["state"], "refuted")
        status, positive_helper_fact = self.request("POST", "/api/v1/reasoning/facts", {
            "fact": "help(dana, bob)",
            "observation_id": observation["id"],
        })
        self.assertEqual(status, 201)
        status, known_helper = self.request("POST", "/api/v1/language/query", {
            "question": "Who helps Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(known_helper["answer"], "yes")
        self.assertEqual(known_helper["reasoning"]["state"], "mixed")
        self.assertIn(
            {"?who": "dana"},
            [row["bindings"] for row in known_helper["reasoning"]["answers"]],
        )
        status, unsupported = self.request("POST", "/api/v1/language/query", {
            "question": "Why did Alice help Bob?",
        })
        self.assertEqual(status, 400)
        self.assertIn("Unsupported question", unsupported["error"])
        status, negative_fact = self.request("POST", "/api/v1/reasoning/facts", {
            "fact": "not parent(alice, bob)",
            "observation_id": observation["id"],
        })
        self.assertEqual(status, 201)
        status, conflict_answer = self.request("POST", "/api/v1/language/query", {
            "question": "Is Alice a parent of Bob?",
        })
        self.assertEqual(status, 200)
        self.assertEqual(conflict_answer["answer"], "conflict")
        self.assertIsNotNone(self.server.reasoning_cache._signature)
        self.assertGreater(_token_feature_signature.cache_info().currsize, 0)
        status, deleted = self.request("DELETE", f"/api/v1/episodes/{episode['id']}")
        self.assertEqual(status, 200)
        self.assertIsNone(self.server.reasoning_cache._signature)
        self.assertIsNone(self.server.reasoning_cache._engine)
        self.assertEqual(_token_feature_signature.cache_info().currsize, 0)
        status, sleep_history = self.request("GET", "/api/v1/sleep")
        self.assertEqual(status, 200)
        self.assertIsNone(sleep_history["latest"])
        status, dashboard = self.request("GET", "/api/v1/dashboard")
        self.assertEqual(status, 200)
        self.assertEqual(dashboard["counts"]["episodes"], 0)
        self.assertEqual(dashboard["counts"]["predictions"], 0)
        self.assertEqual(dashboard["cognition"]["stored_experiences"], 0)
        self.assertEqual(dashboard["cognition"]["specialist_count"], 0)


if __name__ == "__main__":
    unittest.main()
