from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import uuid
import unicodedata
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

from .cognition import CognitiveSystem, TaskPlanner
from .cognition.core import clear_feature_cache
from .cognition.language import MAX_SEGMENTS, interpret_question, interpret_text
from .cognition.reasoning import (
    BoundedRuleEngine,
    Fact,
    Literal,
    ReasoningInputError,
    ReasoningLimitError,
    parse_literal,
    parse_rule,
)
from .cognition.system import StoredExperience
from .voice import LocalVoiceService, VoiceInputError, VoiceUnavailable


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
DEFAULT_DATA_DIR = ROOT / "data"
MAX_BODY_BYTES = 1_000_000
MAX_VOICE_AUDIO_BYTES = 5_000_000
MIN_VALIDATION_CASES = 10
DEFAULT_MEMORY_RETENTION_THRESHOLD = 0.8
MEMORY_COVERAGE_METHOD = "in_sample_normalized_memory_key_coverage_unkeyed_records_lost"
UNBENCHMARKED_MEMORY_METHOD = "legacy_or_unbenchmarked_memory_compression"

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    content TEXT NOT NULL,
    consented INTEGER NOT NULL CHECK (consented = 1),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    evidence_start INTEGER,
    evidence_end INTEGER,
    created_at TEXT NOT NULL,
    CHECK ((evidence_start IS NULL AND evidence_end IS NULL) OR
           (evidence_start >= 0 AND evidence_end >= evidence_start))
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    scope TEXT NOT NULL,
    conditions TEXT NOT NULL,
    expected_outcome TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'validated', 'challenged')),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypothesis_evidence (
    hypothesis_id TEXT NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    stance TEXT NOT NULL CHECK (stance IN ('supports', 'challenges')),
    PRIMARY KEY (hypothesis_id, observation_id)
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    context TEXT NOT NULL,
    probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
    baseline_probability REAL NOT NULL CHECK (baseline_probability >= 0 AND baseline_probability <= 1),
    outcome_definition TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL UNIQUE REFERENCES predictions(id) ON DELETE CASCADE,
    observed INTEGER CHECK (observed IN (0, 1)),
    verification_source TEXT,
    recorded_at TEXT NOT NULL,
    CHECK ((observed IS NULL AND verification_source IS NULL) OR
           (observed IN (0, 1) AND verification_source IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT REFERENCES hypotheses(id) ON DELETE CASCADE,
    sample_size INTEGER NOT NULL,
    predictor_brier REAL,
    baseline_brier REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_episode ON observations(episode_id);
CREATE INDEX IF NOT EXISTS idx_predictions_hypothesis ON predictions(hypothesis_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_context_per_hypothesis
    ON predictions(hypothesis_id, lower(trim(context)));
CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
"""
SLEEP_SCHEMA = """
CREATE TABLE sleep_runs (
    id TEXT PRIMARY KEY,
    evaluated_predictions INTEGER NOT NULL CHECK (evaluated_predictions >= 0),
    hypotheses_evaluated INTEGER NOT NULL CHECK (hypotheses_evaluated >= 0),
    status_changes INTEGER NOT NULL CHECK (status_changes >= 0),
    validated_hypotheses INTEGER NOT NULL CHECK (validated_hypotheses >= 0),
    challenged_hypotheses INTEGER NOT NULL CHECK (challenged_hypotheses >= 0),
    contradictory_hypotheses INTEGER NOT NULL CHECK (contradictory_hypotheses >= 0),
    predictor_brier REAL,
    baseline_brier REAL,
    created_at TEXT NOT NULL
);
"""
THINKING_SCHEMA = """
CREATE TABLE thinking_runs (
    id TEXT PRIMARY KEY,
    recommendation TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""
REASONING_SCHEMA = """
CREATE TABLE IF NOT EXISTS reasoning_facts (
    id TEXT PRIMARY KEY,
    literal TEXT NOT NULL,
    observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    UNIQUE (literal, observation_id)
);
CREATE TABLE IF NOT EXISTS reasoning_rules (
    id TEXT PRIMARY KEY,
    definition TEXT NOT NULL,
    observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reasoning_facts_literal ON reasoning_facts(literal);
CREATE UNIQUE INDEX IF NOT EXISTS idx_reasoning_rules_source_definition
    ON reasoning_rules(definition, observation_id);
"""
SCHEMA_VERSION = 1
COGNITION_SCHEMA = """
CREATE TABLE cognitive_experiences (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    observation_id TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    specialist_id TEXT NOT NULL,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    verification_source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (observation_id, specialist_id)
);
CREATE INDEX idx_cognitive_experiences_specialist
    ON cognitive_experiences(specialist_id, sequence);
"""
COGNITION_METRICS_SCHEMA = """
ALTER TABLE cognitive_experiences
    ADD COLUMN pre_update_probability REAL NOT NULL DEFAULT 0.5
    CHECK (pre_update_probability >= 0 AND pre_update_probability <= 1);
ALTER TABLE cognitive_experiences
    ADD COLUMN surprise REAL NOT NULL DEFAULT 0
    CHECK (surprise >= 0 AND surprise <= 1);
ALTER TABLE cognitive_experiences
    ADD COLUMN replay_updates INTEGER NOT NULL DEFAULT 0
    CHECK (replay_updates >= 0);
"""
SCHEMA_VERSION = 10
MIGRATIONS = {
    1: SCHEMA,
    2: COGNITION_SCHEMA,
    3: COGNITION_METRICS_SCHEMA,
    4: SLEEP_SCHEMA,
    5: THINKING_SCHEMA,
    6: REASONING_SCHEMA,
    7: """
CREATE TABLE IF NOT EXISTS memory_compressions (
    id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    original_units INTEGER NOT NULL CHECK (original_units >= 0),
    retained_units INTEGER NOT NULL CHECK (retained_units >= 0),
    reduction_ratio REAL NOT NULL CHECK (reduction_ratio >= 0 AND reduction_ratio <= 1),
    preserved_claims REAL NOT NULL CHECK (preserved_claims >= 0 AND preserved_claims <= 1),
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_compressions_created_at
    ON memory_compressions(created_at DESC);
""",
    8: "",
}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


@contextmanager
def session(db_path: Path) -> Iterator[sqlite3.Connection]:
    with closing(connect(db_path)) as connection:
        with connection:
            yield connection


def ensure_memory_compression_columns(connection: sqlite3.Connection) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_compressions'"
    ).fetchone()
    if table is None:
        return
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(memory_compressions)").fetchall()
    }
    if "retention_threshold" not in columns:
        connection.execute(
            "ALTER TABLE memory_compressions ADD COLUMN retention_threshold REAL NOT NULL DEFAULT 0.8"
        )
    if "retention_gate_accepted" not in columns:
        connection.execute(
            "ALTER TABLE memory_compressions ADD COLUMN retention_gate_accepted INTEGER NOT NULL DEFAULT 1"
        )
    if "held_out_coverage" not in columns:
        connection.execute(
            "ALTER TABLE memory_compressions ADD COLUMN held_out_coverage REAL NOT NULL DEFAULT 1.0"
        )


def ensure_memory_coverage_method_column(connection: sqlite3.Connection) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_compressions'"
    ).fetchone()
    if table is None:
        return
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(memory_compressions)").fetchall()
    }
    if "coverage_method" not in columns:
        connection.execute(
            "ALTER TABLE memory_compressions ADD COLUMN coverage_method TEXT NOT NULL "
            f"DEFAULT '{UNBENCHMARKED_MEMORY_METHOD}'"
        )
        connection.execute(
            "UPDATE memory_compressions SET summary = "
            "'Legacy compression report; retention coverage method unavailable.'"
        )


def initialize(db_path: Path) -> None:
    with closing(connect(db_path)) as connection:
        current_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {current_version} is newer than supported version {SCHEMA_VERSION}"
            )
        for version in range(current_version + 1, SCHEMA_VERSION + 1):
            if version == 8:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    ensure_memory_compression_columns(connection)
                    connection.execute("PRAGMA user_version = 8")
                    connection.commit()
                except sqlite3.Error:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
                continue
            if version == 9:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    ensure_memory_coverage_method_column(connection)
                    connection.execute("PRAGMA user_version = 9")
                    connection.commit()
                except sqlite3.Error:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
                continue
            if version == 10:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        "UPDATE memory_compressions SET summary = "
                        "'Legacy compression report; retention coverage method unavailable.' "
                        "WHERE coverage_method = ? AND "
                        "(summary LIKE '%gate accepted%' OR summary LIKE '%task coverage%')",
                        (UNBENCHMARKED_MEMORY_METHOD,),
                    )
                    connection.execute("PRAGMA user_version = 10")
                    connection.commit()
                except sqlite3.Error:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
                continue
            script = (
                "BEGIN IMMEDIATE;\n"
                f"{MIGRATIONS[version]}\n"
                f"PRAGMA user_version = {version};\n"
                "COMMIT;"
            )
            try:
                connection.executescript(script)
            except sqlite3.Error:
                if connection.in_transaction:
                    connection.rollback()
                raise


def required_text(payload: dict[str, Any], field: str, max_length: int = 20_000) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, f"'{field}' must be a non-empty string")
    value = value.strip()
    if len(value) > max_length:
        raise ApiError(400, f"'{field}' exceeds {max_length} characters")
    return value


def probability(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ApiError(400, f"'{field}' must be a number between 0 and 1")
    return float(value)


def retention_threshold_value(value: Any) -> float:
    message = "'retention_threshold' must be a number between 0 and 1"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, message)
    if isinstance(value, int):
        if value not in (0, 1):
            raise ApiError(400, message)
        return float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ApiError(400, message)
    return value


def create_episode(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("consented") is not True:
        raise ApiError(400, "Explicit consent is required before storing an episode")
    episode_id = new_id("episode")
    created_at = utc_now()
    connection.execute(
        "INSERT INTO episodes (id, title, source_kind, content, consented, created_at) "
        "VALUES (?, ?, ?, ?, 1, ?)",
        (
            episode_id,
            required_text(payload, "title", 300),
            required_text(payload, "source_kind", 80),
            required_text(payload, "content"),
            created_at,
        ),
    )
    return {"id": episode_id, "created_at": created_at}


def create_observation(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    observation_id = new_id("observation")
    episode_id = required_text(payload, "episode_id", 100)
    statement = required_text(payload, "statement")
    start = payload.get("evidence_start")
    end = payload.get("evidence_end")
    if (start is None) != (end is None):
        raise ApiError(400, "Provide both evidence_start and evidence_end, or neither")
    if start is not None and (
        isinstance(start, bool) or isinstance(end, bool) or
        not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start
    ):
        raise ApiError(400, "Evidence offsets must be non-negative integers with end >= start")
    with connection:
        episode = connection.execute(
            "SELECT content FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if episode is None:
            raise ApiError(404, "Episode not found")
        if end is not None and end > len(episode["content"]):
            raise ApiError(400, "Evidence offsets are outside the episode content")
        created_at = utc_now()
        connection.execute(
            "INSERT INTO observations (id, episode_id, statement, evidence_start, evidence_end, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (observation_id, episode_id, statement, start, end, created_at),
        )
    return {"id": observation_id, "created_at": created_at}


def create_hypothesis(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ApiError(400, "At least one source-backed evidence link is required")
    normalized: list[tuple[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ApiError(400, "Each evidence entry must be an object")
        observation_id = required_text(item, "observation_id", 100)
        stance = item.get("stance")
        if stance not in ("supports", "challenges"):
            raise ApiError(400, "Evidence stance must be 'supports' or 'challenges'")
        normalized.append((observation_id, stance))
    if len({item[0] for item in normalized}) != len(normalized):
        raise ApiError(400, "An observation can only be linked once per hypothesis")
    hypothesis_id = new_id("hypothesis")
    created_at = utc_now()
    scope = payload.get("scope")
    conditions = payload.get("conditions", [])
    if not isinstance(scope, list) or not scope or not all(
        isinstance(item, str) and item.strip() for item in scope
    ):
        raise ApiError(400, "'scope' must be a non-empty list of strings")
    if not isinstance(conditions, list) or not all(isinstance(item, str) for item in conditions):
        raise ApiError(400, "'conditions' must be a list of strings")
    with connection:
        for observation_id, _ in normalized:
            if connection.execute(
                "SELECT 1 FROM observations WHERE id = ?", (observation_id,)
            ).fetchone() is None:
                raise ApiError(404, f"Observation '{observation_id}' not found")
        connection.execute(
            "INSERT INTO hypotheses "
            "(id, claim, scope, conditions, expected_outcome, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'candidate', ?, ?)",
            (
                hypothesis_id,
                required_text(payload, "claim"),
                json.dumps([item.strip() for item in scope]),
                json.dumps([item.strip() for item in conditions]),
                required_text(payload, "expected_outcome", 1000),
                created_at,
                created_at,
            ),
        )
        connection.executemany(
            "INSERT INTO hypothesis_evidence (hypothesis_id, observation_id, stance) VALUES (?, ?, ?)",
            [(hypothesis_id, observation_id, stance) for observation_id, stance in normalized],
        )
    return {"id": hypothesis_id, "status": "candidate", "version": 1, "created_at": created_at}


def create_prediction(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    hypothesis_id = required_text(payload, "hypothesis_id", 100)
    p = probability(payload, "probability")
    prediction_id = new_id("prediction")
    created_at = utc_now()
    with connection:
        hypothesis = connection.execute(
            "SELECT status, expected_outcome FROM hypotheses WHERE id = ?", (hypothesis_id,)
        ).fetchone()
        if hypothesis is None:
            raise ApiError(404, "Hypothesis not found")
        if hypothesis["status"] == "challenged":
            raise ApiError(409, "Challenged hypotheses cannot make new predictions")
        context = required_text(payload, "context")
        outcome_definition = hypothesis["expected_outcome"]
        base = connection.execute(
            "SELECT COUNT(o.observed) AS n, COALESCE(SUM(o.observed), 0) AS yes_count "
            "FROM predictions p JOIN outcomes o ON o.prediction_id = p.id "
            "WHERE p.hypothesis_id = ? AND o.observed IS NOT NULL",
            (hypothesis_id,),
        ).fetchone()
        baseline = (base["yes_count"] + 1) / (base["n"] + 2)
        try:
            connection.execute(
                "INSERT INTO predictions "
                "(id, hypothesis_id, context, probability, baseline_probability, outcome_definition, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    prediction_id,
                    hypothesis_id,
                    context,
                    p,
                    baseline,
                    outcome_definition,
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "idx_prediction_context_per_hypothesis" in str(exc):
                raise ApiError(409, "This hypothesis already has a prediction for that context") from exc
            raise
    return {
        "id": prediction_id,
        "created_at": created_at,
        "probability": p,
        "baseline_probability": baseline,
    }


def record_outcome(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    prediction_id = required_text(payload, "prediction_id", 100)
    observed = payload.get("observed")
    if observed is not True and observed is not False and observed is not None:
        raise ApiError(400, "'observed' must be true, false, or null for unknown")
    verification = payload.get("verification_source")
    if observed is None:
        if verification not in (None, ""):
            raise ApiError(400, "Unknown outcomes cannot have a verification source")
        verification = None
    else:
        if not isinstance(verification, str) or not verification.strip():
            raise ApiError(400, "A verification source is required for a known outcome")
        verification = verification.strip()
    outcome_id = new_id("outcome")
    recorded_at = utc_now()
    with connection:
        if connection.execute(
            "SELECT 1 FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone() is None:
            raise ApiError(404, "Prediction not found")
        try:
            connection.execute(
                "INSERT INTO outcomes (id, prediction_id, observed, verification_source, recorded_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (outcome_id, prediction_id, None if observed is None else int(observed), verification, recorded_at),
            )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: outcomes.prediction_id" in str(exc):
                raise ApiError(409, "This prediction already has an outcome") from exc
            raise
    return {"id": outcome_id, "recorded_at": recorded_at, "scored": observed is not None}


def evaluate(connection: sqlite3.Connection, hypothesis_id: str | None = None) -> dict[str, Any]:
    where = "WHERE p.hypothesis_id = ?" if hypothesis_id else ""
    params: tuple[Any, ...] = (hypothesis_id,) if hypothesis_id else ()
    rows = connection.execute(
        "SELECT p.hypothesis_id, p.probability, p.baseline_probability, o.observed "
        "FROM predictions p JOIN outcomes o ON o.prediction_id = p.id "
        f"{where} {'AND' if where else 'WHERE'} o.observed IS NOT NULL "
        "ORDER BY p.created_at",
        params,
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["hypothesis_id"], []).append(row)
    results: list[dict[str, Any]] = []
    created_at = utc_now()
    with connection:
        for hid, records in grouped.items():
            n = len(records)
            model_score = sum((row["probability"] - row["observed"]) ** 2 for row in records) / n
            baseline_score = sum(
                (row["baseline_probability"] - row["observed"]) ** 2 for row in records
            ) / n
            status = "candidate"
            if n >= MIN_VALIDATION_CASES:
                status = "validated" if model_score < baseline_score else "challenged"
            connection.execute(
                "INSERT INTO evaluation_runs "
                "(id, hypothesis_id, sample_size, predictor_brier, baseline_brier, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (new_id("evaluation"), hid, n, model_score, baseline_score, created_at),
            )
            connection.execute(
                "UPDATE hypotheses SET status = ?, updated_at = ? WHERE id = ?",
                (status, created_at, hid),
            )
            results.append({
                "hypothesis_id": hid,
                "sample_size": n,
                "predictor_brier": round(model_score, 6),
                "baseline_brier": round(baseline_score, 6),
                "status": status,
            })
    return {"created_at": created_at, "minimum_validation_cases": MIN_VALIDATION_CASES, "results": results}


def run_sleep_cycle(connection: sqlite3.Connection) -> dict[str, Any]:
    before = {
        row["id"]: row["status"]
        for row in connection.execute("SELECT id, status FROM hypotheses")
    }
    evaluation = evaluate(connection)
    after = {
        row["id"]: row["status"]
        for row in connection.execute("SELECT id, status FROM hypotheses")
    }
    evaluated_predictions = sum(
        result["sample_size"] for result in evaluation["results"]
    )
    predictor_brier = (
        sum(
            result["predictor_brier"] * result["sample_size"]
            for result in evaluation["results"]
        ) / evaluated_predictions
        if evaluated_predictions
        else None
    )
    baseline_brier = (
        sum(
            result["baseline_brier"] * result["sample_size"]
            for result in evaluation["results"]
        ) / evaluated_predictions
        if evaluated_predictions
        else None
    )
    contradictory_hypotheses = connection.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT hypothesis_id FROM hypothesis_evidence "
        "GROUP BY hypothesis_id "
        "HAVING SUM(CASE WHEN stance = 'supports' THEN 1 ELSE 0 END) > 0 "
        "AND SUM(CASE WHEN stance = 'challenges' THEN 1 ELSE 0 END) > 0"
        ")"
    ).fetchone()[0]
    created_at = evaluation["created_at"]
    report = {
        "id": new_id("sleep"),
        "evaluated_predictions": evaluated_predictions,
        "hypotheses_evaluated": len(evaluation["results"]),
        "status_changes": sum(
            1 for hypothesis_id, status in after.items()
            if before.get(hypothesis_id) != status
        ),
        "validated_hypotheses": sum(
            1 for status in after.values() if status == "validated"
        ),
        "challenged_hypotheses": sum(
            1 for status in after.values() if status == "challenged"
        ),
        "contradictory_hypotheses": contradictory_hypotheses,
        "predictor_brier": round(predictor_brier, 6) if predictor_brier is not None else None,
        "baseline_brier": round(baseline_brier, 6) if baseline_brier is not None else None,
        "created_at": created_at,
        "results": evaluation["results"],
    }
    with connection:
        connection.execute(
            "INSERT INTO sleep_runs "
            "(id, evaluated_predictions, hypotheses_evaluated, status_changes, "
            "validated_hypotheses, challenged_hypotheses, contradictory_hypotheses, "
            "predictor_brier, baseline_brier, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report["id"], report["evaluated_predictions"],
                report["hypotheses_evaluated"], report["status_changes"],
                report["validated_hypotheses"], report["challenged_hypotheses"],
                report["contradictory_hypotheses"], report["predictor_brier"],
                report["baseline_brier"], created_at,
            ),
        )
    return report


def run_deliberation(connection: sqlite3.Connection) -> dict[str, Any]:
    """Select one evidence-grounded next step; never execute the proposed action."""
    conflict = connection.execute(
        "SELECT h.id, h.claim, "
        "SUM(CASE WHEN he.stance = 'supports' THEN 1 ELSE 0 END) AS supports, "
        "SUM(CASE WHEN he.stance = 'challenges' THEN 1 ELSE 0 END) AS challenges "
        "FROM hypotheses h JOIN hypothesis_evidence he ON he.hypothesis_id = h.id "
        "GROUP BY h.id HAVING supports > 0 AND challenges > 0 "
        "ORDER BY challenges DESC, supports DESC, h.created_at LIMIT 1"
    ).fetchone()
    pending = connection.execute(
        "SELECT p.id, p.hypothesis_id, p.context, h.claim "
        "FROM predictions p JOIN hypotheses h ON h.id = p.hypothesis_id "
        "LEFT JOIN outcomes o ON o.prediction_id = p.id "
        "WHERE o.id IS NULL ORDER BY p.created_at LIMIT 1"
    ).fetchone()
    unresolved_candidate = connection.execute(
        "SELECT h.id, h.claim, h.status, COUNT(o.id) AS resolved_predictions, "
        "COUNT(p.id) AS total_predictions "
        "FROM hypotheses h LEFT JOIN predictions p ON p.hypothesis_id = h.id "
        "LEFT JOIN outcomes o ON o.prediction_id = p.id AND o.observed IS NOT NULL "
        "WHERE h.status IN ('candidate', 'validated') "
        "GROUP BY h.id ORDER BY CASE h.status WHEN 'candidate' THEN 0 ELSE 1 END, "
        "resolved_predictions, h.created_at LIMIT 1"
    ).fetchone()
    evidence = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    hypotheses = connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]

    planner = TaskPlanner(min_validation_cases=MIN_VALIDATION_CASES)
    recommendation = planner.plan(
        evidence_count=evidence,
        hypothesis_count=hypotheses,
        conflict=dict(conflict) if conflict is not None else None,
        pending=dict(pending) if pending is not None else None,
        unresolved_candidate=dict(unresolved_candidate) if unresolved_candidate is not None else None,
    )

    created_at = utc_now()
    result = {
        "id": new_id("thought"),
        "created_at": created_at,
        "mode": "propose_only",
        "recommendation": {
            "action": recommendation["action"],
            "target_type": recommendation["target_type"],
            "target_id": recommendation["target_id"],
            "title": recommendation["title"],
            "rationale": recommendation["rationale"],
            "proposed_step": recommendation["proposed_step"],
        },
        "basis": recommendation["basis"],
        "requires_human_approval": recommendation["requires_human_approval"],
        "executed": recommendation["executed"],
        "limitations": recommendation["limitations"],
    }
    with connection:
        connection.execute(
            "INSERT INTO thinking_runs (id, recommendation, created_at) VALUES (?, ?, ?)",
            (result["id"], json.dumps(result), created_at),
        )
    return result


def deliberation_history(connection: sqlite3.Connection) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT recommendation FROM thinking_runs "
        "ORDER BY created_at DESC, rowid DESC LIMIT 20"
    ).fetchall()
    thoughts = [json.loads(row["recommendation"]) for row in rows]
    return {"latest": thoughts[0] if thoughts else None, "thoughts": thoughts}


def create_reasoning_fact(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    literal_text = required_text(payload, "fact", 256)
    observation_id = required_text(payload, "observation_id", 100)
    try:
        literal = parse_literal(literal_text)
    except ReasoningInputError as exc:
        raise ApiError(400, str(exc)) from exc
    if any(term.startswith("?") for term in literal.arguments):
        raise ApiError(400, "Facts must be ground atoms; variables are only allowed in queries and rules")
    if connection.execute(
        "SELECT 1 FROM observations WHERE id = ?", (observation_id,)
    ).fetchone() is None:
        raise ApiError(404, "Source observation not found")
    if connection.execute("SELECT COUNT(*) FROM reasoning_facts").fetchone()[0] >= 2_000:
        raise ApiError(409, "The local reasoning store supports at most 2000 facts")
    canonical = literal.render()
    fact_id = new_id("fact")
    created_at = utc_now()
    try:
        with connection:
            connection.execute(
                "INSERT INTO reasoning_facts (id, literal, observation_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (fact_id, canonical, observation_id, created_at),
            )
    except sqlite3.IntegrityError as exc:
        if "reasoning_facts.literal, reasoning_facts.observation_id" in str(exc):
            raise ApiError(409, "This fact is already linked to that source observation") from exc
        raise
    return {
        "id": fact_id,
        "fact": canonical,
        "observation_id": observation_id,
        "created_at": created_at,
    }


def create_reviewed_language_facts(
    connection: sqlite3.Connection,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("reviewed") is not True:
        raise ApiError(400, "Explicit review confirmation is required")
    observation_id = required_text(payload, "observation_id", 100)
    proposals = payload.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        raise ApiError(400, "Select at least one language proposal to store")
    if len(proposals) > MAX_SEGMENTS * 9:
        raise ApiError(400, "Too many language proposals in one request")
    observation = connection.execute(
        "SELECT statement FROM observations WHERE id = ?", (observation_id,)
    ).fetchone()
    if observation is None:
        raise ApiError(404, "Source observation not found")
    source_text = observation["statement"]
    interpreted = interpret_text(source_text)
    allowable = {
        (row["fact"], row["start"], row["end"], row["text"])
        for row in interpreted["proposals"]
    }
    normalized: list[str] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ApiError(400, "Each reviewed proposal must be an object")
        fact_text = required_text(proposal, "fact", 256)
        start = proposal.get("start")
        end = proposal.get("end")
        source_span = proposal.get("text")
        if (
            isinstance(start, bool) or not isinstance(start, int)
            or isinstance(end, bool) or not isinstance(end, int)
            or not isinstance(source_span, str)
            or start < 0 or end < start or end > len(source_text)
            or source_text[start:end] != source_span
        ):
            raise ApiError(400, "Proposal source span does not match the selected observation")
        if (fact_text, start, end, source_span) not in allowable:
            raise ApiError(400, "Proposal is not supported by this observation's interpretation")
        try:
            literal = parse_literal(fact_text)
        except ReasoningInputError as exc:
            raise ApiError(400, str(exc)) from exc
        if any(term.startswith("?") for term in literal.arguments):
            raise ApiError(400, "Reviewed facts must be ground atoms")
        normalized.append(literal.render())

    if len(set(normalized)) != len(normalized):
        raise ApiError(400, "Duplicate facts cannot be submitted in one review")
    if connection.execute("SELECT COUNT(*) FROM reasoning_facts").fetchone()[0] + len(normalized) > 2_000:
        raise ApiError(409, "The local reasoning store supports at most 2000 facts")
    for literal in normalized:
        if connection.execute(
            "SELECT 1 FROM reasoning_facts WHERE literal = ? AND observation_id = ?",
            (literal, observation_id),
        ).fetchone():
            raise ApiError(409, "A selected fact is already linked to this observation")

    created_at = utc_now()
    rows = [
        {
            "id": new_id("fact"),
            "fact": literal,
            "observation_id": observation_id,
            "created_at": created_at,
        }
        for literal in normalized
    ]
    try:
        with connection:
            connection.executemany(
                "INSERT INTO reasoning_facts (id, literal, observation_id, created_at) "
                "VALUES (:id, :fact, :observation_id, :created_at)",
                rows,
            )
    except sqlite3.IntegrityError as exc:
        if "reasoning_facts.literal, reasoning_facts.observation_id" in str(exc):
            raise ApiError(409, "A selected fact is already linked to this observation") from exc
        raise
    return {"facts": rows, "count": len(rows), "observation_id": observation_id}


def create_reasoning_rule(connection: sqlite3.Connection, payload: dict[str, Any]) -> dict[str, Any]:
    definition = required_text(payload, "rule", 1_000)
    observation_id = required_text(payload, "observation_id", 100)
    rule_id = new_id("rule")
    try:
        rule = parse_rule(definition, rule_id, observation_id)
    except ReasoningInputError as exc:
        raise ApiError(400, str(exc)) from exc
    if connection.execute(
        "SELECT 1 FROM observations WHERE id = ?", (observation_id,)
    ).fetchone() is None:
        raise ApiError(404, "Source observation not found")
    if connection.execute("SELECT COUNT(*) FROM reasoning_rules").fetchone()[0] >= 256:
        raise ApiError(409, "The local reasoning store supports at most 256 rules")
    canonical = " & ".join(item.render() for item in rule.premises) + " -> " + rule.conclusion.render()
    created_at = utc_now()
    try:
        with connection:
            connection.execute(
                "INSERT INTO reasoning_rules (id, definition, observation_id, created_at) "
                "VALUES (?, ?, ?, ?)",
                (rule_id, canonical, observation_id, created_at),
            )
    except sqlite3.IntegrityError as exc:
        if "reasoning_rules.definition, reasoning_rules.observation_id" in str(exc):
            raise ApiError(409, "This rule is already linked to that source observation") from exc
        raise
    return {
        "id": rule_id,
        "rule": canonical,
        "observation_id": observation_id,
        "created_at": created_at,
    }


def reasoning_records(connection: sqlite3.Connection) -> dict[str, Any]:
    facts = connection.execute(
        "SELECT f.id, f.literal AS fact, f.observation_id, o.statement AS source_statement, "
        "f.created_at FROM reasoning_facts f JOIN observations o ON o.id = f.observation_id "
        "ORDER BY f.created_at, f.id LIMIT 2000"
    ).fetchall()
    rules = connection.execute(
        "SELECT r.id, r.definition AS rule, r.observation_id, o.statement AS source_statement, "
        "r.created_at FROM reasoning_rules r JOIN observations o ON o.id = r.observation_id "
        "ORDER BY r.created_at, r.id LIMIT 256"
    ).fetchall()
    return {"facts": [as_dict(row) for row in facts], "rules": [as_dict(row) for row in rules]}


def build_reasoning_engine(connection: sqlite3.Connection) -> BoundedRuleEngine:
    records = reasoning_records(connection)
    facts: list[Fact] = []
    for row in records["facts"]:
        literal = parse_literal(row["fact"])
        facts.append(Fact(row["id"], literal, row["observation_id"]))
    rules = [
        parse_rule(row["rule"], row["id"], row["observation_id"])
        for row in records["rules"]
    ]
    return BoundedRuleEngine(facts, rules)


def build_reasoning_engine_without_episodes(
    connection: sqlite3.Connection,
    episode_ids: set[str],
) -> BoundedRuleEngine:
    if not episode_ids:
        return build_reasoning_engine(connection)
    placeholders = ",".join("?" for _ in episode_ids)
    parameters = tuple(sorted(episode_ids))
    facts = [
        Fact(row["id"], parse_literal(row["fact"]), row["observation_id"])
        for row in connection.execute(
            "SELECT f.id, f.literal AS fact, f.observation_id "
            "FROM reasoning_facts f JOIN observations o ON o.id = f.observation_id "
            f"WHERE o.episode_id NOT IN ({placeholders}) "
            "ORDER BY f.created_at, f.id LIMIT 2000",
            parameters,
        )
    ]
    rules = [
        parse_rule(row["definition"], row["id"], row["observation_id"])
        for row in connection.execute(
            "SELECT r.id, r.definition, r.observation_id "
            "FROM reasoning_rules r JOIN observations o ON o.id = r.observation_id "
            f"WHERE o.episode_id NOT IN ({placeholders}) "
            "ORDER BY r.created_at, r.id LIMIT 256",
            parameters,
        )
    ]
    return BoundedRuleEngine(facts, rules)


def build_reasoning_engine_without_episode(
    connection: sqlite3.Connection,
    episode_id: str,
) -> BoundedRuleEngine:
    return build_reasoning_engine_without_episodes(connection, {episode_id})


class ReasoningCache:
    """Reuse the closed rule set until its exact persisted records change."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._signature: tuple[tuple[tuple[str, str, str], ...], ...] | None = None
        self._engine: BoundedRuleEngine | None = None

    def clear(self) -> None:
        with self._lock:
            self._signature = None
            self._engine = None

    def query(
        self,
        connection: sqlite3.Connection,
        query: Literal,
    ) -> dict[str, object]:
        with self._lock:
            records = reasoning_records(connection)
            signature = (
                tuple((row["id"], row["fact"], row["observation_id"]) for row in records["facts"]),
                tuple((row["id"], row["rule"], row["observation_id"]) for row in records["rules"]),
            )
            if signature != self._signature:
                facts = [
                    Fact(row["id"], parse_literal(row["fact"]), row["observation_id"])
                    for row in records["facts"]
                ]
                rules = [
                    parse_rule(row["rule"], row["id"], row["observation_id"])
                    for row in records["rules"]
                ]
                self._engine = BoundedRuleEngine(facts, rules)
                self._signature = signature
            if self._engine is None:
                raise RuntimeError("Reasoning cache failed to initialize")
            return self._engine.query(query)


def reason_about(
    connection: sqlite3.Connection,
    query_text: str,
    cache: ReasoningCache | None = None,
) -> dict[str, Any]:
    try:
        query = parse_literal(query_text)
        result = (
            cache.query(connection, query)
            if cache is not None
            else build_reasoning_engine(connection).query(query)
        )
    except ReasoningInputError as exc:
        raise ApiError(400, str(exc)) from exc
    except ReasoningLimitError as exc:
        raise ApiError(422, str(exc)) from exc
    return result


def reason_about_without_episode(
    connection: sqlite3.Connection,
    query_text: str,
    episode_id: str,
) -> dict[str, Any]:
    if connection.execute(
        "SELECT 1 FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone() is None:
        raise ApiError(404, "Source episode not found")
    withheld_facts, withheld_rules = connection.execute(
        "SELECT "
        "(SELECT COUNT(*) FROM reasoning_facts f "
        "JOIN observations o ON o.id = f.observation_id WHERE o.episode_id = ?), "
        "(SELECT COUNT(*) FROM reasoning_rules r "
        "JOIN observations o ON o.id = r.observation_id WHERE o.episode_id = ?)",
        (episode_id, episode_id),
    ).fetchone()
    try:
        query = parse_literal(query_text)
        engine = build_reasoning_engine_without_episode(connection, episode_id)
        without_source = engine.query(query)
        with_source = reason_about(connection, query_text)
    except ReasoningInputError as exc:
        raise ApiError(400, str(exc)) from exc
    except ReasoningLimitError as exc:
        raise ApiError(422, str(exc)) from exc
    return {
        "episode_id": episode_id,
        "query": query.render(),
        "with_source": with_source,
        "without_source": without_source,
        "state_changed": with_source["state"] != without_source["state"],
        "withheld_fact_count": withheld_facts,
        "withheld_rule_count": withheld_rules,
        "method": "source_episode_withholding_for_symbolic_closure",
        "interpretation_boundary": (
            "A changed result shows that this derivation depends on records linked "
            "to the selected episode in the current knowledge base; it does not "
            "establish that the source claim is true or causally responsible."
        ),
    }


def audit_reasoning_source_resilience(
    connection: sqlite3.Connection,
    query_text: str,
    max_sources: int = 32,
) -> dict[str, Any]:
    source_rows = connection.execute(
        "SELECT DISTINCT e.id, e.title FROM episodes e "
        "JOIN observations o ON o.episode_id = e.id "
        "LEFT JOIN reasoning_facts f ON f.observation_id = o.id "
        "LEFT JOIN reasoning_rules r ON r.observation_id = o.id "
        "WHERE f.id IS NOT NULL OR r.id IS NOT NULL "
        "ORDER BY e.id LIMIT ?",
        (max_sources + 1,),
    ).fetchall()
    if len(source_rows) > max_sources:
        raise ApiError(
            422,
            f"Resilience audit is limited to {max_sources} source episodes per request",
        )
    try:
        query = parse_literal(query_text)
        baseline = build_reasoning_engine(connection).query(query)
        baseline_supported = {
            json.dumps(row["bindings"], sort_keys=True)
            for row in baseline["answers"]
            if row["state"] in {"supported", "conflict"}
        }
        sources: list[dict[str, Any]] = []
        for source in source_rows:
            counterfactual = build_reasoning_engine_without_episode(
                connection, source["id"]
            ).query(query)
            remaining_supported = {
                json.dumps(row["bindings"], sort_keys=True)
                for row in counterfactual["answers"]
                if row["state"] in {"supported", "conflict"}
            }
            lost_bindings = sorted(baseline_supported - remaining_supported)
            sources.append({
                "episode_id": source["id"],
                "title": source["title"],
                "result_without_source": counterfactual["state"],
                "supported_bindings_lost": [
                    json.loads(binding) for binding in lost_bindings
                ],
                "pivotal_for_any_supported_answer": bool(lost_bindings),
                "withheld_fact_count": connection.execute(
                    "SELECT COUNT(*) FROM reasoning_facts f "
                    "JOIN observations o ON o.id = f.observation_id "
                    "WHERE o.episode_id = ?",
                    (source["id"],),
                ).fetchone()[0],
                "withheld_rule_count": connection.execute(
                    "SELECT COUNT(*) FROM reasoning_rules r "
                    "JOIN observations o ON o.id = r.observation_id "
                    "WHERE o.episode_id = ?",
                    (source["id"],),
                ).fetchone()[0],
            })
    except ReasoningInputError as exc:
        raise ApiError(400, str(exc)) from exc
    except ReasoningLimitError as exc:
        raise ApiError(422, str(exc)) from exc
    pivotal_sources = [
        source["episode_id"] for source in sources
        if source["pivotal_for_any_supported_answer"]
    ]
    return {
        "query": query.render(),
        "baseline": baseline,
        "source_count": len(sources),
        "pivotal_source_episode_ids": pivotal_sources,
        "redundant_source_episode_ids": [
            source["episode_id"] for source in sources
            if not source["pivotal_for_any_supported_answer"]
        ],
        "single_source_resilience": (
            "robust_to_any_one_source_withholding"
            if baseline_supported and not pivotal_sources
            else "depends_on_at_least_one_source"
            if pivotal_sources
            else "no_supported_answers_to_audit"
        ),
        "sources": sources,
        "method": "bounded_leave_one_source_episode_out",
        "limits": {"source_episodes": max_sources},
        "interpretation_boundary": (
            "This is a single-source-at-a-time sensitivity audit. It finds sources "
            "whose removal alone loses a supported binding; it does not test all "
            "combinations, establish source truth, or infer causality."
        ),
    }


def audit_multi_source_failure_sets(
    connection: sqlite3.Connection,
    query_text: str,
    evaluation_budget: int = 529,
    max_total_work: int = 1_000_000,
    max_sources: int = 32,
) -> dict[str, Any]:
    max_pair_count = max_sources * (max_sources - 1) // 2
    max_evaluations = max_pair_count + max_sources + 1
    if (
        isinstance(evaluation_budget, bool)
        or not isinstance(evaluation_budget, int)
        or not 1 <= evaluation_budget <= max_evaluations
    ):
        raise ApiError(
            400,
            f"'evaluation_budget' must be an integer from 1 to {max_evaluations}",
        )
    source_rows = connection.execute(
        "SELECT DISTINCT e.id, e.title FROM episodes e "
        "JOIN observations o ON o.episode_id = e.id "
        "LEFT JOIN reasoning_facts f ON f.observation_id = o.id "
        "LEFT JOIN reasoning_rules r ON r.observation_id = o.id "
        "WHERE f.id IS NOT NULL OR r.id IS NOT NULL "
        "ORDER BY e.id LIMIT ?",
        (max_sources + 1,),
    ).fetchall()
    if len(source_rows) > max_sources:
        raise ApiError(
            422,
            f"Multi-source audit is limited to {max_sources} source episodes",
        )
    source_ids = tuple(row["id"] for row in source_rows)
    source_titles = {row["id"]: row["title"] for row in source_rows}
    started = perf_counter()
    try:
        query = parse_literal(query_text)
        baseline = build_reasoning_engine(connection).query(query)
        work_used = int(baseline["work_used"])
        evaluations = 1
        complete = work_used <= max_total_work
        stopped_reason = None if complete else "reasoning_work_budget"
        binding_keys = {
            json.dumps(row["bindings"], sort_keys=True)
            for row in baseline["answers"]
            if row["state"] in {"supported", "conflict"}
        }
        singleton_states: dict[tuple[str, str], bool] = {}
        singleton_results: list[dict[str, Any]] = []
        if complete:
            for episode_id in source_ids:
                if evaluations >= evaluation_budget:
                    complete = False
                    stopped_reason = "evaluation_budget"
                    break
                result = build_reasoning_engine_without_episode(
                    connection, episode_id
                ).query(query)
                evaluations += 1
                work_used += int(result["work_used"])
                supported = {
                    json.dumps(row["bindings"], sort_keys=True)
                    for row in result["answers"]
                    if row["state"] in {"supported", "conflict"}
                }
                for binding in binding_keys:
                    singleton_states[(episode_id, binding)] = binding in supported
                singleton_results.append({
                    "episode_id": episode_id,
                    "state": result["state"],
                    "supported_bindings_lost": [
                        json.loads(binding)
                        for binding in sorted(binding_keys - supported)
                    ],
                })
                if work_used > max_total_work:
                    complete = False
                    stopped_reason = "reasoning_work_budget"
                    break

        pair_results: list[dict[str, Any]] = []
        minimal_failure_sets: list[dict[str, Any]] = []
        pair_combinations = list(combinations(source_ids, 2))
        pairs_checked = 0
        if complete:
            for left_id, right_id in pair_combinations:
                if evaluations >= evaluation_budget:
                    complete = False
                    stopped_reason = "evaluation_budget"
                    break
                result = build_reasoning_engine_without_episodes(
                    connection, {left_id, right_id}
                ).query(query)
                evaluations += 1
                pairs_checked += 1
                work_used += int(result["work_used"])
                supported = {
                    json.dumps(row["bindings"], sort_keys=True)
                    for row in result["answers"]
                    if row["state"] in {"supported", "conflict"}
                }
                lost = binding_keys - supported
                pair_minimal = {
                    binding for binding in lost
                    if singleton_states.get((left_id, binding), False)
                    and singleton_states.get((right_id, binding), False)
                }
                alternative_answers = []
                for answer in result["answers"]:
                    if answer["state"] not in {"supported", "conflict"}:
                        continue
                    proof = answer.get("proof", {})
                    proof_source_ids: set[str] = set()
                    pending_proofs = [proof]
                    while pending_proofs:
                        current = pending_proofs.pop()
                        if not isinstance(current, dict):
                            continue
                        source_observation_id = current.get("source_observation_id")
                        if isinstance(source_observation_id, str):
                            proof_source_ids.add(source_observation_id)
                        premises = current.get("premises", [])
                        if isinstance(premises, list):
                            pending_proofs.extend(
                                premise.get("proof")
                                for premise in premises
                                if isinstance(premise, dict)
                            )
                    alternative_answers.append({
                        "bindings": answer["bindings"],
                        "state": answer["state"],
                        "proof_source_observation_ids": sorted(proof_source_ids),
                    })
                pair_row = {
                    "episode_ids": [left_id, right_id],
                    "state": result["state"],
                    "supported_bindings_lost": [
                        json.loads(binding) for binding in sorted(lost)
                    ],
                    "minimal_for_bindings": [
                        json.loads(binding) for binding in sorted(pair_minimal)
                    ],
                    "minimal_failure_set": bool(pair_minimal),
                    "alternative_answers": alternative_answers,
                }
                pair_results.append(pair_row)
                if pair_minimal:
                    minimal_failure_sets.append({
                        "episode_ids": [left_id, right_id],
                        "sources": [
                            {"episode_id": source_id, "title": source_titles[source_id]}
                            for source_id in (left_id, right_id)
                        ],
                        "bindings_lost": [
                            json.loads(binding) for binding in sorted(pair_minimal)
                        ],
                        "minimal": True,
                    })
                if work_used > max_total_work:
                    complete = False
                    stopped_reason = "reasoning_work_budget"
                    break
    except ReasoningInputError as exc:
        raise ApiError(400, str(exc)) from exc
    except ReasoningLimitError as exc:
        raise ApiError(422, str(exc)) from exc

    if not complete and stopped_reason is None:
        stopped_reason = "audit_incomplete"
    return {
        "query": query.render(),
        "baseline": baseline,
        "source_count": len(source_ids),
        "single_source_results": singleton_results,
        "pair_evaluations": pair_results,
        "minimal_pair_failure_sets": minimal_failure_sets,
        "evaluations_performed": evaluations,
        "pairs_checked": pairs_checked,
        "pairs_total": len(pair_combinations),
        "evaluation_budget": evaluation_budget,
        "max_evaluation_budget": max_evaluations,
        "reasoning_work_used": work_used,
        "reasoning_work_budget": max_total_work,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "complete": complete and pairs_checked == len(pair_combinations),
        "stopped_reason": stopped_reason,
        "method": "bounded_pairwise_source_withholding",
        "interpretation_boundary": (
            "A minimal pair failure set means each episode alone can be withheld "
            "without losing the specified supported binding, but withholding both "
            "does lose it. Unknown means remaining evidence does not derive the "
            "answer, not that the answer is false. This audit does not test triples "
            "or larger combinations and does not establish source truth or causality."
        ),
    }


def stored_cognitive_experiences(
    connection: sqlite3.Connection,
) -> list[StoredExperience]:
    rows = connection.execute(
        "SELECT ce.id, ce.specialist_id, o.statement AS text, ce.label, ce.sequence "
        "FROM cognitive_experiences ce "
        "JOIN observations o ON o.id = ce.observation_id "
        "ORDER BY sequence"
    ).fetchall()
    return [
        StoredExperience(
            example_id=row["id"],
            specialist_id=row["specialist_id"],
            text=row["text"],
            label=row["label"],
            sequence=row["sequence"],
        )
        for row in rows
    ]


def source_withholding_counterfactual(
    connection: sqlite3.Connection,
    cognitive: CognitiveSystem,
    episode_id: str,
    text: str,
    specialist_id: str | None = None,
) -> dict[str, Any]:
    if connection.execute(
        "SELECT 1 FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone() is None:
        raise ApiError(404, "Source episode not found")
    experiences = stored_cognitive_experiences(connection)
    withheld_ids = {
        row["id"] for row in connection.execute(
            "SELECT id FROM cognitive_experiences WHERE episode_id = ?",
            (episode_id,),
        )
    }
    if not withheld_ids:
        raise ApiError(422, "The selected source episode has no cognitive training examples")

    complete_prediction = cognitive.predict(text, specialist_id)
    counterfactual = CognitiveSystem(
        unit_count=cognitive.core.substrate.unit_count,
        fanout=cognitive.core.substrate.max_fanout,
        replay_capacity_per_specialist=cognitive.replay.capacity_per_specialist,
        replay_per_specialist=cognitive.replay_per_specialist,
        replay_enabled=cognitive.replay_enabled,
        specialists_enabled=cognitive.specialists_enabled,
    )
    counterfactual.router.minimum_similarity = cognitive.router.minimum_similarity
    counterfactual.router.minimum_margin = cognitive.router.minimum_margin
    counterfactual.router.minimum_example_similarity = (
        cognitive.router.minimum_example_similarity
    )
    counterfactual.router.exemplar_capacity = cognitive.router.exemplar_capacity
    retained_experiences = [
        experience for experience in experiences
        if experience.example_id not in withheld_ids
    ]
    counterfactual.rebuild(retained_experiences)
    without_source_prediction = counterfactual.predict(text, specialist_id)
    before_probability = complete_prediction.get("probability")
    after_probability = without_source_prediction.get("probability")
    probability_delta = (
        round(float(after_probability) - float(before_probability), 6)
        if isinstance(before_probability, (int, float))
        and isinstance(after_probability, (int, float))
        else None
    )
    changed = any(
        complete_prediction.get(key) != without_source_prediction.get(key)
        for key in ("decision", "specialist_id", "predicted_label")
    ) or probability_delta not in (None, 0.0)
    return {
        "episode_id": episode_id,
        "withheld_experience_count": len(withheld_ids),
        "retained_experience_count": len(retained_experiences),
        "complete_memory_prediction": complete_prediction,
        "without_selected_source_prediction": without_source_prediction,
        "probability_delta_without_source": probability_delta,
        "prediction_changed": changed,
        "method": "rebuild_from_all_retained_experiences_except_selected_episode",
        "interpretation_boundary": (
            "This is a model sensitivity test under source withholding, not proof that "
            "the selected episode caused the prediction or that its information was "
            "perfectly removed from every external copy."
        ),
    }


def cognitive_status(
    connection: sqlite3.Connection,
    cognitive: CognitiveSystem,
) -> dict[str, Any]:
    return {
        **cognitive.status(),
        "stored_experiences": connection.execute(
            "SELECT COUNT(*) FROM cognitive_experiences"
        ).fetchone()[0],
        "evidence_source": "consented_episodes_and_linked_observations",
    }


def create_cognitive_experience(
    connection: sqlite3.Connection,
    cognitive: CognitiveSystem,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("consented") is not True:
        raise ApiError(400, "Explicit consent is required for cognitive training")
    episode_id = required_text(payload, "episode_id", 100)
    observation_id = required_text(payload, "observation_id", 100)
    specialist_id = required_text(payload, "specialist_id", 80).casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", specialist_id):
        raise ApiError(400, "'specialist_id' may contain lowercase letters, digits, dot, underscore, and hyphen")
    label = payload.get("label")
    if label is True:
        label = 1
    elif label is False:
        label = 0
    elif isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
        raise ApiError(400, "'label' must be true/false or the integer 0/1")
    verification_source = required_text(payload, "verification_source", 1000)
    row = connection.execute(
        "SELECT e.id AS episode_id, o.id AS observation_id, o.statement "
        "FROM episodes e JOIN observations o ON o.episode_id = e.id "
        "WHERE e.id = ? AND o.id = ? AND e.consented = 1",
        (episode_id, observation_id),
    ).fetchone()
    if row is None:
        raise ApiError(404, "Consented episode and its linked observation were not found")
    if connection.execute(
        "SELECT 1 FROM cognitive_experiences WHERE observation_id = ? AND specialist_id = ?",
        (observation_id, specialist_id),
    ).fetchone():
        raise ApiError(409, "This observation has already trained that specialist")

    experience_id = new_id("cognitive")
    created_at = utc_now()
    with connection:
        try:
            connection.execute(
                "INSERT INTO cognitive_experiences "
                "(id, episode_id, observation_id, specialist_id, label, verification_source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    experience_id,
                    episode_id,
                    observation_id,
                    specialist_id,
                    label,
                    verification_source,
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed: cognitive_experiences.observation_id, cognitive_experiences.specialist_id" in str(exc):
                raise ApiError(409, "This observation has already trained that specialist") from exc
            raise
    try:
        learned = cognitive.observe(
            experience_id,
            specialist_id,
            row["statement"],
            label,
        )
        with connection:
            connection.execute(
                "UPDATE cognitive_experiences SET pre_update_probability = ?, surprise = ?, "
                "replay_updates = ? WHERE id = ?",
                (
                    learned["pre_update_probability"],
                    learned["surprise"],
                    learned["replay_updates"],
                    experience_id,
                ),
            )
    except ValueError as exc:
        with connection:
            connection.execute("DELETE FROM cognitive_experiences WHERE id = ?", (experience_id,))
        cognitive.rebuild(stored_cognitive_experiences(connection))
        raise ApiError(400, str(exc)) from exc
    return {
        **learned,
        "created_at": created_at,
        "episode_id": episode_id,
        "observation_id": observation_id,
        "verification_source": verification_source,
    }


def delete_episode(connection: sqlite3.Connection, episode_id: str) -> None:
    with connection:
        existing = connection.execute("SELECT 1 FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if existing is None:
            raise ApiError(404, "Episode not found")
        affected = connection.execute(
            "SELECT DISTINCT he.hypothesis_id FROM hypothesis_evidence he "
            "JOIN observations o ON o.id = he.observation_id WHERE o.episode_id = ?",
            (episode_id,),
        ).fetchall()
        connection.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        connection.execute("DELETE FROM sleep_runs")
        connection.execute("DELETE FROM thinking_runs")
        connection.execute("DELETE FROM memory_compressions")
        for row in affected:
            connection.execute("DELETE FROM hypotheses WHERE id = ?", (row["hypothesis_id"],))


def hypothesis_records(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT h.id, h.claim, h.scope, h.conditions, h.expected_outcome, h.status, h.version, "
        "h.created_at, h.updated_at, "
        "SUM(CASE WHEN he.stance = 'supports' THEN 1 ELSE 0 END) AS support_count, "
        "SUM(CASE WHEN he.stance = 'challenges' THEN 1 ELSE 0 END) AS challenge_count, "
        "(SELECT e.sample_size FROM evaluation_runs e WHERE e.hypothesis_id = h.id "
        "ORDER BY e.created_at DESC LIMIT 1) AS last_sample_size, "
        "(SELECT e.predictor_brier FROM evaluation_runs e WHERE e.hypothesis_id = h.id "
        "ORDER BY e.created_at DESC LIMIT 1) AS last_predictor_brier, "
        "(SELECT e.baseline_brier FROM evaluation_runs e WHERE e.hypothesis_id = h.id "
        "ORDER BY e.created_at DESC LIMIT 1) AS last_baseline_brier "
        "FROM hypotheses h LEFT JOIN hypothesis_evidence he ON he.hypothesis_id = h.id "
        "GROUP BY h.id ORDER BY h.updated_at DESC LIMIT 100"
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            **as_dict(row),
            "scope": json.loads(row["scope"]),
            "conditions": json.loads(row["conditions"]),
        }
        record["evidence"] = [
            as_dict(evidence) for evidence in connection.execute(
                "SELECT o.id AS observation_id, o.statement, o.created_at, e.id AS episode_id, "
                "e.title AS episode_title, he.stance "
                "FROM hypothesis_evidence he "
                "JOIN observations o ON o.id = he.observation_id "
                "JOIN episodes e ON e.id = o.episode_id "
                "WHERE he.hypothesis_id = ? ORDER BY o.created_at",
                (row["id"],),
            )
        ]
        records.append(record)
    return records


def as_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def dashboard_data(connection: sqlite3.Connection) -> dict[str, Any]:
    counts = {
        "episodes": connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
        "observations": connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0],
        "candidate_hypotheses": connection.execute(
            "SELECT COUNT(*) FROM hypotheses WHERE status = 'candidate'"
        ).fetchone()[0],
        "validated_traces": connection.execute(
            "SELECT COUNT(*) FROM hypotheses WHERE status = 'validated'"
        ).fetchone()[0],
        "challenged_hypotheses": connection.execute(
            "SELECT COUNT(*) FROM hypotheses WHERE status = 'challenged'"
        ).fetchone()[0],
        "predictions": connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
        "evaluated_outcomes": connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE observed IS NOT NULL"
        ).fetchone()[0],
        "unknown_outcomes": connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE observed IS NULL"
        ).fetchone()[0],
        "cognitive_experiences": connection.execute(
            "SELECT COUNT(*) FROM cognitive_experiences"
        ).fetchone()[0],
        "memory_compressions": connection.execute(
            "SELECT COUNT(*) FROM memory_compressions"
        ).fetchone()[0],
    }
    latest_memory = connection.execute(
        "SELECT id, strategy, original_units, retained_units, reduction_ratio, preserved_claims, "
        "retention_threshold, retention_gate_accepted, held_out_coverage, coverage_method, summary, created_at "
        "FROM memory_compressions ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    latest_memory_data = as_dict(latest_memory) if latest_memory is not None else None
    if latest_memory_data is not None:
        latest_memory_data["task_coverage"] = (
            latest_memory_data["held_out_coverage"]
            if latest_memory_data["coverage_method"] == MEMORY_COVERAGE_METHOD
            else None
        )
    activity = connection.execute(
        "SELECT 'episode' AS kind, id, title AS label, created_at FROM episodes "
        "UNION ALL SELECT 'observation', id, statement, created_at FROM observations "
        "UNION ALL SELECT 'hypothesis', id, claim, created_at FROM hypotheses "
        "UNION ALL SELECT 'prediction', id, context, created_at FROM predictions "
        "UNION ALL SELECT 'evaluation', id, 'Evaluation run', created_at FROM evaluation_runs "
        "UNION ALL SELECT 'sleep_cycle', id, 'Self-evaluation cycle', created_at FROM sleep_runs "
        "UNION ALL SELECT 'thought', id, 'ELIA proposed a next step', created_at FROM thinking_runs "
        "UNION ALL SELECT 'cognitive_learning', id, specialist_id, created_at FROM cognitive_experiences "
        "ORDER BY created_at DESC LIMIT 20"
    ).fetchall()
    return {
        "counts": counts,
        "activity": [as_dict(row) for row in activity],
        "hypotheses": hypothesis_records(connection)[:20],
        "latest_memory_compression": latest_memory_data,
    }


def _memory_token_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE).strip()


def compress_memory(
    connection: sqlite3.Connection,
    strategy: str = "evidence_preserving",
    max_fragments: int = 64,
) -> dict[str, Any]:
    if not isinstance(strategy, str) or not strategy.strip():
        raise ApiError(400, "'strategy' must be a non-empty string")
    if isinstance(max_fragments, bool) or not isinstance(max_fragments, int) or max_fragments < 1:
        raise ApiError(400, "'max_fragments' must be a positive integer")

    observation_rows = connection.execute(
        "SELECT o.id, o.statement, o.episode_id, e.title AS episode_title FROM observations o "
        "JOIN episodes e ON e.id = o.episode_id ORDER BY o.created_at DESC"
    ).fetchall()
    hypothesis_rows = connection.execute(
        "SELECT h.id, h.claim, h.created_at, NULL AS episode_id, NULL AS episode_title "
        "FROM hypotheses h ORDER BY h.created_at DESC"
    ).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    for row in observation_rows:
        key = _memory_token_key(row["statement"])
        if not key:
            continue
        bucket = grouped.setdefault(key, {
            "key": key,
            "records": [],
            "source_ids": set(),
        })
        bucket["records"].append({
            "kind": "observation",
            "id": row["id"],
            "text": row["statement"],
            "episode_id": row["episode_id"],
            "episode_title": row["episode_title"],
        })
        bucket["source_ids"].add(row["episode_id"])

    for row in hypothesis_rows:
        key = _memory_token_key(row["claim"])
        if not key:
            continue
        bucket = grouped.setdefault(key, {
            "key": key,
            "records": [],
            "source_ids": set(),
        })
        bucket["records"].append({
            "kind": "hypothesis",
            "id": row["id"],
            "text": row["claim"],
            "episode_id": None,
            "episode_title": None,
        })
        bucket["source_ids"].add(row["id"])

    fragments = []
    for key, bucket in grouped.items():
        records = sorted(bucket["records"], key=lambda item: (item["text"].lower(), item["id"]))
        representative = records[0]["text"]
        fragments.append({
            "memory_key": key,
            "record_count": len(records),
            "source_count": len(bucket["source_ids"]),
            "kind": "mixed" if {record["kind"] for record in records} != {"observation"} else "observation",
            "representative_text": representative,
            "summary": (
                f"Merged {len(records)} memory record(s) from {len(bucket['source_ids'])} source(s): "
                f"{representative[:200]}"
            ),
        })

    fragments.sort(key=lambda item: (-item["record_count"], -item["source_count"], item["memory_key"]))
    selected = fragments[: max_fragments]
    original_units = len(observation_rows) + len(hypothesis_rows)
    retained_units = len(selected)
    preserved_claims = 0.0 if original_units == 0 else retained_units / original_units
    reduction_ratio = 1.0 - preserved_claims
    report = {
        "id": new_id("memory"),
        "strategy": strategy,
        "original_units": original_units,
        "retained_units": retained_units,
        "reduction_ratio": round(reduction_ratio, 6),
        "preserved_claims": round(preserved_claims, 6),
        "coverage_method": UNBENCHMARKED_MEMORY_METHOD,
        "summary": (
            f"{strategy} reduced {original_units} raw memory units into {retained_units} canonical fragments "
            f"({reduction_ratio:.1%} reduction)."
        ),
        "fragments": selected,
        "created_at": utc_now(),
    }
    with connection:
        connection.execute(
            "INSERT INTO memory_compressions "
            "(id, strategy, original_units, retained_units, reduction_ratio, preserved_claims, "
            "coverage_method, summary, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                report["id"],
                report["strategy"],
                report["original_units"],
                report["retained_units"],
                report["reduction_ratio"],
                report["preserved_claims"],
                report["coverage_method"],
                report["summary"],
                report["created_at"],
            ),
        )
    return report


def benchmark_memory_compression(
    connection: sqlite3.Connection,
    strategy: str = "evidence_preserving",
    max_fragments: int = 64,
    retention_threshold: float | None = None,
) -> dict[str, Any]:
    if retention_threshold is None:
        threshold = DEFAULT_MEMORY_RETENTION_THRESHOLD
    else:
        threshold = retention_threshold_value(retention_threshold)

    report = compress_memory(connection, strategy=strategy, max_fragments=max_fragments)
    raw_rows = connection.execute(
        "SELECT statement AS text, 'observation' AS kind FROM observations "
        "UNION ALL SELECT claim AS text, 'hypothesis' AS kind FROM hypotheses"
    ).fetchall()
    if not raw_rows:
        benchmark = {
            "coverage": 1.0,
            "duplicate_ratio": 0.0,
            "preserved_tokens": 0.0,
            "total_tokens": 0,
            "task_preservation": 1.0,
            "retention_threshold": round(threshold, 6),
            "unique_retained_memory_keys": 0,
            "retention_gate": {
                "accepted": True,
                "minimum_task_coverage": round(threshold, 6),
                "reason": "There are no memory units to compare; the empty case is considered preserved.",
                "task_preservation": 1.0,
            },
            "strategy": strategy,
            "memory_key_coverage": {
                "evaluated_keys": 0,
                "unkeyed_memory_units": 0,
                "evaluated_units": 0,
                "coverage": 1.0,
                "method": MEMORY_COVERAGE_METHOD,
            },
        }
        result = {**report, "benchmark": benchmark, "task_preservation": benchmark["task_preservation"]}
        with connection:
            connection.execute(
                "UPDATE memory_compressions SET preserved_claims = ?, summary = ?, reduction_ratio = ?, "
                "retention_threshold = ?, retention_gate_accepted = ?, held_out_coverage = ?, "
                "coverage_method = ? WHERE id = ?",
                (
                    benchmark["task_preservation"],
                    f"{strategy} retained {benchmark['task_preservation']:.1%} task coverage after compression; gate accepted.",
                    report["reduction_ratio"],
                    threshold,
                    1,
                    benchmark["memory_key_coverage"]["coverage"],
                    benchmark["memory_key_coverage"]["method"],
                    report["id"],
                ),
            )
        result["preserved_claims"] = benchmark["task_preservation"]
        result["summary"] = (
            f"{strategy} retained {benchmark['task_preservation']:.1%} task coverage after compression; "
            "gate accepted."
        )
        result["retention_gate"] = benchmark["retention_gate"]
        result["retention_threshold"] = round(threshold, 6)
        result["task_coverage"] = round(benchmark["memory_key_coverage"]["coverage"], 6)
        result["coverage_method"] = benchmark["memory_key_coverage"]["method"]
        result["held_out_coverage"] = result["task_coverage"]
        return result

    groups: dict[str, list[str]] = {}
    unkeyed_memory_units = 0
    for row in raw_rows:
        key = _memory_token_key(row["text"])
        if not key:
            unkeyed_memory_units += 1
            continue
        groups.setdefault(key, []).append(row["text"])

    retained_weights: dict[str, int] = {}
    for fragment in report["fragments"]:
        key = _memory_token_key(fragment["representative_text"])
        if not key:
            continue
        retained_weights[key] = max(retained_weights.get(key, 0), int(fragment["record_count"]))

    cluster_scores = []
    for key, values in groups.items():
        retained_count = retained_weights.get(key, 0)
        total_count = len(values)
        if total_count <= 0:
            continue
        cluster_scores.append(min(1.0, retained_count / total_count))

    preserved = sum(cluster_scores)
    evaluated_units = len(cluster_scores) + unkeyed_memory_units
    task_preservation = 1.0 if evaluated_units == 0 else preserved / evaluated_units
    unique_keys = set(groups)
    unique_retained = {key for key in unique_keys if retained_weights.get(key, 0) > 0}
    unique_ratio = 1.0 if not unique_keys else len(unique_retained) / len(unique_keys)
    duplicate_ratio = max(0.0, 1.0 - unique_ratio)
    accepted = task_preservation >= threshold and duplicate_ratio >= 0.0
    memory_key_coverage = {
        "evaluated_keys": len(cluster_scores),
        "unkeyed_memory_units": unkeyed_memory_units,
        "evaluated_units": evaluated_units,
        "coverage": round(task_preservation, 6),
        "method": MEMORY_COVERAGE_METHOD,
    }
    reason = (
        "The compacted memory retains enough task-relevant evidence to pass the retention gate."
        if accepted
        else "The compacted memory drops too much task-relevant coverage to be accepted."
    )
    benchmark = {
        "coverage": round(task_preservation, 6),
        "duplicate_ratio": round(duplicate_ratio, 6),
        "preserved_tokens": round(preserved, 6),
        "total_tokens": evaluated_units,
        "task_preservation": round(task_preservation, 6),
        "retention_threshold": round(threshold, 6),
        "unique_retained_memory_keys": len(unique_retained),
        "retention_gate": {
            "accepted": accepted,
            "minimum_task_coverage": round(threshold, 6),
            "reason": reason,
            "task_preservation": round(task_preservation, 6),
        },
        "strategy": strategy,
        "memory_key_coverage": memory_key_coverage,
    }
    result = {**report, "benchmark": benchmark, "task_preservation": benchmark["task_preservation"]}
    with connection:
        connection.execute(
            "UPDATE memory_compressions SET preserved_claims = ?, summary = ?, reduction_ratio = ?, "
            "retention_threshold = ?, retention_gate_accepted = ?, held_out_coverage = ?, "
            "coverage_method = ? WHERE id = ?",
            (
                benchmark["task_preservation"],
                f"{strategy} retained {benchmark['task_preservation']:.1%} task coverage after compression; gate {'accepted' if accepted else 'failed'}.",
                report["reduction_ratio"],
                threshold,
                int(accepted),
                benchmark["memory_key_coverage"]["coverage"],
                benchmark["memory_key_coverage"]["method"],
                report["id"],
            ),
        )
    result["preserved_claims"] = benchmark["task_preservation"]
    result["summary"] = f"{strategy} retained {benchmark['task_preservation']:.1%} task coverage after compression; gate {'accepted' if accepted else 'failed'}."
    result["retention_gate"] = benchmark["retention_gate"]
    result["retention_threshold"] = round(threshold, 6)
    result["task_coverage"] = benchmark["memory_key_coverage"]["coverage"]
    result["coverage_method"] = benchmark["memory_key_coverage"]["method"]
    result["held_out_coverage"] = result["task_coverage"]
    return result


class EliaHandler(BaseHTTPRequestHandler):
    server_version = "ELIA/0.1"

    @property
    def db_path(self) -> Path:
        return self.server.db_path  # type: ignore[attr-defined]

    @property
    def cognitive(self) -> CognitiveSystem:
        return self.server.cognitive  # type: ignore[attr-defined]

    @property
    def voice(self) -> LocalVoiceService:
        return self.server.voice  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[ELIA] {self.address_string()} - {fmt % args}")

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError(400, "Invalid Content-Length") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ApiError(400, f"Request body must be between 1 and {MAX_BODY_BYTES} bytes")
        if "application/json" not in self.headers.get("Content-Type", ""):
            raise ApiError(415, "Content-Type must be application/json")
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError(400, "Request body must contain valid JSON") from exc
        if not isinstance(payload, dict):
            raise ApiError(400, "Request JSON must be an object")
        return payload

    def _audio_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError(400, "Invalid Content-Length") from exc
        if length <= 0 or length > MAX_VOICE_AUDIO_BYTES:
            raise ApiError(
                400,
                f"Audio body must be between 1 and {MAX_VOICE_AUDIO_BYTES} bytes",
            )
        if self.headers.get_content_type() != "audio/wav":
            raise ApiError(415, "Content-Type must be audio/wav")
        audio = self.rfile.read(length)
        if len(audio) != length:
            raise ApiError(400, "Audio body ended before Content-Length bytes were read")
        return audio

    def _audio(self, audio: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(audio)

    def _serve_static(self) -> None:
        page = WEB_ROOT / "index.html"
        body = page.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        print(f"DBG path={path!r} command={self.command!r} content_type={self.headers.get('Content-Type')!r}", flush=True)
        if path == "/" and self.command == "GET":
            self._serve_static()
            return
        if not path.startswith("/api/v1/"):
            raise ApiError(404, "Route not found")

        if path == "/api/v1/voice/status" and self.command == "GET":
            self._json(200, self.voice.status())
            return
        if path == "/api/v1/voice/transcribe" and self.command == "POST":
            try:
                self._json(200, self.voice.transcribe_wav(self._audio_body()))
            except VoiceInputError as exc:
                raise ApiError(400, str(exc)) from exc
            except VoiceUnavailable as exc:
                raise ApiError(503, str(exc)) from exc
            return
        if path == "/api/v1/voice/synthesize" and self.command == "POST":
            payload = self._body()
            try:
                self._audio(self.voice.synthesize(payload.get("text")))
            except VoiceInputError as exc:
                raise ApiError(400, str(exc)) from exc
            except VoiceUnavailable as exc:
                raise ApiError(503, str(exc)) from exc
            return

        with session(self.db_path) as connection:
            if path == "/api/v1/health" and self.command == "GET":
                self._json(200, {"status": "ok", "cpu_only": True})
                return
            if path == "/api/v1/dashboard" and self.command == "GET":
                self._json(200, {
                    **dashboard_data(connection),
                    "cognition": cognitive_status(connection, self.cognitive),
                    "functional": self.cognitive.functional_status(),
                })
                return
            if path == "/api/v1/memory" and self.command == "GET":
                rows = connection.execute(
                    "SELECT id, strategy, original_units, retained_units, reduction_ratio, preserved_claims, "
                    "coverage_method, summary, created_at "
                    "FROM memory_compressions ORDER BY created_at DESC LIMIT 20"
                ).fetchall()
                self._json(200, {"compressions": [as_dict(row) for row in rows]})
                return
            if path == "/api/v1/memory/compress" and self.command == "POST":
                payload = self._body()
                strategy = payload.get("strategy", "evidence_preserving")
                max_fragments = payload.get("max_fragments", 64)
                if (
                    isinstance(max_fragments, bool)
                    or not isinstance(max_fragments, int)
                    or max_fragments < 1
                ):
                    raise ApiError(400, "'max_fragments' must be a positive integer")
                self._json(200, compress_memory(connection, strategy, max_fragments))
                return
            if path == "/api/v1/memory/benchmark" and self.command == "POST":
                payload = self._body()
                strategy = payload.get("strategy", "evidence_preserving")
                max_fragments = payload.get("max_fragments", 64)
                retention_threshold = payload.get("retention_threshold", DEFAULT_MEMORY_RETENTION_THRESHOLD)
                if (
                    isinstance(max_fragments, bool)
                    or not isinstance(max_fragments, int)
                    or max_fragments < 1
                ):
                    raise ApiError(400, "'max_fragments' must be a positive integer")
                retention_threshold = retention_threshold_value(retention_threshold)
                self._json(
                    200,
                    benchmark_memory_compression(
                        connection,
                        strategy,
                        max_fragments,
                        retention_threshold=retention_threshold,
                    ),
                )
                return
            if path == "/api/v1/cognition/status" and self.command == "GET":
                self._json(200, cognitive_status(connection, self.cognitive))
                return
            if path == "/api/v1/functional/status" and self.command == "GET":
                self._json(200, self.cognitive.functional_status())
                return
            if path == "/api/v1/functional/consciousness" and self.command == "GET":
                self._json(200, self.cognitive.functional_controller())
                return
            if path == "/api/v1/functional/focus" and self.command == "GET":
                self._json(200, self.cognitive.functional_focus())
                return
            if path == "/api/v1/functional/update" and self.command == "POST":
                payload = self._body()
                for field in ("evidence", "hypotheses", "questions"):
                    if field in payload and (
                        not isinstance(payload[field], list)
                        or not all(isinstance(item, str) and item.strip() for item in payload[field])
                    ):
                        raise ApiError(400, f"'{field}' must be a list of non-empty strings")
                for field in ("priorities", "confidence", "uncertainty"):
                    if field in payload:
                        value = payload[field]
                        if not isinstance(value, dict):
                            raise ApiError(400, f"'{field}' must be an object mapping strings to numbers")
                        for key, item in value.items():
                            if not isinstance(key, str) or not key.strip():
                                raise ApiError(400, f"'{field}' keys must be non-empty strings")
                            if isinstance(item, bool) or not isinstance(item, (int, float)):
                                raise ApiError(400, f"'{field}' values must be finite numbers")
                            numeric = float(item)
                            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                                raise ApiError(400, f"'{field}' values must be finite numbers between 0 and 1")
                for field in ("conflict", "pending"):
                    if field in payload and not isinstance(payload[field], dict):
                        raise ApiError(400, f"'{field}' must be an object when supplied")
                if "task" in payload and (not isinstance(payload["task"], str) or not payload["task"].strip()):
                    raise ApiError(400, "'task' must be a non-empty string")
                for field in ("decision", "trace"):
                    if field in payload and (not isinstance(payload[field], str) or not payload[field].strip()):
                        raise ApiError(400, f"'{field}' must be a non-empty string")
                result = self.cognitive.update_functional_state(
                    task=str(payload.get("task", "idle")),
                    evidence=payload.get("evidence"),
                    hypotheses=payload.get("hypotheses"),
                    questions=payload.get("questions"),
                    priorities=payload.get("priorities"),
                    confidence=payload.get("confidence"),
                    decision=payload.get("decision"),
                    trace=payload.get("trace"),
                    uncertainty=payload.get("uncertainty"),
                    conflict=payload.get("conflict"),
                    pending=payload.get("pending"),
                )
                self._json(200, result)
                return
            if path == "/api/v1/functional/consolidation/enqueue" and self.command == "POST":
                payload = self._body()
                if payload:
                    raise ApiError(400, "Consolidation enqueue does not accept request fields")
                try:
                    result = self.cognitive.enqueue_consolidation()
                except ValueError as exc:
                    raise ApiError(409, str(exc)) from exc
                self._json(202, result)
                return
            if path == "/api/v1/functional/consolidation/run" and self.command == "POST":
                payload = self._body()
                if payload:
                    raise ApiError(400, "Consolidation run does not accept request fields")
                try:
                    report = self.cognitive.run_consolidation_once()
                except ValueError as exc:
                    raise ApiError(409, str(exc)) from exc
                self._json(200, report)
                return
            if path == "/api/v1/think" and self.command == "GET":
                self._json(200, deliberation_history(connection))
                return
            if path == "/api/v1/think" and self.command == "POST":
                self._json(200, run_deliberation(connection))
                return
            if path == "/api/v1/language/interpret" and self.command == "POST":
                payload = self._body()
                text = required_text(payload, "text", 4_000)
                try:
                    interpretation = interpret_text(text)
                except ReasoningInputError as exc:
                    raise ApiError(400, str(exc)) from exc
                self._json(200, interpretation)
                return
            if path == "/api/v1/language/query" and self.command == "POST":
                payload = self._body()
                question = required_text(payload, "question", 4_000)
                try:
                    interpretation = interpret_question(question)
                except ReasoningInputError as exc:
                    raise ApiError(400, str(exc)) from exc
                reasoning = reason_about(
                    connection, interpretation["query"], self.server.reasoning_cache
                )
                if interpretation["template"].startswith("wh_"):
                    row_states = [row["state"] for row in reasoning["answers"]]
                    answer = "yes" if "supported" in row_states else (
                        "conflict" if "conflict" in row_states else "unknown"
                    )
                else:
                    answer = {
                        "supported": "yes",
                        "refuted": "no",
                        "unknown": "unknown",
                        "conflict": "conflict",
                        "mixed": "mixed",
                    }[reasoning["state"]]
                is_why_question = interpretation["template"].startswith("why_")
                if is_why_question:
                    answer = {
                        "supported": "evidence_found",
                        "refuted": "claim_refuted",
                        "unknown": "unknown",
                        "conflict": "conflict",
                        "mixed": "mixed",
                    }[reasoning["state"]]
                result: dict[str, Any] = {
                    **interpretation,
                    "answer": answer,
                    "reasoning": reasoning,
                    "evidence_scope": "stored_source_linked_facts_and_rules",
                }
                if is_why_question:
                    explanation = reasoning["answers"][0] if reasoning["answers"] else None
                    result["evidence_trace"] = (
                        {
                            "proof": explanation.get("proof"),
                            "counterproof": explanation.get("counterproof"),
                        }
                        if explanation is not None
                        else None
                    )
                    result["explanation_boundary"] = (
                        "This is a stored evidence derivation, not an explanation of "
                        "real-world cause or intent."
                    )
                self._json(200, result)
                return
            if path == "/api/v1/reasoning" and self.command == "GET":
                self._json(200, reasoning_records(connection))
                return
            if path == "/api/v1/reasoning/facts" and self.command == "POST":
                self._json(201, create_reasoning_fact(connection, self._body()))
                return
            if path == "/api/v1/reasoning/facts/reviewed-language" and self.command == "POST":
                self._json(201, create_reviewed_language_facts(connection, self._body()))
                return
            if path == "/api/v1/reasoning/rules" and self.command == "POST":
                self._json(201, create_reasoning_rule(connection, self._body()))
                return
            if path == "/api/v1/reasoning/query" and self.command == "POST":
                payload = self._body()
                query_text = required_text(payload, "query", 256)
                self._json(200, reason_about(connection, query_text, self.server.reasoning_cache))
                return
            if path == "/api/v1/reasoning/counterfactual" and self.command == "POST":
                payload = self._body()
                query_text = required_text(payload, "query", 256)
                episode_id = required_text(payload, "episode_id", 100)
                self._json(200, reason_about_without_episode(
                    connection, query_text, episode_id
                ))
                return
            if path == "/api/v1/reasoning/resilience" and self.command == "POST":
                payload = self._body()
                query_text = required_text(payload, "query", 256)
                if not connection.in_transaction:
                    connection.execute("BEGIN")
                self._json(200, audit_reasoning_source_resilience(
                    connection, query_text
                ))
                return
            if path == "/api/v1/reasoning/failure-sets" and self.command == "POST":
                payload = self._body()
                query_text = required_text(payload, "query", 256)
                evaluation_budget = payload.get("evaluation_budget", 529)
                if not connection.in_transaction:
                    connection.execute("BEGIN")
                self._json(200, audit_multi_source_failure_sets(
                    connection, query_text, evaluation_budget
                ))
                return
            if path == "/api/v1/cognition/experiences" and self.command == "GET":
                self._json(200, {"experiences": [
                    as_dict(row) for row in connection.execute(
                        "SELECT id, episode_id, observation_id, specialist_id, label, "
                        "pre_update_probability, surprise, replay_updates, verification_source, created_at "
                        "FROM cognitive_experiences ORDER BY sequence DESC LIMIT 100"
                    )
                ]})
                return
            if path == "/api/v1/cognition/predict" and self.command == "POST":
                payload = self._body()
                text = required_text(payload, "text", 16_000)
                specialist_id = payload.get("specialist_id")
                if specialist_id is not None and (
                    not isinstance(specialist_id, str) or not specialist_id.strip()
                ):
                    raise ApiError(400, "'specialist_id' must be a non-empty string when supplied")
                try:
                    result = self.cognitive.predict(text, specialist_id)
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
                self._json(200, result)
                return
            if path == "/api/v1/cognition/counterfactual" and self.command == "POST":
                payload = self._body()
                episode_id = required_text(payload, "episode_id", 100)
                text = required_text(payload, "text", 16_000)
                specialist_id = payload.get("specialist_id")
                if specialist_id is not None and (
                    not isinstance(specialist_id, str) or not specialist_id.strip()
                ):
                    raise ApiError(400, "'specialist_id' must be a non-empty string when supplied")
                try:
                    result = source_withholding_counterfactual(
                        connection,
                        self.cognitive,
                        episode_id,
                        text,
                        specialist_id.strip() if specialist_id else None,
                    )
                except ValueError as exc:
                    raise ApiError(400, str(exc)) from exc
                self._json(200, result)
                return
            if path == "/api/v1/cognition/experiences" and self.command == "POST":
                self._json(201, create_cognitive_experience(
                    connection,
                    self.cognitive,
                    self._body(),
                ))
                return
            if path == "/api/v1/episodes" and self.command == "GET":
                self._json(200, {"episodes": [
                    as_dict(row) for row in connection.execute(
                        "SELECT id, title, source_kind, created_at FROM episodes ORDER BY created_at DESC LIMIT 100"
                    )
                ]})
                return
            if path == "/api/v1/episodes" and self.command == "POST":
                self._json(201, create_episode(connection, self._body()))
                return
            if path == "/api/v1/observations" and self.command == "GET":
                self._json(200, {"observations": [
                    as_dict(row) for row in connection.execute(
                        "SELECT o.id, o.episode_id, e.title AS episode_title, o.statement, "
                        "o.evidence_start, o.evidence_end, o.created_at "
                        "FROM observations o JOIN episodes e ON e.id = o.episode_id "
                        "ORDER BY o.created_at DESC LIMIT 100"
                    )
                ]})
                return
            if path.startswith("/api/v1/episodes/") and self.command == "GET":
                episode_id = path.rsplit("/", 1)[-1]
                episode = connection.execute(
                    "SELECT id, title, source_kind, content, created_at "
                    "FROM episodes WHERE id = ?", (episode_id,)
                ).fetchone()
                if episode is None:
                    raise ApiError(404, "Episode not found")
                self._json(200, {"episode": as_dict(episode)})
                return
            if path.startswith("/api/v1/episodes/") and self.command == "DELETE":
                episode_id = path.rsplit("/", 1)[-1]
                delete_episode(connection, episode_id)
                self.cognitive.rebuild(stored_cognitive_experiences(connection))
                self.server.reasoning_cache.clear()
                clear_feature_cache()
                self._json(200, {"deleted": True, "episode_id": episode_id})
                return
            if path == "/api/v1/observations" and self.command == "POST":
                self._json(201, create_observation(connection, self._body()))
                return
            if path == "/api/v1/hypotheses" and self.command == "GET":
                self._json(200, {"hypotheses": hypothesis_records(connection)})
                return
            if path == "/api/v1/hypotheses" and self.command == "POST":
                self._json(201, create_hypothesis(connection, self._body()))
                return
            if path == "/api/v1/predictions" and self.command == "POST":
                self._json(201, create_prediction(connection, self._body()))
                return
            if path == "/api/v1/predictions" and self.command == "GET":
                self._json(200, {"predictions": [
                    as_dict(row) for row in connection.execute(
                        "SELECT p.id, p.hypothesis_id, h.claim, p.context, p.probability, "
                        "p.baseline_probability, p.outcome_definition, p.created_at, "
                        "o.observed, o.verification_source, o.recorded_at "
                        "FROM predictions p JOIN hypotheses h ON h.id = p.hypothesis_id "
                        "LEFT JOIN outcomes o ON o.prediction_id = p.id "
                        "ORDER BY p.created_at DESC LIMIT 100"
                    )
                ]})
                return
            if path == "/api/v1/outcomes" and self.command == "POST":
                self._json(201, record_outcome(connection, self._body()))
                return
            if path == "/api/v1/evaluations" and self.command == "POST":
                payload = self._body()
                hypothesis_id = payload.get("hypothesis_id")
                if hypothesis_id is not None and (
                    not isinstance(hypothesis_id, str) or not hypothesis_id.strip()
                ):
                    raise ApiError(400, "'hypothesis_id' must be a non-empty string when supplied")
                if hypothesis_id is not None and connection.execute(
                    "SELECT 1 FROM hypotheses WHERE id = ?", (hypothesis_id,)
                ).fetchone() is None:
                    raise ApiError(404, "Hypothesis not found")
                self._json(200, evaluate(connection, hypothesis_id))
                return
            if path == "/api/v1/evaluations" and self.command == "GET":
                self._json(200, {"evaluations": [
                    as_dict(row) for row in connection.execute(
                        "SELECT e.id, e.hypothesis_id, h.claim, e.sample_size, "
                        "e.predictor_brier, e.baseline_brier, e.created_at "
                        "FROM evaluation_runs e LEFT JOIN hypotheses h ON h.id = e.hypothesis_id "
                        "ORDER BY e.created_at DESC LIMIT 100"
                    )
                ]})
                return
            if path == "/api/v1/sleep" and self.command == "GET":
                runs = [
                    as_dict(row) for row in connection.execute(
                        "SELECT id, evaluated_predictions, hypotheses_evaluated, status_changes, "
                        "validated_hypotheses, challenged_hypotheses, contradictory_hypotheses, "
                        "predictor_brier, baseline_brier, created_at "
                        "FROM sleep_runs ORDER BY created_at DESC LIMIT 20"
                    )
                ]
                self._json(200, {"latest": runs[0] if runs else None, "runs": runs})
                return
            if path == "/api/v1/sleep" and self.command == "POST":
                self._json(200, run_sleep_cycle(connection))
                return
        raise ApiError(405 if path.startswith("/api/v1/") else 404, "Method not allowed")

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        try:
            self._handle()
        except ApiError as exc:
            self._json(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            return
        except sqlite3.Error as exc:
            print(f"[ELIA] Database request failed: {type(exc).__name__}: {exc}")
            self._json(500, {"error": "Database operation failed"})
        except OSError as exc:
            print(f"[ELIA] I/O request failed: {type(exc).__name__}: {exc}")
            self._json(500, {"error": "I/O operation failed"})


class EliaServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], db_path: Path):
        super().__init__(address, EliaHandler)
        self.db_path = db_path
        self.cognitive = CognitiveSystem()
        self.reasoning_cache = ReasoningCache()
        self.voice = LocalVoiceService()


def make_server(db_path: Path | None = None, port: int = 8765) -> EliaServer:
    actual_path = db_path or Path(os.environ.get("ELIA_DATA_DIR", DEFAULT_DATA_DIR)) / "elia.sqlite3"
    initialize(actual_path)
    server = EliaServer(("127.0.0.1", port), actual_path)
    with session(actual_path) as connection:
        server.cognitive.rebuild(stored_cognitive_experiences(connection))
    return server


def main() -> None:
    port = int(os.environ.get("ELIA_PORT", "8765"))
    server = make_server(port=port)
    print(f"ELIA is listening locally at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ELIA.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
