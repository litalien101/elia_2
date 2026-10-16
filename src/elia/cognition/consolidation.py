from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any


MAX_CONSOLIDATION_BATCH_ITEMS = 64
MAX_CONSOLIDATION_TEXT_LENGTH = 2_000
MAX_CONSOLIDATION_QUEUE = 8


@dataclass
class ConsolidationSummary:
    clusters: list[str] = field(default_factory=list)
    retained: list[str] = field(default_factory=list)
    compression_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "clusters": list(self.clusters),
            "retained": list(self.retained),
            "compression_ratio": self.compression_ratio,
            "notes": list(self.notes),
        }


class BackgroundConsolidator:
    """Queues bounded, review-required reports without writing to primary memory."""

    def __init__(self) -> None:
        self.last_summary: ConsolidationSummary | None = None
        self._queue: deque[tuple[list[str], list[str], float, int]] = deque()
        self._next_batch_id = 0
        self.batches_queued = 0
        self.batches_processed = 0
        self.items_processed = 0
        self.queue_rejections = 0

    @staticmethod
    def _validate_items(items: list[str] | None, label: str) -> list[str]:
        if items is None:
            return []
        if not isinstance(items, list) or len(items) > MAX_CONSOLIDATION_BATCH_ITEMS:
            raise ValueError(
                f"{label} must be a list of at most {MAX_CONSOLIDATION_BATCH_ITEMS} items"
            )
        if any(
            not isinstance(item, str) or not item.strip()
            or len(item) > MAX_CONSOLIDATION_TEXT_LENGTH
            for item in items
        ):
            raise ValueError(
                f"{label} items must contain 1 to {MAX_CONSOLIDATION_TEXT_LENGTH} characters"
            )
        return [item.strip() for item in items]

    @staticmethod
    def _validate_threshold(retention_threshold: float) -> float:
        if (
            isinstance(retention_threshold, bool)
            or not isinstance(retention_threshold, (int, float))
            or not math.isfinite(float(retention_threshold))
            or not 0 <= float(retention_threshold) <= 1
        ):
            raise ValueError("retention_threshold must be a finite number between 0 and 1")
        return float(retention_threshold)

    def enqueue(
        self,
        *,
        evidence: list[str] | None = None,
        hypotheses: list[str] | None = None,
        retention_threshold: float = 0.8,
    ) -> dict[str, Any]:
        checked_evidence = self._validate_items(evidence, "evidence")
        checked_hypotheses = self._validate_items(hypotheses, "hypotheses")
        threshold = self._validate_threshold(retention_threshold)
        if len(checked_evidence) + len(checked_hypotheses) > MAX_CONSOLIDATION_BATCH_ITEMS:
            raise ValueError(f"at most {MAX_CONSOLIDATION_BATCH_ITEMS} items can be queued")
        if not checked_evidence and not checked_hypotheses:
            raise ValueError("cannot queue an empty consolidation batch")
        if len(self._queue) >= MAX_CONSOLIDATION_QUEUE:
            self.queue_rejections += 1
            raise ValueError("consolidation queue is full")
        self._next_batch_id += 1
        self._queue.append((
            checked_evidence,
            checked_hypotheses,
            threshold,
            self._next_batch_id,
        ))
        self.batches_queued += 1
        return {
            "batch_id": self._next_batch_id,
            "queued_items": len(checked_evidence) + len(checked_hypotheses),
            "queue_depth": len(self._queue),
        }

    def run_one(self) -> dict[str, Any]:
        if not self._queue:
            raise ValueError("consolidation queue is empty")
        evidence, hypotheses, threshold, batch_id = self._queue.popleft()
        summary = self.compress(
            evidence=evidence,
            hypotheses=hypotheses,
            retention_threshold=threshold,
        )
        self.batches_processed += 1
        self.items_processed += len(evidence) + len(hypotheses)
        return {
            "batch_id": batch_id,
            **summary.summary(),
            "persisted": False,
            "requires_human_review": True,
        }

    def compress(
        self,
        *,
        evidence: list[str] | None = None,
        hypotheses: list[str] | None = None,
        retention_threshold: float = 0.8,
    ) -> ConsolidationSummary:
        evidence_items = self._validate_items(evidence, "evidence")
        hypothesis_items = self._validate_items(hypotheses, "hypotheses")
        retention_threshold = self._validate_threshold(retention_threshold)
        if len(evidence_items) + len(hypothesis_items) > MAX_CONSOLIDATION_BATCH_ITEMS:
            raise ValueError(f"at most {MAX_CONSOLIDATION_BATCH_ITEMS} items can be consolidated")
        items = evidence_items + hypothesis_items
        unique = list(dict.fromkeys(items))
        clusters = [item for item in unique if item]
        retained = unique[: max(1, min(len(unique), int(len(unique) * retention_threshold)))]
        ratio = 1.0 - (len(retained) / max(1, len(unique)))
        summary = ConsolidationSummary(
            clusters=clusters,
            retained=retained,
            compression_ratio=round(ratio, 6),
            notes=[
                "intentional compression preserves explicit evidence identities",
                "latent motifs are summarized, not replaced",
            ],
        )
        self.last_summary = summary
        return summary

    def status(self) -> dict[str, Any]:
        return {
            "queue_depth": len(self._queue),
            "queue_capacity": MAX_CONSOLIDATION_QUEUE,
            "batch_item_capacity": MAX_CONSOLIDATION_BATCH_ITEMS,
            "batches_queued": self.batches_queued,
            "batches_processed": self.batches_processed,
            "items_processed": self.items_processed,
            "queue_rejections": self.queue_rejections,
            "execution_model": "explicit_single_batch_background_scheduler",
            "primary_memory_writes": 0,
        }


__all__ = ["BackgroundConsolidator", "ConsolidationSummary"]
