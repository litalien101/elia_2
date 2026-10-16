from __future__ import annotations

import hashlib
import math
import os
import platform
import re
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache

from .substrate import SparseNeuronSubstrate


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?|[.!?;,:]", re.UNICODE)
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
NEGATION_MARKERS = {"no", "neither", "never", "nor", "not"}
NEGATION_BOUNDARIES = {
    "!", ",", ".", ":", ";", "?", "although", "but", "however", "instead",
    "rather", "yet",
}
NEGATION_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "i", "in", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "ours", "she", "that", "the", "their", "theirs", "them",
    "they", "this", "those", "to", "us", "was", "we", "were", "you",
    "your", "yours",
}
_RELATION_VERB_FORMS = {
    "help": "help",
    "helps": "help",
    "know": "know",
    "knows": "know",
    "like": "like",
    "likes": "like",
    "lead": "lead",
    "leads": "lead",
    "need": "need",
    "needs": "need",
    "own": "own",
    "owns": "own",
    "prefer": "prefer",
    "prefers": "prefer",
    "support": "support",
    "supports": "support",
    "trust": "trust",
    "trusts": "trust",
    "use": "use",
    "uses": "use",
}


def stable_unit(feature: str, unit_count: int) -> int:
    digest = hashlib.blake2b(
        feature.encode("utf-8"), digest_size=8, person=b"elia-core-v2"
    ).digest()
    return int.from_bytes(digest, "little") % unit_count


def detect_hardware_profile() -> dict[str, object]:
    """Report the visible CPU surface that ELIA can exploit without leaving local-first bounds."""
    cpu_count = max(1, os.cpu_count() or 1)
    machine = platform.machine().lower() or "unknown"
    processor = platform.processor().lower() or "unknown"
    allows_avx2 = "avx2" in processor or "x86_64" in machine or "amd64" in machine
    return {
        "cpu_count": cpu_count,
        "machine": machine,
        "processor": processor,
        "supports_avx2": allows_avx2,
        "worker_budget": min(cpu_count, 8),
        "feature_budget_multiplier": 1.0 + min(max(cpu_count - 4, 0), 8) * 0.08,
    }


class SparseLogisticReadout:
    """An online CPU-only binary readout over sparse shared features."""

    def __init__(self, learning_rate: float = 0.04, l2: float = 0.0005) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights: dict[int, float] = {}
        self.bias = 0.0
        self.updates = 0

    @staticmethod
    def sigmoid(logit: float) -> float:
        if logit >= 0:
            z = math.exp(-min(logit, 40.0))
            return 1.0 / (1.0 + z)
        z = math.exp(max(logit, -40.0))
        return z / (1.0 + z)

    def logit(self, features: Mapping[int, float]) -> float:
        return self.bias + sum(self.weights.get(unit, 0.0) * value for unit, value in features.items())

    def probability(self, features: Mapping[int, float]) -> float:
        return self.sigmoid(self.logit(features))

    def update(
        self,
        features: Mapping[int, float],
        target: int,
        learning_rate_scale: float = 1.0,
    ) -> None:
        error = target - self.probability(features)
        rate = self.learning_rate * learning_rate_scale
        for unit, value in features.items():
            old = self.weights.get(unit, 0.0)
            updated = old + rate * (error * value - self.l2 * old)
            if abs(updated) < 1e-8:
                self.weights.pop(unit, None)
            else:
                self.weights[unit] = updated
        self.bias += rate * error
        self.updates += 1


@lru_cache(maxsize=8192)
def _token_feature_signature(text: str) -> tuple[tuple[str, int], ...]:
    tokens = [token.casefold() for token in TOKEN_PATTERN.findall(text)][:256]
    if not tokens:
        return ()
    words = [token for token in tokens if WORD_PATTERN.fullmatch(token)]
    raw = Counter(f"word:{token}" for token in words)
    raw.update(
        f"pair:{left}:{right}"
        for left, right in zip(tokens, tokens[1:])
        if WORD_PATTERN.fullmatch(left) and WORD_PATTERN.fullmatch(right)
    )
    raw.update(
        f"skip:{tokens[index]}:{tokens[index + 2]}"
        for index in range(len(tokens) - 2)
        if WORD_PATTERN.fullmatch(tokens[index])
        and WORD_PATTERN.fullmatch(tokens[index + 2])
        and tokens[index + 1] not in NEGATION_BOUNDARIES
        and WORD_PATTERN.fullmatch(tokens[index + 1])
    )

    for index, token in enumerate(tokens[:-1]):
        lemma = _RELATION_VERB_FORMS.get(token.casefold())
        if lemma is None:
            continue
        subject = tokens[index - 1] if index > 0 and WORD_PATTERN.fullmatch(tokens[index - 1]) else None
        obj = tokens[index + 1] if WORD_PATTERN.fullmatch(tokens[index + 1]) else None
        if subject is not None and obj is not None:
            raw[f"rel:{lemma}:subject:{subject}"] += 1
            raw[f"rel:{lemma}:object:{obj}"] += 1
            raw[f"rel:{lemma}"] += 1

    for index, token in enumerate(tokens):
        if token in {"is", "are", "was", "were"} and index + 1 < len(tokens):
            quality = tokens[index + 1]
            if WORD_PATTERN.fullmatch(quality):
                raw[f"qual:is_{quality}"] += 1
                if index > 0 and WORD_PATTERN.fullmatch(tokens[index - 1]):
                    raw[f"qual:subject:{tokens[index - 1]}:{quality}"] += 1

    negation_remaining = 0
    for index, token in enumerate(tokens):
        if token in NEGATION_BOUNDARIES:
            negation_remaining = 0
            continue
        if token in NEGATION_MARKERS or token.replace("’", "'").endswith("n't"):
            negation_remaining = (
                0 if token == "not" and tokens[index + 1:index + 2] == ["only"] else 3
            )
            continue
        if not WORD_PATTERN.fullmatch(token):
            continue
        if negation_remaining:
            if token not in NEGATION_FUNCTION_WORDS:
                raw[f"negated:{token}"] += 1
                negation_remaining -= 1

    for token in dict.fromkeys(words[:64]):
        compact = f"^{token}$"
        raw.update(f"char:{compact[i:i + 3]}" for i in range(max(1, len(compact) - 2)))
    return tuple(sorted(raw.items()))


def clear_feature_cache() -> None:
    _token_feature_signature.cache_clear()


class SharedCognitiveCore:
    """Shared deterministic language features, sparse activations, and readout."""

    def __init__(
        self,
        unit_count: int = 10_000,
        max_fanout: int = 16,
        max_features: int = 64,
        hardware_profile: Mapping[str, object] | None = None,
    ) -> None:
        self.hardware_profile = dict(detect_hardware_profile() if hardware_profile is None else hardware_profile)
        self.substrate = SparseNeuronSubstrate(unit_count, max_fanout)
        self.readout = SparseLogisticReadout()
        self.max_features = max_features
        self._feature_limit = max(
            max_features,
            min(96, int(max_features * self.hardware_profile.get("feature_budget_multiplier", 1.0))),
        )
        self._token_cache: dict[str, dict[int, float]] = {}
        self._relation_cache: dict[str, dict[int, float]] = {}

    def relation_features(self, text: str) -> dict[int, float]:
        cached = self._relation_cache.get(text)
        if cached is not None:
            return dict(cached)
        signature = _token_feature_signature(text)
        relation_candidates = [
            feature for feature, _ in signature if feature.startswith("rel:") or feature.startswith("qual:")
        ]
        if not relation_candidates:
            self._relation_cache[text] = {}
            return {}
        features: dict[int, float] = {}
        for feature in relation_candidates:
            unit = stable_unit(feature, self.substrate.unit_count)
            features[unit] = features.get(unit, 0.0) + 1.0
        norm = math.sqrt(sum(value * value for value in features.values())) or 1.0
        normalized = {unit: value / norm for unit, value in features.items()}
        self._relation_cache[text] = dict(normalized)
        return dict(normalized)

    def token_features(self, text: str) -> dict[int, float]:
        cached = self._token_cache.get(text)
        if cached is not None:
            return dict(cached)
        signature = _token_feature_signature(text)
        if not signature:
            self._token_cache[text] = {}
            return {}
        if len(signature) <= self._feature_limit:
            selected = signature
        else:
            selected = sorted(
                signature,
                key=lambda item: (
                    {"negated": 0, "word": 1, "pair": 2, "skip": 3, "char": 4, "rel": 1, "qual": 1}.get(
                        item[0].split(":", 1)[0], 5
                    ),
                    -item[1],
                    item[0],
                ),
            )[: self.max_features]
        features: dict[int, float] = {}
        for feature, count in selected:
            unit = stable_unit(feature, self.substrate.unit_count)
            value = min(1.0, math.log1p(count))
            if feature.startswith("negated:"):
                value *= 2.5
            if feature.startswith("rel:") or feature.startswith("qual:"):
                value *= 1.5
            features[unit] = features.get(unit, 0.0) + value
        norm = math.sqrt(sum(value * value for value in features.values())) or 1.0
        normalized = {unit: value / norm for unit, value in features.items()}
        self._token_cache[text] = dict(normalized)
        return dict(normalized)

    def combined_features(self, text: str) -> dict[int, float]:
        token = self.token_features(text)
        relation = self.relation_features(text)
        if not token and not relation:
            return {}
        merged: dict[int, float] = {}
        for unit, value in token.items():
            merged[unit] = merged.get(unit, 0.0) + value
        for unit, value in relation.items():
            merged[unit] = merged.get(unit, 0.0) + value * 1.8
        norm = math.sqrt(sum(value * value for value in merged.values())) or 1.0
        return {unit: value / norm for unit, value in merged.items()}

    def encode(self, text: str) -> dict[int, float]:
        direct = self.combined_features(text)
        return self.substrate.activate(direct)

    def learn_associations(self, features: Mapping[int, float], surprise: float) -> int:
        strongest = sorted(features, key=features.__getitem__, reverse=True)[:24]
        return self.substrate.learn_associations(strongest, surprise)

    def reset(self) -> None:
        self.substrate.clear()
        self._token_cache.clear()
        self._relation_cache.clear()
        self.readout = SparseLogisticReadout(
            learning_rate=self.readout.learning_rate,
            l2=self.readout.l2,
        )
