from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Mapping


MAX_SIMILARITY_CACHE_ENTRIES = 8_192
MAX_ROUTE_CACHE_ENTRIES = 8_192


class SpecialistRouter:
    """Routes by profile similarity and abstains outside retained examples."""

    def __init__(
        self,
        minimum_similarity: float = 0.2,
        minimum_margin: float = 0.025,
        minimum_example_similarity: float = 0.45,
        exemplar_capacity: int = 128,
    ) -> None:
        if not 0 <= minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be between 0 and 1")
        if not 0 <= minimum_margin <= 1:
            raise ValueError("minimum_margin must be between 0 and 1")
        if not 0 <= minimum_example_similarity <= 1:
            raise ValueError("minimum_example_similarity must be between 0 and 1")
        if exemplar_capacity < 1:
            raise ValueError("exemplar_capacity must be positive")
        self.minimum_similarity = minimum_similarity
        self.minimum_margin = minimum_margin
        self.minimum_example_similarity = minimum_example_similarity
        self.exemplar_capacity = exemplar_capacity
        self._profiles: dict[str, dict[int, float]] = defaultdict(dict)
        self._example_counts: dict[str, int] = defaultdict(int)
        self._exemplars: dict[str, list[dict[int, float]]] = defaultdict(list)
        self._norm_cache: OrderedDict[tuple[tuple[int, float], ...], float] = OrderedDict()
        self._similarity_cache: OrderedDict[
            tuple[tuple[tuple[int, float], ...], tuple[tuple[int, float], ...]],
            float,
        ] = OrderedDict()
        self._route_cache: OrderedDict[tuple[tuple[int, float], ...], dict[str, object]] = OrderedDict()

    @property
    def specialist_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def _feature_signature(self, features: Mapping[int, float]) -> tuple[tuple[int, float], ...]:
        return tuple(sorted(features.items()))

    def _norm(self, features: Mapping[int, float]) -> float:
        signature = self._feature_signature(features)
        cached = self._norm_cache.get(signature)
        if cached is not None:
            self._norm_cache.move_to_end(signature)
            return cached
        norm = math.sqrt(sum(value * value for _, value in signature))
        self._norm_cache[signature] = norm
        self._norm_cache.move_to_end(signature)
        if len(self._norm_cache) > MAX_SIMILARITY_CACHE_ENTRIES:
            self._norm_cache.popitem(last=False)
        return norm

    def _similarity(self, features: Mapping[int, float], profile: Mapping[int, float]) -> float:
        left_items = self._feature_signature(features)
        right_items = self._feature_signature(profile)
        cache_key = (left_items, right_items)
        cached = self._similarity_cache.get(cache_key)
        if cached is not None:
            self._similarity_cache.move_to_end(cache_key)
            return cached
        left_norm = self._norm(features)
        right_norm = self._norm(profile)
        if not left_items or not right_items or left_norm == 0.0 or right_norm == 0.0:
            score = 0.0
        else:
            dot = sum(value * profile.get(unit, 0.0) for unit, value in features.items())
            score = dot / (left_norm * right_norm)
        self._similarity_cache[cache_key] = score
        self._similarity_cache.move_to_end(cache_key)
        if len(self._similarity_cache) > MAX_SIMILARITY_CACHE_ENTRIES:
            self._similarity_cache.popitem(last=False)
        return score

    def learn(self, specialist_id: str, features: Mapping[int, float]) -> None:
        profile = self._profiles[specialist_id]
        for unit, value in features.items():
            profile[unit] = profile.get(unit, 0.0) + value
        self._example_counts[specialist_id] += 1
        exemplars = self._exemplars[specialist_id]
        exemplars.append(dict(features))
        if len(exemplars) > self.exemplar_capacity:
            del exemplars[:len(exemplars) - self.exemplar_capacity]
        self._norm_cache.clear()
        self._similarity_cache.clear()

    def route(self, features: Mapping[int, float]) -> dict[str, object]:
        signature = self._feature_signature(features)
        cached = self._route_cache.get(signature)
        if cached is not None:
            self._route_cache.move_to_end(signature)
            return cached

        scores = sorted(
            (
                (specialist_id, self._similarity(features, profile))
                for specialist_id, profile in self._profiles.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not scores:
            result = {
                "specialist_id": None,
                "mode": "abstain",
                "reason": "no_specialists",
                "top_similarity": 0.0,
                "margin": 0.0,
                "top_example_similarity": 0.0,
                "scores": [],
            }
            self._route_cache[signature] = result
            self._route_cache.move_to_end(signature)
            if len(self._route_cache) > MAX_ROUTE_CACHE_ENTRIES:
                self._route_cache.popitem(last=False)
            return result
        top_id, top_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else 0.0
        margin = top_score - second_score
        example_similarity = max(
            (
                self._similarity(features, exemplar)
                for exemplar in self._exemplars[top_id]
            ),
            default=0.0,
        )
        reason = None
        if top_score < self.minimum_similarity:
            reason = "low_similarity"
        elif margin < self.minimum_margin:
            reason = "ambiguous"
        elif example_similarity < self.minimum_example_similarity:
            reason = "outside_retained_examples"
        if reason:
            result = {
                "specialist_id": None,
                "mode": "abstain",
                "reason": reason,
                "top_similarity": round(top_score, 6),
                "margin": round(margin, 6),
                "top_example_similarity": round(example_similarity, 6),
                "minimum_example_similarity": self.minimum_example_similarity,
                "scores": [
                    {"specialist_id": specialist_id, "similarity": round(score, 6)}
                    for specialist_id, score in scores[:5]
                ],
            }
            self._route_cache[signature] = result
            self._route_cache.move_to_end(signature)
            if len(self._route_cache) > MAX_ROUTE_CACHE_ENTRIES:
                self._route_cache.popitem(last=False)
            return result
        result = {
            "specialist_id": top_id,
            "mode": "inferred",
            "reason": "matched_retained_example",
            "top_similarity": round(top_score, 6),
            "margin": round(margin, 6),
            "top_example_similarity": round(example_similarity, 6),
            "minimum_example_similarity": self.minimum_example_similarity,
            "scores": [
                {"specialist_id": specialist_id, "similarity": round(score, 6)}
                for specialist_id, score in scores[:5]
            ],
        }
        self._route_cache[signature] = result
        self._route_cache.move_to_end(signature)
        if len(self._route_cache) > MAX_ROUTE_CACHE_ENTRIES:
            self._route_cache.popitem(last=False)
        return result

    def clear(self) -> None:
        self._profiles.clear()
        self._example_counts.clear()
        self._exemplars.clear()
        self._norm_cache.clear()
        self._similarity_cache.clear()
        self._route_cache.clear()
