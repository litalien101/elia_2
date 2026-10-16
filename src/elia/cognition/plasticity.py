from __future__ import annotations

import math
import random
from array import array
from collections.abc import Mapping


class CompetitivePlasticSubstrate:
    """Experimental CPU-only sparse competitive units with local plasticity."""

    def __init__(
        self,
        unit_count: int = 96,
        feature_count: int = 10_000,
        top_k: int = 5,
        learning_rate: float = 0.06,
        max_fanout: int = 16,
        usage_decay: float = 0.996,
        usage_penalty: float = 0.018,
        associative_learning: bool = True,
        seed: int = 0,
    ) -> None:
        if unit_count < 2:
            raise ValueError("unit_count must be at least 2")
        if feature_count < 8:
            raise ValueError("feature_count must be at least 8")
        if not 1 <= top_k <= unit_count:
            raise ValueError("top_k must be between 1 and unit_count")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if max_fanout < 1:
            raise ValueError("max_fanout must be positive")
        if not 0 < usage_decay <= 1:
            raise ValueError("usage_decay must be in (0, 1]")
        if usage_penalty < 0:
            raise ValueError("usage_penalty must be non-negative")

        self.unit_count = unit_count
        self.feature_count = feature_count
        self.top_k = top_k
        self.learning_rate = learning_rate
        self.max_fanout = max_fanout
        self.usage_decay = usage_decay
        self.usage_penalty = usage_penalty
        self.associative_learning = associative_learning
        self._weights = array("f", [0.0]) * (unit_count * feature_count)
        self._norm_squared = [0.0] * unit_count
        self._usage = [0.05] * unit_count
        self._associations: list[dict[int, float]] = [{} for _ in range(unit_count)]
        self.observations = 0

        rng = random.Random(seed)
        for unit in range(unit_count):
            offset = unit * feature_count
            norm_squared = 0.0
            for feature in range(feature_count):
                value = rng.random() + 0.05
                self._weights[offset + feature] = value
                norm_squared += value * value
            norm = math.sqrt(norm_squared)
            for feature in range(feature_count):
                self._weights[offset + feature] /= norm
            self._norm_squared[unit] = 1.0

    def _normalized_input(self, features: Mapping[int, float]) -> dict[int, float]:
        cleaned: dict[int, float] = {}
        for feature, value in features.items():
            if isinstance(feature, bool) or not isinstance(feature, int):
                raise ValueError("feature ids must be integers")
            if not 0 <= feature < self.feature_count:
                raise ValueError(f"feature id out of range: {feature}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("feature activations must be finite non-negative numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("feature activations must be finite non-negative numbers")
            if numeric:
                cleaned[feature] = numeric
        norm = math.sqrt(sum(value * value for value in cleaned.values()))
        if not norm:
            raise ValueError("features must contain at least one positive activation")
        return {feature: value / norm for feature, value in cleaned.items()}

    def encode(self, features: Mapping[int, float]) -> dict[int, float]:
        normalized = self._normalized_input(features)
        usage_mean = sum(self._usage) / self.unit_count
        scores: list[tuple[int, float]] = []
        for unit in range(self.unit_count):
            offset = unit * self.feature_count
            dot = sum(
                self._weights[offset + feature] * value
                for feature, value in normalized.items()
            )
            score = dot / math.sqrt(self._norm_squared[unit])
            if self.usage_penalty:
                score -= self.usage_penalty * self._usage[unit] / max(usage_mean, 1e-8)
            scores.append((unit, score))
        winners = sorted(scores, key=lambda row: (-row[1], row[0]))[:self.top_k]
        peak = max((score for _, score in winners), default=0.0)
        if peak <= 0:
            return {unit: 1.0 / len(winners) for unit, _ in winners}
        return {unit: max(0.0, score) / peak for unit, score in winners}

    def observe(self, features: Mapping[int, float]) -> dict[int, float]:
        normalized = self._normalized_input(features)
        activity = self.encode(normalized)
        ranked = sorted(activity.items(), key=lambda row: (-row[1], row[0]))

        for rank, (unit, _) in enumerate(ranked):
            rate = self.learning_rate if rank == 0 else self.learning_rate * 0.18 / (rank + 1)
            offset = unit * self.feature_count
            norm_squared = self._norm_squared[unit]
            for feature, value in normalized.items():
                index = offset + feature
                old = self._weights[index]
                updated = old + rate * (value - old)
                norm_squared += updated * updated - old * old
                self._weights[index] = updated
            self._norm_squared[unit] = max(norm_squared, 1e-12)

        self._usage = [usage * self.usage_decay for usage in self._usage]
        self._usage[ranked[0][0]] += 0.025
        if self.associative_learning:
            active = [unit for unit, value in ranked if value > 0]
            for source in active:
                outgoing = self._associations[source]
                for target in active:
                    if source == target:
                        continue
                    if target in outgoing:
                        outgoing[target] = min(1.0, outgoing[target] + 0.025)
                    elif len(outgoing) < self.max_fanout:
                        outgoing[target] = 0.025
                    else:
                        weakest = min(outgoing, key=outgoing.__getitem__)
                        if outgoing[weakest] < 0.025:
                            del outgoing[weakest]
                            outgoing[target] = 0.025

        self.observations += 1
        return activity

    def recall(self, cues: Mapping[int, float], limit: int | None = None) -> dict[int, float]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        direct = self._normalized_unit_activity(cues)
        spread: dict[int, float] = {}
        for source, activation in direct.items():
            for target, weight in self._associations[source].items():
                spread[target] = spread.get(target, 0.0) + activation * weight
        result = {unit: value for unit, value in spread.items() if unit not in direct}
        return dict(sorted(
            result.items(), key=lambda row: (-row[1], row[0])
        )[:limit])

    def _normalized_unit_activity(self, activity: Mapping[int, float]) -> dict[int, float]:
        cleaned: dict[int, float] = {}
        for unit, value in activity.items():
            if isinstance(unit, bool) or not isinstance(unit, int) or not 0 <= unit < self.unit_count:
                raise ValueError(f"unit id out of range: {unit}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("unit activations must be finite non-negative numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError("unit activations must be finite non-negative numbers")
            if numeric:
                cleaned[unit] = numeric
        if not cleaned:
            raise ValueError("unit activity must contain a positive activation")
        return cleaned

    def learn_association(self, first: Mapping[int, float], second: Mapping[int, float]) -> None:
        left = self._normalized_unit_activity(first)
        right = self._normalized_unit_activity(second)
        for source, source_activation in left.items():
            for target, target_activation in right.items():
                if source == target:
                    continue
                outgoing = self._associations[source]
                increment = 0.025 * source_activation * target_activation
                if target in outgoing:
                    outgoing[target] = min(1.0, outgoing[target] + increment)
                elif len(outgoing) < self.max_fanout:
                    outgoing[target] = increment
                else:
                    weakest = min(outgoing, key=outgoing.__getitem__)
                    if outgoing[weakest] < increment:
                        del outgoing[weakest]
                        outgoing[target] = increment

    @property
    def synapse_count(self) -> int:
        return sum(len(outgoing) for outgoing in self._associations)

    def receptive_field(self, unit: int) -> dict[int, float]:
        if isinstance(unit, bool) or not isinstance(unit, int) or not 0 <= unit < self.unit_count:
            raise ValueError(f"unit id out of range: {unit}")
        offset = unit * self.feature_count
        norm = math.sqrt(self._norm_squared[unit])
        return {
            feature: self._weights[offset + feature] / norm
            for feature in range(self.feature_count)
        }

    def status(self) -> dict[str, int | float | str]:
        return {
            "kind": "experimental_competitive_plasticity",
            "unit_count": self.unit_count,
            "feature_count": self.feature_count,
            "top_k": self.top_k,
            "observations": self.observations,
            "synapses": self.synapse_count,
            "synapse_capacity": self.unit_count * self.max_fanout,
            "parameter_count": self.unit_count * self.feature_count,
            "learning_rate": self.learning_rate,
            "associative_learning": self.associative_learning,
        }

    def clear(self, seed: int = 0) -> None:
        self.__init__(
            unit_count=self.unit_count,
            feature_count=self.feature_count,
            top_k=self.top_k,
            learning_rate=self.learning_rate,
            max_fanout=self.max_fanout,
            usage_decay=self.usage_decay,
            usage_penalty=self.usage_penalty,
            associative_learning=self.associative_learning,
            seed=seed,
        )
