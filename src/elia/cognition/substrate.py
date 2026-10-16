from __future__ import annotations

from array import array
from collections.abc import Mapping
from typing import Iterable


class SparseNeuronSubstrate:
    """A bounded, sparse associative network of simple stateful units."""

    def __init__(
        self,
        unit_count: int = 10_000,
        max_fanout: int = 16,
        activation_threshold: float = 0.02,
        decay: float = 0.85,
    ) -> None:
        if unit_count < 128:
            raise ValueError("unit_count must be at least 128")
        if max_fanout < 1:
            raise ValueError("max_fanout must be positive")
        self.unit_count = unit_count
        self.max_fanout = max_fanout
        self.activation_threshold = activation_threshold
        self.decay = decay
        self._state = array("f", [0.0]) * unit_count
        self._outgoing: list[dict[int, float]] = [{} for _ in range(unit_count)]

    @property
    def synapse_count(self) -> int:
        return sum(map(len, self._outgoing))

    @property
    def synapse_capacity(self) -> int:
        return self.unit_count * self.max_fanout

    def activate(self, input_activity: Mapping[int, float]) -> dict[int, float]:
        for unit, activation in input_activity.items():
            if not 0 <= unit < self.unit_count:
                raise ValueError(f"unit id out of range: {unit}")
            self._state[unit] = max(self._state[unit] * self.decay, min(1.0, activation))

        spread: dict[int, float] = {}
        for source, activation in input_activity.items():
            for target, weight in self._outgoing[source].items():
                spread[target] = min(1.0, spread.get(target, 0.0) + activation * weight)

        output = dict(input_activity)
        for unit, activation in spread.items():
            if activation >= self.activation_threshold:
                self._state[unit] = max(self._state[unit] * self.decay, activation)
                output[unit] = max(output.get(unit, 0.0), activation)

        if len(output) > 256:
            strongest = sorted(output.items(), key=lambda item: (-item[1], item[0]))[:256]
            output = dict(strongest)
        return output

    def learn_associations(
        self,
        co_activated: Iterable[int],
        surprise: float,
    ) -> int:
        """Strengthen bounded directed links among co-active units."""
        units = list(dict.fromkeys(co_activated))[:64]
        if len(units) < 2 or surprise <= 0:
            return 0
        increment = min(0.12, max(0.005, surprise * 0.08))
        added = 0
        for source in units:
            outgoing = self._outgoing[source]
            for target in units:
                if source == target:
                    continue
                if target in outgoing:
                    outgoing[target] = min(1.0, outgoing[target] + increment)
                    continue
                if len(outgoing) < self.max_fanout:
                    outgoing[target] = increment
                    added += 1
                    continue
                weakest = min(outgoing, key=outgoing.__getitem__)
                if outgoing[weakest] < increment:
                    del outgoing[weakest]
                    outgoing[target] = increment
                    added += 1
        return added

    def clear(self) -> None:
        self._state = array("f", [0.0]) * self.unit_count
        self._outgoing = [{} for _ in range(self.unit_count)]
