from __future__ import annotations

from collections.abc import Mapping

from .core import SparseLogisticReadout


class SpecialistCircuit:
    """A small task-specific readout connected to the shared representation."""

    def __init__(self, specialist_id: str) -> None:
        self.specialist_id = specialist_id
        self.readout = SparseLogisticReadout(learning_rate=0.035, l2=0.001)
        self.examples_seen = 0

    def logit(self, features: Mapping[int, float]) -> float:
        return self.readout.logit(features)

    def probability(self, features: Mapping[int, float]) -> float:
        return SparseLogisticReadout.sigmoid(self.logit(features))

    def learn(self, features: Mapping[int, float], target: int) -> None:
        self.readout.update(features, target)
        self.examples_seen += 1
