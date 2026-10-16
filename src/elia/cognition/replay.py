from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayExample:
    example_id: str
    specialist_id: str
    text: str
    label: int
    surprise: float
    sequence: int


class BalancedReplayBuffer:
    """Bounded per-specialist replay with class coverage and surprise priority."""

    def __init__(self, capacity_per_specialist: int = 128) -> None:
        if capacity_per_specialist < 2:
            raise ValueError("capacity_per_specialist must be at least 2")
        self.capacity_per_specialist = capacity_per_specialist
        self._examples: dict[str, list[ReplayExample]] = defaultdict(list)

    @property
    def size(self) -> int:
        return sum(map(len, self._examples.values()))

    @property
    def specialist_counts(self) -> dict[str, int]:
        return {key: len(rows) for key, rows in sorted(self._examples.items())}

    def add(self, example: ReplayExample) -> bool:
        rows = self._examples[example.specialist_id]
        if any(row.example_id == example.example_id for row in rows):
            return False
        if len(rows) < self.capacity_per_specialist:
            rows.append(example)
            return True
        same_label = [row for row in rows if row.label == example.label]
        replaceable = same_label if same_label else rows
        lowest = min(replaceable, key=lambda row: (row.surprise, row.sequence))
        if example.surprise <= lowest.surprise:
            return False
        rows.remove(lowest)
        rows.append(example)
        return True

    def contains(self, example_id: str) -> bool:
        return any(
            row.example_id == example_id
            for rows in self._examples.values()
            for row in rows
        )

    def balanced_batch(
        self,
        max_per_specialist: int,
        include_id: str | None = None,
        max_total: int | None = None,
    ) -> list[ReplayExample]:
        if max_total is not None and max_total < 0:
            raise ValueError("max_total must be non-negative")
        if max_per_specialist <= 0 or max_total == 0:
            return []
        selected_by_specialist: dict[str, list[ReplayExample]] = {}
        for specialist_id in sorted(self._examples):
            rows = self._examples[specialist_id]
            if len(rows) <= max_per_specialist:
                selected = list(rows)
            else:
                selected = []
                for label in (0, 1):
                    class_rows = sorted(
                        (row for row in rows if row.label == label),
                        key=lambda row: (-row.surprise, row.sequence),
                    )
                    selected.extend(class_rows[:max_per_specialist // 2])
                selected_ids = {row.example_id for row in selected}
                remaining = sorted(
                    (row for row in rows if row.example_id not in selected_ids),
                    key=lambda row: (-row.surprise, row.sequence),
                )
                selected.extend(remaining[:max_per_specialist - len(selected)])
                selected.sort(key=lambda row: row.sequence)
            if include_id and specialist_id in self._examples:
                current = next((row for row in rows if row.example_id == include_id), None)
                if current and all(row.example_id != include_id for row in selected):
                    if len(selected) >= max_per_specialist:
                        selected[-1] = current
                    else:
                        selected.append(current)
            selected_by_specialist[specialist_id] = selected

        result: list[ReplayExample] = []
        if max_total is None:
            for specialist_id in sorted(selected_by_specialist):
                result.extend(selected_by_specialist[specialist_id])
        else:
            positions = {specialist_id: 0 for specialist_id in selected_by_specialist}
            while len(result) < max_total:
                added = False
                for specialist_id in sorted(selected_by_specialist):
                    position = positions[specialist_id]
                    rows = selected_by_specialist[specialist_id]
                    if position < len(rows):
                        result.append(rows[position])
                        positions[specialist_id] += 1
                        added = True
                        if len(result) == max_total:
                            break
                if not added:
                    break

        if include_id:
            current = next(
                (
                    row for rows in self._examples.values()
                    for row in rows if row.example_id == include_id
                ),
                None,
            )
            if current and all(row.example_id != include_id for row in result):
                if max_total is not None and len(result) >= max_total:
                    same_specialist = next(
                        (index for index in range(len(result) - 1, -1, -1)
                         if result[index].specialist_id == current.specialist_id),
                        len(result) - 1,
                    )
                    result.pop(same_specialist)
                result.append(current)
        return result

    def clear(self) -> None:
        self._examples.clear()
