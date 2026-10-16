from __future__ import annotations

import argparse
import json
import math
import random

from .plasticity import CompetitivePlasticSubstrate


FEATURE_COUNT = 64
CONCEPT_COUNT = 8


def _make_prototypes() -> list[set[int]]:
    rng = random.Random(9001)
    return [set(rng.sample(range(24), 10)) for _ in range(CONCEPT_COUNT)]


def _make_examples(
    prototypes: list[set[int]],
    seed: int,
    examples_per_concept: int,
) -> list[tuple[int, dict[int, float]]]:
    rng = random.Random(seed)
    examples: list[tuple[int, dict[int, float]]] = []
    for concept, prototype in enumerate(prototypes):
        outside = [feature for feature in range(24) if feature not in prototype]
        for _ in range(examples_per_concept):
            features = {
                feature: rng.uniform(0.7, 1.3)
                for feature in prototype
                if rng.random() > 0.12
            }
            for feature in rng.sample(outside, 3):
                features[feature] = features.get(feature, 0.0) + 0.4
            examples.append((concept, features))
    rng.shuffle(examples)
    return examples


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    dot = sum(value * right.get(feature, 0.0) for feature, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _nearest_centroid_accuracy(
    substrate: CompetitivePlasticSubstrate,
    training_rows: list[tuple[int, dict[int, float]]],
    held_out_rows: list[tuple[int, dict[int, float]]],
) -> float:
    sums: list[dict[int, float]] = [{} for _ in range(CONCEPT_COUNT)]
    counts = [0] * CONCEPT_COUNT
    for concept, features in training_rows:
        for unit, activation in substrate.encode(features).items():
            sums[concept][unit] = sums[concept].get(unit, 0.0) + activation
        counts[concept] += 1
    centroids = [
        {unit: value / counts[concept] for unit, value in sums[concept].items()}
        for concept in range(CONCEPT_COUNT)
    ]

    correct = 0
    for expected, features in held_out_rows:
        activity = substrate.encode(features)
        predicted = max(
            range(CONCEPT_COUNT),
            key=lambda concept: _cosine(activity, centroids[concept]),
        )
        correct += predicted == expected
    return correct / len(held_out_rows)


def _run_seed(seed: int) -> dict[str, float | int | bool]:
    prototypes = _make_prototypes()
    training_rows = _make_examples(prototypes, seed + 100, 40)
    held_out_rows = _make_examples(prototypes, seed + 200, 30)

    frozen = CompetitivePlasticSubstrate(
        unit_count=96,
        feature_count=FEATURE_COUNT,
        top_k=5,
        seed=seed,
    )
    plastic = CompetitivePlasticSubstrate(
        unit_count=96,
        feature_count=FEATURE_COUNT,
        top_k=5,
        associative_learning=False,
        seed=seed,
    )
    frozen_accuracy = _nearest_centroid_accuracy(frozen, training_rows, held_out_rows)
    for _, features in training_rows:
        plastic.observe(features)
    plastic_accuracy = _nearest_centroid_accuracy(plastic, training_rows, held_out_rows)

    first_concept = next(features for concept, features in training_rows if concept == 0)
    second_concept = next(features for concept, features in training_rows if concept == 1)
    first_activity = plastic.encode(first_concept)
    second_activity = plastic.encode(second_concept)
    no_recurrence_recall = plastic.recall(first_activity)
    plastic.learn_association(first_activity, second_activity)
    association_recall = bool(
        set(plastic.recall(first_activity)).intersection(second_activity)
    )
    status = plastic.status()
    return {
        "seed": seed,
        "frozen_held_out_accuracy": frozen_accuracy,
        "plastic_held_out_accuracy": plastic_accuracy,
        "association_recalled_after_pairing": association_recall,
        "recall_without_association_learning_is_empty": not no_recurrence_recall,
        "observations": int(status["observations"]),
        "synapses": int(status["synapses"]),
        "synapse_capacity": int(status["synapse_capacity"]),
    }


def run_plasticity_benchmark(seed: int = 2026, seeds: int = 5) -> dict[str, object]:
    """Compare frozen and plastic features on synthetic held-out concept examples."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    runs = [_run_seed(seed + offset) for offset in range(seeds)]
    frozen_accuracies = [float(run["frozen_held_out_accuracy"]) for run in runs]
    plastic_accuracies = [float(run["plastic_held_out_accuracy"]) for run in runs]
    alignment_changes = [
        float(run["plastic_held_out_accuracy"]) - float(run["frozen_held_out_accuracy"])
        for run in runs
    ]
    return {
        "benchmark": "elia_competitive_plasticity_synthetic_v2",
        "first_seed": seed,
        "seeds": seeds,
        "warning": (
            "Synthetic overlapping-feature concepts only. Competitive updates receive "
            "unlabeled inputs; labels are used only by an offline nearest-centroid "
            "evaluation. This does not measure natural-language understanding or "
            "generalization beyond this generated task."
        ),
        "training_examples_per_concept": 40,
        "held_out_examples_per_concept": 30,
        "mean_frozen_held_out_accuracy": sum(frozen_accuracies) / seeds,
        "mean_plastic_held_out_accuracy": sum(plastic_accuracies) / seeds,
        "mean_accuracy_change": sum(alignment_changes) / seeds,
        "association_recall_all_seeds": all(
            bool(run["association_recalled_after_pairing"]) for run in runs
        ),
        "association_disabled_recall_empty_all_seeds": all(
            bool(run["recall_without_association_learning_is_empty"]) for run in runs
        ),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_plasticity_benchmark(args.seed, args.seeds), indent=2))


if __name__ == "__main__":
    main()
