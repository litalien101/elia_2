from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from elia.cognition.core import detect_hardware_profile
    from elia.cognition.reasoning import BoundedRuleEngine, Fact, parse_literal, parse_rule
    from elia.cognition.system import CognitiveSystem
else:
    from .core import detect_hardware_profile
    from .reasoning import BoundedRuleEngine, Fact, parse_literal, parse_rule
    from .system import CognitiveSystem

DOMAINS = ("graphics", "networking", "storage", "planning", "audio", "security")
HELP_TRAINING_NAMES = (
    "Alice", "Dana", "Nina", "Omar", "Priya",
    "Ravi", "Tara", "Uma", "Vik", "Wendy",
)
HELP_HELD_OUT_NAMES = (
    "Luca", "Mara", "Noah", "Olive", "Pia", "Quinn",
    "Rosa", "Sam", "Victor", "Wren", "Vera", "Xim",
    "Yara", "Zoe", "Aster", "Bram", "Celine", "Dorian",
    "Elena", "Faye", "Gino", "Hugo", "Iris",
)
HELP_POSITIVE_FORMS = (
    "{actor} helps {target}",
    "{actor} is helpful to {target}",
    "{actor} supports {target}",
    "{actor} assists {target}",
    "{actor} was there for {target}",
)
HELP_NEGATIVE_FORMS = (
    "{actor} does not help {target}",
    "{actor} is not helpful to {target}",
    "{actor} never helps {target}",
    "{actor} does not support {target}",
    "{target} is not helped by {actor}",
)
HELP_UNSEEN_POSITIVE_FORMS = (
    "{actor} lends {target} a hand",
    "{actor} comes to the aid of {target}",
    "{actor} offers assistance to {target}",
    "{actor} steps in for {target}",
    "{actor} provides support for {target}",
)
HELP_UNSEEN_NEGATIVE_FORMS = (
    "{actor} refuses to help {target}",
    "{actor} offers no help to {target}",
    "{actor} withholds assistance from {target}",
    "{actor} leaves {target} without help",
    "{actor} does not come to the aid of {target}",
)


@dataclass(frozen=True)
class Example:
    specialist_id: str
    text: str
    label: int


def build_curriculum(seed: int, examples_per_class: int) -> tuple[list[list[Example]], list[list[Example]]]:
    rng = random.Random(seed)
    training: list[list[Example]] = []
    testing: list[list[Example]] = []
    for task, domain in enumerate(DOMAINS):
        specialist_id = domain
        positive_signal = "alpha" if task % 2 == 0 else "beta"
        train_rows: list[Example] = []
        test_rows: list[Example] = []
        for signal in ("alpha", "beta"):
            label = int(signal == positive_signal)
            for index in range(examples_per_class):
                train_rows.append(Example(
                    specialist_id,
                    f"{domain} signal {signal} training sample {index}",
                    label,
                ))
            for index in range(max(20, examples_per_class)):
                test_rows.append(Example(
                    specialist_id,
                    f"for {domain} evaluate signal {signal} heldout sample {index}",
                    label,
                ))
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)
        training.append(train_rows)
        testing.append(test_rows)
    return training, testing


def evaluate(system: CognitiveSystem, tasks: list[list[Example]]) -> dict[str, object]:
    per_specialist: dict[str, float] = {}
    route_correct = 0
    examples = 0
    correct = 0
    abstentions = 0
    for task in tasks:
        task_correct = 0
        for example in task:
            result = system.predict(example.text)
            examples += 1
            if result["decision"] == "abstain":
                abstentions += 1
                continue
            if result["specialist_id"] == example.specialist_id:
                route_correct += 1
            if result["predicted_label"] == example.label:
                correct += 1
                task_correct += 1
        per_specialist[task[0].specialist_id] = round(
            task_correct / len(task) if task else 0.0, 6
        )
    accuracy = correct / examples if examples else 0.0
    route_accuracy = route_correct / max(1, examples - abstentions)
    return {
        "accuracy": accuracy,
        "per_specialist_accuracy": per_specialist,
        "routing_accuracy": route_accuracy,
        "abstention_rate": abstentions / max(1, examples),
        "sample_size": examples,
    }


def run_architecture(
    training: list[list[Example]],
    testing: list[list[Example]],
    replay: bool,
    specialists: bool,
    replay_per_specialist: int,
) -> dict[str, object]:
    system = CognitiveSystem(
        replay_enabled=replay,
        specialists_enabled=specialists,
        replay_per_specialist=replay_per_specialist,
    )
    checkpoints: list[dict[str, object]] = []
    started = time.perf_counter()
    for task in training:
        for sequence, example in enumerate(task):
            system.observe(
                example_id=f"{example.specialist_id}-{sequence}",
                specialist_id=example.specialist_id,
                text=example.text,
                label=example.label,
            )
        checkpoints.append(evaluate(system, testing))
    elapsed = time.perf_counter() - started
    final = checkpoints[-1] if checkpoints else evaluate(system, testing)
    forgetting_by_task: list[float] = []
    for task_index, task in enumerate(testing):
        key = task[0].specialist_id
        seen = [float(stage["per_specialist_accuracy"][key]) for stage in checkpoints[task_index:]]
        forgetting_by_task.append(max(seen, default=0.0) - (seen[-1] if seen else 0.0))
    return {
        "architecture": (
            "shared_core_specialists_balanced_replay" if specialists and replay
            else "shared_core_specialists_no_replay" if specialists
            else "shared_core_no_replay"
        ),
        **final,
        "mean_forgetting": sum(forgetting_by_task) / max(1, len(forgetting_by_task)),
        "training_seconds": round(elapsed, 6),
        "system": system,
    }


def run_benchmark(seed: int = 2026, examples_per_class: int = 24) -> dict[str, object]:
    if examples_per_class < 2:
        raise ValueError("examples_per_class must be at least 2")
    training, testing = build_curriculum(seed, examples_per_class)
    configs: tuple[tuple[bool, bool, int], ...] = (
        (False, False, 1),
        (False, True, 1),
        (True, True, 16),
    )
    raw = [
        run_architecture(training, testing, replay, specialists, replay_size)
        for replay, specialists, replay_size in configs
    ]
    results: list[dict[str, object]] = []
    for result in raw:
        row = {key: value for key, value in result.items() if key != "system"}
        row["architecture_status"] = result["system"].status()
        results.append(row)
    return {
        "benchmark": "elia_synthetic_continual_learning_v1",
        "seed": seed,
        "tasks": 6,
        "examples_per_class_per_task": examples_per_class,
        "seed_variation": "only training-example order changes; generated examples and held-out cases stay fixed",
        "warning": (
            "Synthetic lexical curriculum with known task cues. This evaluates retention and routing "
            "in a controlled toy setting; it is not a natural-language, open-world, or intelligence test."
        ),
        "results": results,
    }


def run_sweep(
    first_seed: int = 2026,
    seeds: int = 5,
    examples_per_class: int = 24,
) -> dict[str, object]:
    if seeds < 2:
        raise ValueError("seeds must be at least 2 for a sweep")
    runs = [run_benchmark(first_seed + offset, examples_per_class) for offset in range(seeds)]
    names = [row["architecture"] for row in runs[0]["results"]]
    aggregate: list[dict[str, object]] = []
    for index, name in enumerate(names):
        rows = [run["results"][index] for run in runs]
        summary: dict[str, object] = {"architecture": name, "seed_runs": seeds}
        for metric in ("accuracy", "routing_accuracy", "abstention_rate", "mean_forgetting", "training_seconds"):
            values = [float(row[metric]) for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            sd = variance ** 0.5
            summary[metric] = round(mean, 6)
            if metric != "training_seconds":
                summary[f"{metric}_order_sd"] = round(sd, 6)
        specialist_ids = rows[0]["per_specialist_accuracy"].keys()
        summary["per_specialist_accuracy"] = {
            specialist_id: round(
                sum(float(row["per_specialist_accuracy"][specialist_id]) for row in rows) / len(rows),
                6,
            )
            for specialist_id in specialist_ids
        }
        summary["sample_size_per_seed"] = rows[0]["sample_size"]
        summary["last_run_architecture_status"] = rows[-1]["architecture_status"]
        aggregate.append(summary)
    return {
        "benchmark": "elia_synthetic_continual_learning_v1",
        "first_seed": first_seed,
        "seeds": seeds,
        "tasks": runs[0]["tasks"],
        "examples_per_class_per_task": examples_per_class,
        "seed_variation": runs[0]["seed_variation"],
        "warning": runs[0]["warning"],
        "results": aggregate,
        "per_seed_results": [
            {
                "seed": run["seed"],
                "results": [
                    {key: value for key, value in row.items() if key != "architecture_status"}
                    for row in run["results"]
                ],
            }
            for run in runs
        ],
    }


def _help_relation_training_examples() -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    names = HELP_TRAINING_NAMES
    for index in range(24):
        actor = names[(index * 3) % len(names)]
        target = names[(index * 5 + 2) % len(names)]
        other = names[(index * 7 + 4) % len(names)]
        if actor == target:
            target = names[(index + 1) % len(names)]
        pairs.extend([
            (HELP_POSITIVE_FORMS[index % len(HELP_POSITIVE_FORMS)].format(actor=actor, target=target), 1),
            (HELP_NEGATIVE_FORMS[index % len(HELP_NEGATIVE_FORMS)].format(actor=actor, target=target), 0),
            (f"{actor} never helps {other}", 0),
        ])
    return pairs


def run_generalization_benchmark(seed: int = 2026) -> dict[str, object]:
    return run_unseen_entity_pair_benchmark(seed=seed)


def run_paraphrased_generalization_benchmark(seed: int = 2026) -> dict[str, object]:
    """Backward-compatible name for the unseen-entity-pair benchmark."""
    return run_unseen_entity_pair_benchmark(seed=seed)


def run_unseen_entity_pair_benchmark(seed: int = 2026) -> dict[str, object]:
    rng = random.Random(seed)
    training = _help_relation_training_examples()
    rng.shuffle(training)

    held_out_pairs = [
        (name, HELP_HELD_OUT_NAMES[(index + len(HELP_HELD_OUT_NAMES) // 3) % len(HELP_HELD_OUT_NAMES)])
        for index, name in enumerate(HELP_HELD_OUT_NAMES)
    ]
    held_out = [
        (form.format(actor=actor, target=target), label)
        for actor, target in held_out_pairs
        for label, forms in ((1, HELP_POSITIVE_FORMS), (0, HELP_NEGATIVE_FORMS))
        for form in forms
    ]
    training_entity_names = sorted({
        name for name in HELP_TRAINING_NAMES
        if any(name in text.split() for text, _ in training)
    })
    held_out_entity_names = sorted({
        name for name in HELP_HELD_OUT_NAMES
        if any(name in text.split() for text, _ in held_out)
    })
    if held_out_entity_names != sorted(HELP_HELD_OUT_NAMES):
        raise ValueError("all declared held-out entity names must appear in held-out examples")
    entity_names_disjoint = not set(training_entity_names) & set(held_out_entity_names)
    if not entity_names_disjoint:
        raise ValueError("held-out entity names must be disjoint from training names")

    system = CognitiveSystem(unit_count=4096, fanout=16, replay_capacity_per_specialist=128, replay_per_specialist=32)
    for sequence, (text, label) in enumerate(training):
        system.observe(f"generalization-{sequence}", "helpfulness", text, label)

    results: list[dict[str, object]] = []
    correct = 0
    positive_correct = 0
    negative_correct = 0
    positive_total = 0
    negative_total = 0
    positive_probabilities: list[float] = []
    negative_probabilities: list[float] = []
    for text, label in held_out:
        prediction = system.predict(text, "helpfulness")
        is_correct = prediction["predicted_label"] == label
        if is_correct:
            correct += 1
        if label == 1:
            positive_total += 1
            if is_correct:
                positive_correct += 1
            positive_probabilities.append(float(prediction.get("probability", 0.0)))
        else:
            negative_total += 1
            if is_correct:
                negative_correct += 1
            negative_probabilities.append(float(prediction.get("probability", 0.0)))
        results.append({
            "text": text,
            "label": label,
            "predicted_label": int(prediction["predicted_label"]),
            "probability": round(float(prediction.get("probability", 0.0)), 6),
            "decision": prediction["decision"],
            "specialist_id": prediction.get("specialist_id"),
        })

    accuracy = correct / len(held_out)
    positive_accuracy = positive_correct / max(1, positive_total)
    negative_accuracy = negative_correct / max(1, negative_total)
    class_balance_pass = positive_accuracy > 0.5 and negative_accuracy > 0.5
    report = {
        "benchmark": "elia_help_relation_unseen_entity_pairs_v4",
        "seed": seed,
        "training_examples": len(training),
        "held_out_examples": len(held_out),
        "held_out_positive_examples": positive_total,
        "held_out_negative_examples": negative_total,
        "generalization_scope": "unseen_entity_names_and_pairs_with_training_seen_templates",
        "entity_names_disjoint": entity_names_disjoint,
        "training_entity_names": training_entity_names,
        "held_out_entity_names": held_out_entity_names,
        "unseen_templates": False,
        "accuracy": round(accuracy, 6),
        "positive_accuracy": round(positive_accuracy, 6),
        "negative_accuracy": round(negative_accuracy, 6),
        "class_balance_pass": class_balance_pass,
        "mean_positive_probability": round(sum(positive_probabilities) / max(1, len(positive_probabilities)), 6),
        "mean_negative_probability": round(sum(negative_probabilities) / max(1, len(negative_probabilities)), 6),
        "status": (
            "pass"
            if accuracy >= 0.8 and class_balance_pass
            else "needs_more_generalization"
        ),
        "warning": (
            "Synthetic transfer benchmark for a simple help/no-help relation. "
            "Held-out examples use unseen names and entity pairs but templates seen during training; "
            "this does not measure transfer to unseen paraphrases or open-ended world understanding."
        ),
        "results": results,
    }
    return report


def run_unseen_template_benchmark(seed: int = 2026) -> dict[str, object]:
    rng = random.Random(seed)
    training = _help_relation_training_examples()
    rng.shuffle(training)
    held_out_pairs = [
        (name, HELP_HELD_OUT_NAMES[(index + len(HELP_HELD_OUT_NAMES) // 3) % len(HELP_HELD_OUT_NAMES)])
        for index, name in enumerate(HELP_HELD_OUT_NAMES)
    ]
    held_out = [
        (form.format(actor=actor, target=target), label, form)
        for actor, target in held_out_pairs
        for label, forms in (
            (1, HELP_UNSEEN_POSITIVE_FORMS),
            (0, HELP_UNSEEN_NEGATIVE_FORMS),
        )
        for form in forms
    ]
    if any(form in HELP_POSITIVE_FORMS + HELP_NEGATIVE_FORMS for _, _, form in held_out):
        raise ValueError("unseen-template forms must not overlap training templates")

    training_entity_names = sorted({
        name for name in HELP_TRAINING_NAMES
        if any(name in text.split() for text, _ in training)
    })
    held_out_entity_names = sorted({
        name for name in HELP_HELD_OUT_NAMES
        if any(name in text.split() for text, _, _ in held_out)
    })
    entity_names_disjoint = not set(training_entity_names) & set(held_out_entity_names)
    if held_out_entity_names != sorted(HELP_HELD_OUT_NAMES) or not entity_names_disjoint:
        raise ValueError("unseen-template evaluation requires complete, disjoint entity names")

    system = CognitiveSystem(
        unit_count=4096,
        fanout=16,
        replay_capacity_per_specialist=128,
        replay_per_specialist=32,
    )
    for sequence, (text, label) in enumerate(training):
        system.observe(f"unseen-template-{sequence}", "helpfulness", text, label)

    results: list[dict[str, object]] = []
    template_totals: Counter[str] = Counter()
    template_correct: Counter[str] = Counter()
    positive_correct = negative_correct = 0
    positive_total = negative_total = correct = 0
    for text, label, template in held_out:
        prediction = system.predict(text, "helpfulness")
        predicted_label = int(prediction["predicted_label"])
        is_correct = predicted_label == label
        correct += int(is_correct)
        template_totals[template] += 1
        template_correct[template] += int(is_correct)
        if label:
            positive_total += 1
            positive_correct += int(is_correct)
        else:
            negative_total += 1
            negative_correct += int(is_correct)
        results.append({
            "text": text,
            "template": template,
            "label": label,
            "predicted_label": predicted_label,
            "probability": round(float(prediction.get("probability", 0.0)), 6),
            "decision": prediction["decision"],
        })

    accuracy = correct / len(held_out)
    positive_accuracy = positive_correct / positive_total
    negative_accuracy = negative_correct / negative_total
    class_balance_pass = positive_accuracy > 0.5 and negative_accuracy > 0.5
    return {
        "benchmark": "elia_help_relation_unseen_templates_v1",
        "seed": seed,
        "training_examples": len(training),
        "held_out_examples": len(held_out),
        "held_out_positive_examples": positive_total,
        "held_out_negative_examples": negative_total,
        "generalization_scope": "unseen_entity_names_and_templates",
        "entity_names_disjoint": entity_names_disjoint,
        "training_entity_names": training_entity_names,
        "held_out_entity_names": held_out_entity_names,
        "unseen_templates": True,
        "training_templates": list(HELP_POSITIVE_FORMS + HELP_NEGATIVE_FORMS),
        "held_out_templates": list(HELP_UNSEEN_POSITIVE_FORMS + HELP_UNSEEN_NEGATIVE_FORMS),
        "accuracy": round(accuracy, 6),
        "positive_accuracy": round(positive_accuracy, 6),
        "negative_accuracy": round(negative_accuracy, 6),
        "class_balance_pass": class_balance_pass,
        "template_accuracy": {
            template: round(template_correct[template] / template_totals[template], 6)
            for template in template_totals
        },
        "status": (
            "pass"
            if accuracy >= 0.8 and class_balance_pass
            else "needs_more_generalization"
        ),
        "warning": (
            "Synthetic transfer benchmark using held-out entity names and relation templates. "
            "Templates are not present in training, but reuse some familiar vocabulary; "
            "this does not establish broad natural-language generalization."
        ),
        "results": results,
    }


def run_unseen_template_sweep(first_seed: int = 2026, seeds: int = 5) -> dict[str, object]:
    if seeds < 2:
        raise ValueError("seeds must be at least 2 for a sweep")
    reports = [
        run_unseen_template_benchmark(first_seed + offset)
        for offset in range(seeds)
    ]
    metrics = (
        "accuracy",
        "positive_accuracy",
        "negative_accuracy",
    )
    summary: dict[str, object] = {}
    for metric in metrics:
        values = [float(report[metric]) for report in reports]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        summary[f"mean_{metric}"] = round(mean, 6)
        summary[f"{metric}_order_sd"] = round(variance ** 0.5, 6)
        summary[f"min_{metric}"] = round(min(values), 6)
        summary[f"max_{metric}"] = round(max(values), 6)
    pass_count = sum(report["status"] == "pass" for report in reports)
    pass_rate = pass_count / seeds
    z = 1.96
    denominator = 1 + z * z / seeds
    center = (pass_rate + z * z / (2 * seeds)) / denominator
    margin = (
        z
        * ((pass_rate * (1 - pass_rate) / seeds + z * z / (4 * seeds * seeds)) ** 0.5)
        / denominator
    )
    return {
        "benchmark": "elia_help_relation_unseen_templates_sweep_v1",
        "first_seed": first_seed,
        "seeds": seeds,
        "pass_count": pass_count,
        "pass_rate": round(pass_rate, 6),
        "pass_rate_wilson_95_interval": [
            round(max(0.0, center - margin), 6),
            round(min(1.0, center + margin), 6),
        ],
        "interval_scope": "sampled_training_order_pass_rate_for_this_fixed_synthetic_benchmark",
        "seed_results": [
            {
                "seed": report["seed"],
                "accuracy": report["accuracy"],
                "positive_accuracy": report["positive_accuracy"],
                "negative_accuracy": report["negative_accuracy"],
                "status": report["status"],
            }
            for report in reports
        ],
        **summary,
        "warning": (
            "Training-order sweep of a synthetic benchmark with unseen names and templates. "
            "The Wilson interval describes sampled training-order pass rate for this fixed curriculum; "
            "it is not an interval for broad language generalization."
        ),
    }


def run_symbolic_composition_benchmark(seed: int = 2026) -> dict[str, object]:
    graph_templates: list[dict[str, set[str]]] = [
        {
            "n0": {"n1", "n2"},
            "n1": {"n3"},
            "n2": {"n3"},
            "n3": {"n4"},
            "n4": set(),
        },
        {
            "n0": {"n1", "n2"},
            "n1": {"n3", "n4"},
            "n2": {"n3", "n4"},
            "n3": set(),
            "n4": set(),
        },
        {
            "n0": {"n1"},
            "n1": {"n2"},
            "n2": {"n3"},
            "n3": {"n4"},
            "n4": set(),
        },
        {
            "n0": {"n1", "n2"},
            "n1": {"n3"},
            "n2": {"n4"},
            "n3": set(),
            "n4": set(),
        },
        {
            "n0": {"n1", "n2"},
            "n1": {"n2", "n3"},
            "n2": {"n3"},
            "n3": {"n4"},
            "n4": set(),
        },
        {
            "n0": {"n1"},
            "n1": {"n2", "n3"},
            "n2": {"n4"},
            "n3": {"n4"},
            "n4": set(),
        },
    ]
    graph_cases: list[dict[str, object]] = []
    for graph_index, adjacency in enumerate(graph_templates):
        facts = [
            Fact(
                f"support_{graph_index}_{source}_{target}",
                parse_literal(f"supports({source}, {target})"),
                f"observation_{graph_index}_{source}_{target}",
            )
            for source, targets in adjacency.items()
            for target in sorted(targets)
        ]
        rules = [
            parse_rule("supports(?x, ?y) -> endorsed(?x, ?y)", "direct_endorse"),
            parse_rule("supports(?x, ?y) & supports(?y, ?z) -> chained_support(?x, ?z)", "chain_support"),
            parse_rule("supports(?x, ?y) & chained_support(?y, ?z) -> chained_support(?x, ?z)", "chain_continue"),
            parse_rule("chained_support(?x, ?y) -> endorsed(?x, ?y)", "compose_endorse"),
            parse_rule("endorsed(?x, ?y) -> trusted(?x, ?y)", "trusted_by_endorse"),
        ]
        engine = BoundedRuleEngine(facts, rules, max_work=10_000)
        queue = ["n0"]
        expected = set()
        seen = {"n0"}
        while queue:
            current = queue.pop(0)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
                    expected.add(neighbor)
        answer = engine.query(parse_literal("endorsed(n0, ?target)"))
        actual = {row["bindings"]["?target"] for row in answer["answers"]}
        expected_set = set(expected)
        accuracy = 1.0 if expected_set == actual else (
            len(expected_set & actual) / max(1, len(expected_set | actual))
        )
        graph_cases.append({
            "graph_index": graph_index,
            "expected_targets": sorted(expected_set),
            "actual_targets": sorted(actual),
            "accuracy": round(accuracy, 6),
            "state": answer["state"],
            "work_used": int(answer["work_used"]),
            "closure_size": int(answer["facts_considered"]),
        })
    mean_accuracy = sum(float(case["accuracy"]) for case in graph_cases) / max(1, len(graph_cases))
    max_work = max(int(case["work_used"]) for case in graph_cases)
    return {
        "benchmark": "elia_symbolic_composition_bounded_v1",
        "seed": seed,
        "graphs": len(graph_cases),
        "mean_accuracy": round(mean_accuracy, 6),
        "max_work_used": max_work,
        "status": "pass" if mean_accuracy >= 0.9 and max_work <= 10_000 else "needs_more_resolution",
        "warning": (
            "Bounded symbolic composition benchmark: it checks chained support relations and explicit proof provenance "
            "under fixed work limits, not open-ended planning or unrestricted natural-language reasoning."
        ),
        "results": graph_cases,
    }


def run_hardware_profile_benchmark(samples: int = 256) -> dict[str, object]:
    """Measure a bounded local throughput profile on the current machine without claiming GPT-level capability."""
    if samples < 8:
        raise ValueError("samples must be at least 8")
    hardware = detect_hardware_profile()
    system = CognitiveSystem(
        unit_count=4096,
        fanout=16,
        replay_capacity_per_specialist=64,
        replay_per_specialist=8,
        hardware_profile=hardware,
    )
    prompts = [
        f"domain {domain} signal alpha training sample {index}"
        for domain in ("graphics", "networking", "storage", "planning")
        for index in range(samples // 4)
    ]
    start = time.perf_counter()
    for sequence, text in enumerate(prompts):
        system.observe(f"bench-{sequence}", "benchmark", text, 1 if "alpha" in text else 0)
        system.predict(text, "benchmark")
    elapsed = time.perf_counter() - start
    throughput = len(prompts) / max(elapsed, 1e-6)
    return {
        "benchmark": "elia_local_hardware_profile_v1",
        "hardware_profile": hardware,
        "samples": len(prompts),
        "elapsed_seconds": round(elapsed, 6),
        "observations_per_second": round(throughput, 3),
        "prediction_cache_limit": getattr(system, "_prediction_cache_limit", None),
        "status": "pass" if throughput > 50.0 else "limited_by_cpu",
        "warning": (
            "This is a local CPU throughput estimate for ELIA's bounded evidence pipeline. It does not imply GPT-class capability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, default=1, help="Number of independent seeds (>=2 reports aggregate statistics).")
    parser.add_argument("--examples-per-class", type=int, default=24)
    parser.add_argument(
        "--benchmark",
        choices=("continual", "generalization", "unseen-templates"),
        default="continual",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.benchmark == "generalization":
        report = run_generalization_benchmark(args.seed)
    elif args.benchmark == "unseen-templates":
        report = (
            run_unseen_template_benchmark(args.seed)
            if args.seeds == 1
            else run_unseen_template_sweep(args.seed, args.seeds)
        )
    elif args.seeds == 1:
        report = run_benchmark(args.seed, args.examples_per_class)
    else:
        report = run_sweep(args.seed, args.seeds, args.examples_per_class)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Saved benchmark report to {args.output.resolve()}")
    print(serialized)


if __name__ == "__main__":
    main()
