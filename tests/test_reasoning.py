from __future__ import annotations

import random
import unittest

from elia.cognition.reasoning import (
    BoundedRuleEngine,
    Fact,
    ReasoningInputError,
    ReasoningLimitError,
    parse_literal,
    parse_rule,
)


def fact(text: str, fact_id: str) -> Fact:
    return Fact(fact_id, parse_literal(text), f"observation_{fact_id}")


def rule(text: str, rule_id: str):
    return parse_rule(text, rule_id)


class SymbolicReasoningTests(unittest.TestCase):
    def test_chains_two_rules_and_returns_a_source_linked_proof(self):
        engine = BoundedRuleEngine(
            [fact("parent(alice, bob)", "f1"), fact("parent(bob, cara)", "f2")],
            [
                rule("parent(?x, ?y) & parent(?y, ?z) -> grandparent(?x, ?z)", "r1"),
                rule("grandparent(?x, ?y) -> ancestor(?x, ?y)", "r2"),
            ],
        )
        result = engine.query(parse_literal("ancestor(alice, ?who)"))

        self.assertEqual(result["state"], "supported")
        self.assertEqual(result["answers"][0]["bindings"], {"?who": "cara"})
        self.assertEqual(result["derived_facts"], 2)
        self.assertEqual(result["answers"][0]["proof"]["rule_id"], "r2")
        self.assertEqual(
            result["answers"][0]["proof"]["premises"][0]["proof"]["rule_id"],
            "r1",
        )

    def test_repeated_variables_enforce_identity(self):
        engine = BoundedRuleEngine([
            fact("connected(alice, bob)", "f1"),
            fact("connected(cara, cara)", "f2"),
        ], [])
        result = engine.query(parse_literal("connected(?person, ?person)"))
        self.assertEqual(result["answers"][0]["bindings"], {"?person": "cara"})

    def test_recursive_transitive_closure_terminates_on_cycles(self):
        engine = BoundedRuleEngine(
            [
                fact("edge(a, b)", "f1"),
                fact("edge(b, c)", "f2"),
                fact("edge(c, a)", "f3"),
            ],
            [
                rule("edge(?x, ?y) -> path(?x, ?y)", "r1"),
                rule("path(?x, ?y) & edge(?y, ?z) -> path(?x, ?z)", "r2"),
            ],
        )
        answer = engine.query(parse_literal("path(a, ?destination)"))
        self.assertEqual(
            {row["bindings"]["?destination"] for row in answer["answers"]},
            {"a", "b", "c"},
        )
        self.assertLessEqual(answer["facts_considered"], 12)

    def test_incremental_closure_handles_long_dependency_chains_with_bounded_work(self):
        length = 80
        facts = [
            fact(f"edge(n{index}, n{index + 1})", f"edge{index}")
            for index in range(length)
        ]
        facts.append(fact(f"reachable(n{length})", "destination"))
        engine = BoundedRuleEngine(
            facts,
            [
                rule(
                    "edge(?source, ?destination) & reachable(?destination) "
                    "-> reachable(?source)",
                    "reverse_reachability",
                ),
            ],
        )

        result = engine.query(parse_literal("reachable(n0)"))

        self.assertEqual(result["state"], "supported")
        self.assertEqual(result["derived_facts"], length)
        self.assertLess(result["work_used"], 1_000)
        proof = result["answers"][0]["proof"]
        for _ in range(length):
            self.assertEqual(proof["kind"], "rule")
            proof = next(
                premise["proof"]
                for premise in proof["premises"]
                if premise["literal"].startswith("reachable(")
            )
        self.assertEqual(proof["kind"], "source_fact")

    def test_generated_graphs_match_independent_transitive_closure_oracle(self):
        for seed in range(5):
            rng = random.Random(seed)
            edges = {
                (f"n{left}", f"n{right}")
                for left in range(6)
                for right in range(6)
                if rng.random() < 0.22
            }
            expected = set(edges)
            changed = True
            while changed:
                additions = {
                    (source, target)
                    for source, middle in expected
                    for next_middle, target in edges
                    if middle == next_middle
                } - expected
                changed = bool(additions)
                expected.update(additions)

            engine = BoundedRuleEngine(
                [
                    fact(f"edge({source}, {target})", f"e{index}")
                    for index, (source, target) in enumerate(sorted(edges))
                ],
                [
                    rule("edge(?x, ?y) -> path(?x, ?y)", "direct"),
                    rule("path(?x, ?y) & edge(?y, ?z) -> path(?x, ?z)", "extend"),
                ],
            )
            actual = {
                (source, answer["bindings"]["?destination"])
                for source in (f"n{index}" for index in range(6))
                for answer in engine.query(
                    parse_literal(f"path({source}, ?destination)")
                )["answers"]
            }
            self.assertEqual(actual, expected)

    def test_explicit_negative_evidence_is_distinct_from_unknown(self):
        engine = BoundedRuleEngine([
            fact("safe(alice)", "f1"),
            fact("not safe(bob)", "f2"),
        ], [])
        self.assertEqual(engine.query(parse_literal("safe(alice)"))["state"], "supported")
        self.assertEqual(engine.query(parse_literal("safe(bob)"))["state"], "refuted")
        self.assertEqual(engine.query(parse_literal("safe(cara)"))["state"], "unknown")
        self.assertEqual(engine.query(parse_literal("not safe(bob)"))["state"], "supported")
        self.assertEqual(engine.query(parse_literal("safe(?person)"))["state"], "mixed")

    def test_conflicting_facts_are_reported_not_silently_resolved(self):
        engine = BoundedRuleEngine([
            fact("eligible(alice)", "f1"),
            fact("not eligible(alice)", "f2"),
        ], [])
        answer = engine.query(parse_literal("eligible(alice)"))
        self.assertEqual(answer["state"], "conflict")
        self.assertIn("proof", answer["answers"][0])
        self.assertIn("counterproof", answer["answers"][0])

    def test_rejects_unsafe_rules_and_negation_as_absence(self):
        with self.assertRaises(ReasoningInputError):
            rule("person(?x) -> likes(?x, ?y)", "unsafe")
        with self.assertRaises(ReasoningInputError):
            rule("person(?x) & not banned(?x) -> eligible(?x)", "default-negation")

    def test_indexed_matching_avoids_scanning_irrelevant_predicates(self):
        facts = [
            fact(f"unrelated_{index}(item{index})", f"noise{index}")
            for index in range(1_000)
        ]
        facts.append(fact("seed(alice)", "seed"))
        engine = BoundedRuleEngine(
            facts,
            [rule("seed(?x) -> found(?x)", "r1")],
        )
        result = engine.query(parse_literal("found(alice)"))
        self.assertEqual(result["state"], "supported")
        self.assertLess(result["work_used"], 20)

    def test_resource_limits_fail_explicitly_instead_of_returning_partial_answers(self):
        with self.assertRaises(ReasoningLimitError):
            BoundedRuleEngine(
                [fact("start(a)", "f1")],
                [
                    rule("start(?x) -> step1(?x)", "r1"),
                    rule("step1(?x) -> step2(?x)", "r2"),
                    rule("step2(?x) -> step3(?x)", "r3"),
                ],
                max_rounds=1,
            )
        with self.assertRaises(ReasoningLimitError):
            BoundedRuleEngine(
                [fact("p(a)", "f1")],
                [rule("p(?x) -> q(?x)", "r1")],
                max_work=0,
            )
        engine = BoundedRuleEngine([
            fact(f"member(item{index})", f"member{index}")
            for index in range(101)
        ], [])
        with self.assertRaises(ReasoningLimitError):
            engine.query(parse_literal("member(?item)"))
        with self.assertRaises(ReasoningLimitError):
            BoundedRuleEngine(
                [fact("p(a)", "f1"), fact("p(b)", "f2")],
                [],
                max_facts=1,
            )
        with self.assertRaises(ReasoningLimitError):
            BoundedRuleEngine(
                [],
                [rule("p(a) -> q(a)", "r1"), rule("p(b) -> q(b)", "r2")],
                max_rules=1,
            )


if __name__ == "__main__":
    unittest.main()
