from __future__ import annotations

import unittest

from elia.cognition.language import interpret_question, interpret_text
from elia.cognition.reasoning import ReasoningInputError


class BoundedLanguageInterpretationTests(unittest.TestCase):
    def test_interprets_supported_wh_and_yes_no_questions(self):
        examples = {
            "Who helps Bob?": ("help(?who, bob)", "wh_subject_transitive_verb"),
            "Who does not help Bob?": ("not help(?who, bob)", "wh_subject_transitive_verb"),
            "Who does Alice help?": ("help(alice, ?who)", "wh_object_transitive_verb"),
            "What does Alice use?": ("use(alice, ?what)", "wh_object_transitive_verb"),
            "Who does Alice not help?": (
                "not help(alice, ?who)", "wh_object_transitive_verb"
            ),
            "Does Alice help Bob?": ("help(alice, bob)", "yes_no_transitive_verb"),
            "Does Alice not help Bob?": ("not help(alice, bob)", "yes_no_transitive_verb"),
            "Why does Alice help Bob?": ("help(alice, bob)", "why_transitive_verb"),
            "Why does Alice not help Bob?": (
                "not help(alice, bob)", "why_transitive_verb"
            ),
            "Why is Alice a parent of Bob?": ("parent(alice, bob)", "why_relation"),
            "Why is Alice not helpful?": ("not is_helpful(alice)", "why_quality"),
            "Who is a parent of Bob?": ("parent(?who, bob)", "wh_subject_relation"),
            "Is Alice the parent of Bob?": ("parent(alice, bob)", "yes_no_relation"),
            "Is Alice not really helpful?": ("not is_helpful(alice)", "yes_no_quality"),
        }

        for question, expected in examples.items():
            with self.subTest(question=question):
                result = interpret_question(question)
                self.assertEqual((result["query"], result["template"]), expected)
                self.assertEqual(result["question"], question)

    def test_abstains_on_unsupported_questions_and_invalid_input(self):
        for question in (
            "Who might help Bob?",
            "Who does Alice helps?",
            "Who does not helps Bob?",
            "Does Alice helps Bob?",
            " ",
        ):
            with self.subTest(question=question), self.assertRaises(ReasoningInputError):
                interpret_question(question)
        with self.assertRaises(ReasoningInputError):
            interpret_question("x" * 4_001)

    def test_extracts_reviewable_relations_and_keeps_source_spans(self):
        text = "  Alice is parent of Bob.  Bob helps Cara!"

        result = interpret_text(text)

        self.assertEqual(
            [proposal["fact"] for proposal in result["proposals"]],
            ["parent(alice, bob)", "help(bob, cara)"],
        )
        self.assertEqual(
            [text[proposal["start"]:proposal["end"]] for proposal in result["proposals"]],
            ["Alice is parent of Bob.", "Bob helps Cara!"],
        )
        self.assertTrue(all(
            proposal["status"] == "review_required"
            for proposal in result["proposals"]
        ))
        self.assertFalse(result["stored"])

    def test_preserves_explicit_negative_polarity(self):
        result = interpret_text(
            "Alice is not the parent of Bob. Bob does not help Cara. "
            "Dana is not really available."
        )

        self.assertEqual(
            [proposal["fact"] for proposal in result["proposals"]],
            ["not parent(alice, bob)", "not help(bob, cara)", "not is_available(dana)"],
        )

    def test_expands_small_and_coordinations_without_losing_provenance(self):
        text = "Alice and Bob like Cara and Dana."

        result = interpret_text(text)

        self.assertEqual(
            [proposal["fact"] for proposal in result["proposals"]],
            [
                "like(alice, cara)",
                "like(alice, dana)",
                "like(bob, cara)",
                "like(bob, dana)",
            ],
        )
        self.assertTrue(all(
            text[proposal["start"]:proposal["end"]] == text
            for proposal in result["proposals"]
        ))

    def test_expands_quality_coordination_with_held_out_pairs(self):
        result = interpret_text("Alice and Bob are helpful and available.")
        self.assertEqual(
            [proposal["fact"] for proposal in result["proposals"]],
            [
                "is_helpful(alice)",
                "is_available(alice)",
                "is_helpful(bob)",
                "is_available(bob)",
            ],
        )

    def test_normalizes_supported_passive_voice_to_active_relation(self):
        result = interpret_text("Bob and Cara were helped by Alice.")

        self.assertEqual(
            [proposal["fact"] for proposal in result["proposals"]],
            ["help(alice, bob)", "help(alice, cara)"],
        )
        self.assertEqual(result["proposals"][0]["template"], "passive_verb")

    def test_abstains_on_or_conjunction_and_ungrammatical_agreement(self):
        result = interpret_text(
            "Alice or Bob likes Cara. Alice are helpful. Bob and Cara is helped by Dana."
        )

        self.assertEqual(result["proposals"], [])
        self.assertEqual(len(result["unsupported"]), 3)

    def test_extracts_shared_subject_predicates_and_respects_independent_negation(self):
        result = interpret_text("Alice likes Bob and does not help Cara.")

        self.assertEqual(
            [proposal["fact"] for proposal in result["proposals"]],
            ["like(alice, bob)", "not help(alice, cara)"],
        )
        self.assertEqual(
            {proposal["template"] for proposal in result["proposals"]},
            {"shared_subject_coordination"},
        )

    def test_abstains_on_unrecognized_language_without_inventing_facts(self):
        result = interpret_text(
            "Because Alice might help Bob, everyone should agree. "
            "The meeting was moved yesterday."
        )

        self.assertEqual(result["proposals"], [])
        self.assertEqual(len(result["unsupported"]), 2)
        self.assertTrue(all(
            row["reason"] == "no_supported_pattern"
            for row in result["unsupported"]
        ))

    def test_rejects_empty_oversized_and_excessive_segment_input(self):
        for text in (" ", "x" * 4_001, ".".join("Alice likes Bob" for _ in range(33))):
            with self.subTest(length=len(text)), self.assertRaises(ReasoningInputError):
                interpret_text(text)


if __name__ == "__main__":
    unittest.main()
