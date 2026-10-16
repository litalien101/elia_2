from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
import threading
from typing import Iterable


MAX_FACTS = 2_000
MAX_RULES = 256
MAX_ROUNDS = 256
MAX_WORK = 20_000
MAX_QUERY_ANSWERS = 100
MAX_PROOF_NODES = 256
MAX_PROOF_DEPTH = 128
MAX_ARGUMENTS = 8
MAX_RULE_PREMISES = 8
_PREDICATE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_TERM = re.compile(r"^(?:\?[a-z][a-z0-9_]{0,31}|[A-Za-z0-9_.:-]{1,80})$")
_ATOM = re.compile(r"^(not\s+)?([a-z][a-z0-9_]*)\s*\((.*)\)$", re.IGNORECASE)


class ReasoningInputError(ValueError):
    pass


class ReasoningLimitError(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class Literal:
    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False

    def opposite(self) -> "Literal":
        return Literal(self.predicate, self.arguments, not self.negated)

    def render(self) -> str:
        terms = ", ".join(self.arguments)
        atom = f"{self.predicate}({terms})"
        return f"not {atom}" if self.negated else atom


@dataclass(frozen=True)
class Rule:
    rule_id: str
    premises: tuple[Literal, ...]
    conclusion: Literal
    source_observation_id: str | None = None


@dataclass(frozen=True)
class Fact:
    fact_id: str
    literal: Literal
    source_observation_id: str


def parse_literal(text: str) -> Literal:
    if not isinstance(text, str) or len(text) > 256:
        raise ReasoningInputError("A fact/query must be a string of at most 256 characters")
    match = _ATOM.fullmatch(text.strip())
    if match is None:
        raise ReasoningInputError(
            "Use atom syntax such as parent(alice, bob), not parent(?x, ?y)"
        )
    negated, predicate, raw_arguments = match.groups()
    predicate = predicate.lower()
    if not _PREDICATE.fullmatch(predicate):
        raise ReasoningInputError("Predicate names must start with a letter and use letters, digits, or _")
    arguments = tuple(part.strip() for part in raw_arguments.split(",") if part.strip())
    if raw_arguments.strip() and len(arguments) != len(raw_arguments.split(",")):
        raise ReasoningInputError("Arguments must be non-empty and comma-separated")
    if len(arguments) > MAX_ARGUMENTS:
        raise ReasoningInputError(f"Atoms support at most {MAX_ARGUMENTS} arguments")
    if any(not _TERM.fullmatch(term) for term in arguments):
        raise ReasoningInputError(
            "Terms must be simple constants or variables such as ?person; quoted text is not supported"
        )
    return Literal(predicate, arguments, bool(negated))


def parse_rule(
    text: str,
    rule_id: str,
    source_observation_id: str | None = None,
) -> Rule:
    if not isinstance(text, str) or len(text) > 1_000:
        raise ReasoningInputError("A rule must be a string of at most 1000 characters")
    normalized = text.strip()
    if "=>" in normalized:
        normalized = normalized.replace("=>", "->")
    if normalized.count("->") != 1:
        raise ReasoningInputError("A rule must contain exactly one '->' or '=>' link")
    left, right = (part.strip() for part in normalized.split("->", 1))
    if not left or not right:
        raise ReasoningInputError("A rule needs premises and one conclusion")
    left = left.replace(" and ", " & ")
    raw_premises = [part.strip() for part in left.split("&")]
    if len(raw_premises) > MAX_RULE_PREMISES:
        raise ReasoningInputError(f"Rules support at most {MAX_RULE_PREMISES} premises")
    premises = tuple(parse_literal(part) for part in raw_premises)
    if any(premise.negated for premise in premises):
        raise ReasoningInputError(
            "Negated premises are not supported; use explicit negative facts and positive rules"
        )
    conclusion = parse_literal(right)
    premise_variables = {
        term
        for premise in premises
        for term in premise.arguments
        if term.startswith("?")
    }
    conclusion_variables = {
        term for term in conclusion.arguments if term.startswith("?")
    }
    if not conclusion_variables.issubset(premise_variables):
        raise ReasoningInputError(
            "Every variable in a conclusion must also appear in its premises"
        )
    return Rule(rule_id, premises, conclusion, source_observation_id)


def _unify(
    pattern: Literal,
    ground: Literal,
    initial: dict[str, str] | None = None,
) -> dict[str, str] | None:
    if (
        pattern.predicate != ground.predicate
        or pattern.negated != ground.negated
        or len(pattern.arguments) != len(ground.arguments)
    ):
        return None
    bindings = dict(initial or {})
    for expected, actual in zip(pattern.arguments, ground.arguments):
        if expected.startswith("?"):
            previous = bindings.get(expected)
            if previous is not None and previous != actual:
                return None
            bindings[expected] = actual
        elif expected != actual:
            return None
    return bindings


def _instantiate(literal: Literal, bindings: dict[str, str]) -> Literal:
    return Literal(
        literal.predicate,
        tuple(bindings.get(term, term) for term in literal.arguments),
        literal.negated,
    )


class BoundedRuleEngine:
    """Positive, range-restricted Horn reasoning with explicit proof provenance."""

    def __init__(
        self,
        facts: Iterable[Fact],
        rules: Iterable[Rule],
        *,
        max_facts: int = MAX_FACTS,
        max_rules: int = MAX_RULES,
        max_rounds: int = MAX_ROUNDS,
        max_work: int = MAX_WORK,
    ) -> None:
        self.facts = tuple(facts)
        self.rules = tuple(rules)
        self.max_facts = max_facts
        self.max_rules = max_rules
        self.max_rounds = max_rounds
        self.max_work = max_work
        self._work = 0
        self._known: dict[Literal, dict[str, object]] = {}
        self._proofs: dict[Literal, dict[str, object]] = {}
        self._candidate_cache: dict[
            tuple[Literal, tuple[tuple[str, str], ...]], list[Literal]
        ] = {}
        self._query_cache: dict[Literal, dict[str, object]] = {}
        self._index: dict[tuple[str, bool, int], list[Literal]] = defaultdict(list)
        self._position_index: dict[
            tuple[str, bool, int, int, str], list[Literal]
        ] = defaultdict(list)
        if len(self.facts) > max_facts:
            raise ReasoningLimitError(f"Fact limit exceeded ({max_facts})")
        if len(self.rules) > max_rules:
            raise ReasoningLimitError(f"Rule limit exceeded ({max_rules})")
        for fact in self.facts:
            if fact.literal in self._known:
                continue
            self._known[fact.literal] = {
                "kind": "source_fact",
                "fact_id": fact.fact_id,
                "source_observation_id": fact.source_observation_id,
                "literal": fact.literal.render(),
            }
            self._index_literal(fact.literal)
        self._close()
        self._closure_work = self._work

    @property
    def derived_count(self) -> int:
        return sum(
            1 for proof in self._proofs.values()
            if proof.get("kind") == "rule"
        )

    def _tick(self) -> None:
        self._work += 1
        if self._work > self.max_work:
            raise ReasoningLimitError(f"Reasoning work limit exceeded ({self.max_work})")

    def _index_literal(self, literal: Literal) -> None:
        self._index[(
            literal.predicate, literal.negated, len(literal.arguments)
        )].append(literal)
        for position, value in enumerate(literal.arguments):
            self._position_index[(
                literal.predicate, literal.negated, len(literal.arguments),
                position, value,
            )].append(literal)

    def _candidates(
        self,
        pattern: Literal,
        bindings: dict[str, str] | None = None,
    ) -> list[Literal]:
        binding_key = tuple(sorted((key, value) for key, value in (bindings or {}).items()))
        cache_key = (pattern, binding_key)
        cached = self._candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        signature = (pattern.predicate, pattern.negated, len(pattern.arguments))
        candidates = self._index.get(signature, [])
        choices = [candidates]
        for position, term in enumerate(pattern.arguments):
            value = bindings.get(term) if bindings and term.startswith("?") else None
            if not term.startswith("?"):
                value = term
            if value is not None:
                choices.append(self._position_index.get((*signature, position, value), []))
        result = min(choices, key=len)
        self._candidate_cache[cache_key] = result
        return result

    def _match_premises(
        self,
        rule: Rule,
        delta_by_signature: dict[tuple[str, bool, int], tuple[Literal, ...]],
    ):
        def visit(
            remaining: tuple[int, ...],
            bindings: dict[str, str],
            proofs: dict[int, Literal],
        ):
            if not remaining:
                yield bindings, [proofs[index] for index in range(len(rule.premises))]
                return
            index = min(
                remaining,
                key=lambda candidate: len(
                    self._candidates(rule.premises[candidate], bindings)
                ),
            )
            premise = rule.premises[index]
            for known_literal in self._candidates(premise, bindings):
                self._tick()
                matched = _unify(premise, known_literal, bindings)
                if matched is not None:
                    yield from visit(
                        tuple(candidate for candidate in remaining if candidate != index),
                        matched,
                        {**proofs, index: known_literal},
                    )

        for delta_index, premise in enumerate(rule.premises):
            signature = (
                premise.predicate,
                premise.negated,
                len(premise.arguments),
            )
            for known_literal in delta_by_signature.get(signature, ()):
                self._tick()
                bindings = _unify(premise, known_literal)
                if bindings is not None:
                    yield from visit(
                        tuple(
                            index for index in range(len(rule.premises))
                            if index != delta_index
                        ),
                        bindings,
                        {delta_index: known_literal},
                    )

    def _proof_for(self, literal: Literal) -> dict[str, object]:
        proof = self._proofs.get(literal) or self._known.get(literal)
        if proof is None:
            raise RuntimeError("Internal reasoning error: proof not found for known literal")
        return proof

    @staticmethod
    def _proof_depth(node: dict[str, object] | None) -> int:
        if not node or node.get("kind") == "source_fact":
            return 0
        premises = node.get("premises", ())
        if not premises:
            return 1
        return 1 + max(BoundedRuleEngine._proof_depth(candidate.get("proof")) for candidate in premises if isinstance(candidate, dict) and isinstance(candidate.get("proof"), dict))

    def _explain(
        self,
        literal: Literal,
        remaining: list[int],
        depth: int = 0,
    ) -> dict[str, object]:
        if remaining[0] <= 0 or depth >= MAX_PROOF_DEPTH:
            return {"kind": "truncated", "literal": literal.render()}
        remaining[0] -= 1
        proof = self._proof_for(literal)
        if proof.get("kind") == "source_fact":
            return dict(proof)
        premises = proof.get("premises", [])
        assert isinstance(premises, tuple)
        return {
            "kind": "rule",
            "rule_id": proof["rule_id"],
            "rule": proof["rule"],
            "source_observation_id": proof["source_observation_id"],
            "premises": [
                {
                    "literal": premise.render(),
                    "proof": self._explain(premise, remaining, depth + 1),
                }
                for premise in premises
            ],
        }

    def _close(self) -> None:
        delta = tuple(self._known)
        for _ in range(self.max_rounds):
            delta_by_signature: dict[
                tuple[str, bool, int], list[Literal]
            ] = defaultdict(list)
            for literal in delta:
                delta_by_signature[(
                    literal.predicate,
                    literal.negated,
                    len(literal.arguments),
                )].append(literal)
            indexed_delta = {
                signature: tuple(literals)
                for signature, literals in delta_by_signature.items()
            }
            additions: dict[Literal, dict[str, object]] = {}
            for rule in self.rules:
                for bindings, matched_premises in self._match_premises(
                    rule,
                    indexed_delta,
                ):
                    conclusion = _instantiate(rule.conclusion, bindings)
                    if conclusion in self._known or conclusion in additions:
                        continue
                    additions[conclusion] = {
                        "kind": "rule",
                        "rule_id": rule.rule_id,
                        "source_observation_id": rule.source_observation_id,
                        "rule": " & ".join(p.render() for p in rule.premises)
                        + " -> " + rule.conclusion.render(),
                        "premises": tuple(matched_premises),
                    }
                    self._tick()
                    if len(self._known) + len(additions) > self.max_facts:
                        raise ReasoningLimitError(
                            f"Derived fact limit exceeded ({self.max_facts})"
                        )
            if not additions:
                return
            for literal, proof in additions.items():
                self._known[literal] = proof
                self._proofs[literal] = proof
                self._index_literal(literal)
            delta = tuple(additions)
        if delta:
            raise ReasoningLimitError(f"Reasoning round limit exceeded ({self.max_rounds})")

    def query(self, query: Literal) -> dict[str, object]:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        self._work = self._closure_work
        positive_query = Literal(query.predicate, query.arguments, False)
        negative_query = positive_query.opposite()
        positive: dict[tuple[tuple[str, str], ...], tuple[Literal, dict[str, object]]] = {}
        negative: dict[tuple[tuple[str, str], ...], tuple[Literal, dict[str, object]]] = {}
        for target, matches in (
            (positive_query, positive),
            (negative_query, negative),
        ):
            for literal in self._candidates(target):
                self._tick()
                bindings = _unify(target, literal)
                if bindings is not None:
                    key = tuple(sorted(bindings.items()))
                    matches.setdefault(key, (literal, self._proof_for(literal)))
        keys = sorted(set(positive) | set(negative))
        if len(keys) > MAX_QUERY_ANSWERS:
            raise ReasoningLimitError(
                f"Query answer limit exceeded ({MAX_QUERY_ANSWERS})"
            )
        answers = []
        proof_budget = [MAX_PROOF_NODES]
        for key in keys:
            has_positive = key in positive
            has_negative = key in negative
            state = "conflict" if has_positive and has_negative else (
                "supported" if has_positive else "refuted"
            )
            supporting = negative if query.negated else positive
            refuting = positive if query.negated else negative
            has_support = key in supporting
            has_refutation = key in refuting
            state = "conflict" if has_support and has_refutation else (
                "supported" if has_support else "refuted"
            )
            answer: dict[str, object] = {
                "bindings": dict(key),
                "state": state,
            }
            if has_support:
                answer["proof"] = self._explain(supporting[key][0], proof_budget)
            if has_refutation:
                answer["counterproof"] = self._explain(refuting[key][0], proof_budget)
            answers.append(answer)
        answer_states = {str(row["state"]) for row in answers}
        state = "conflict" if "conflict" in answer_states else (
            "mixed" if {"supported", "refuted"}.issubset(answer_states)
            else "supported" if "supported" in answer_states
            else "refuted" if "refuted" in answer_states
            else "unknown"
        )
        result = {
            "query": query.render(),
            "state": state,
            "answers": answers,
            "facts_considered": len(self._known),
            "derived_facts": self.derived_count,
            "rules_considered": len(self.rules),
            "work_used": self._work,
            "limits": {
                "facts": self.max_facts,
                "rules": self.max_rules,
                "rounds": self.max_rounds,
                "work": self.max_work,
                "answers": MAX_QUERY_ANSWERS,
                "proof_nodes": MAX_PROOF_NODES,
                "proof_depth": MAX_PROOF_DEPTH,
            },
            "method": "bounded_forward_chaining_positive_horn_rules",
            "claim_boundary": (
                "Conclusions follow only from supplied facts and rules. Unknown means "
                "not derivable here; absence is not proof of falsehood."
            ),
        }
        self._query_cache[query] = result
        return result
