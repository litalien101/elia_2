from __future__ import annotations

import re

from .reasoning import ReasoningInputError, parse_literal


MAX_TEXT_LENGTH = 4_000
MAX_SEGMENTS = 32
_ENTITY = r"[A-Za-z][A-Za-z0-9_-]{0,31}"
_WORD = r"[A-Za-z][A-Za-z0-9_-]{0,31}"
_ENTITY_LIST = rf"{_ENTITY}(?:\s+and\s+{_ENTITY}){{0,2}}"
_SEGMENT = re.compile(r"[^.!?;\n]+[.!?;]?")
_RELATION = re.compile(
    rf"(?P<subject>{_ENTITY_LIST})\s+(?P<copula>is|are)\s+"
    rf"(?P<negation>not\s+)?(?:(?:a|an|the)\s+)?"
    rf"(?P<relation>{_WORD})\s+of\s+"
    rf"(?P<object>{_ENTITY_LIST})",
    re.IGNORECASE,
)
_VERB_FORMS = {
    form: lemma
    for lemma, forms in {
        "help": ("help", "helps"),
        "know": ("know", "knows"),
        "like": ("like", "likes"),
        "lead": ("lead", "leads"),
        "need": ("need", "needs"),
        "own": ("own", "owns"),
        "prefer": ("prefer", "prefers"),
        "support": ("support", "supports"),
        "trust": ("trust", "trusts"),
        "use": ("use", "uses"),
    }.items()
    for form in forms
}
_VERB = re.compile(
    rf"(?P<subject>{_ENTITY_LIST})\s+"
    rf"(?:(?:does\s+not|doesn't)\s+)?"
    rf"(?P<verb>{'|'.join(sorted(_VERB_FORMS, key=lambda value: (-len(value), value)))})"
    rf"\s+(?P<object>{_ENTITY_LIST})",
    re.IGNORECASE,
)
_BASE_VERBS = "|".join(
    sorted(
        (form for form, lemma in _VERB_FORMS.items() if form == lemma),
        key=lambda value: (-len(value), value),
    )
)
_SINGLE_VERB = "|".join(
    sorted(_VERB_FORMS, key=lambda value: (-len(value), value))
)
_COORDINATED_VERBS = re.compile(
    rf"(?P<subject>{_ENTITY})\s+"
    rf"(?P<first_negation>does\s+not|doesn't)?\s*"
    rf"(?P<first_verb>{_SINGLE_VERB})\s+(?P<first_object>{_ENTITY})\s+"
    rf"(?P<conjunction>and|but)\s+"
    rf"(?P<second_negation>does\s+not|doesn't)?\s*"
    rf"(?P<second_verb>{_SINGLE_VERB})\s+(?P<second_object>{_ENTITY})",
    re.IGNORECASE,
)
_PASSIVE_FORMS = {
    "helped": "help",
    "known": "know",
    "liked": "like",
    "led": "lead",
    "needed": "need",
    "owned": "own",
    "preferred": "prefer",
    "supported": "support",
    "trusted": "trust",
    "used": "use",
}
_PASSIVE = re.compile(
    rf"(?P<object>{_ENTITY_LIST})\s+(?P<auxiliary>is|are|was|were)\s+"
    rf"(?P<negation>not\s+)?"
    rf"(?P<verb>{'|'.join(sorted(_PASSIVE_FORMS, key=lambda value: (-len(value), value)))})"
    rf"\s+by\s+(?P<subject>{_ENTITY_LIST})",
    re.IGNORECASE,
)
_QUALITY = re.compile(
    rf"(?P<subject>{_ENTITY_LIST})\s+(?P<copula>is|are)\s+(?P<negation>not\s+)?"
    rf"(?:(?:very|quite|really|slightly|at\s+all)\s+)*"
    rf"(?P<quality>{_WORD})",
    re.IGNORECASE,
)
_QUALITY_COORDINATION = re.compile(
    rf"(?P<subject>{_ENTITY_LIST})\s+(?P<copula>is|are)\s+"
    rf"(?P<first_negation>not\s+)?"
    rf"(?:(?:very|quite|really|slightly|at\s+all)\s+)*"
    rf"(?P<first_quality>{_WORD})\s+and\s+"
    rf"(?P<second_negation>not\s+)?"
    rf"(?:(?:very|quite|really|slightly|at\s+all)\s+)*"
    rf"(?P<second_quality>{_WORD})",
    re.IGNORECASE,
)
_QUESTION_WHO_VERB = re.compile(
    rf"who\s+(?:(?:does\s+not|doesn't)\s+)?"
    rf"(?P<verb>{'|'.join(sorted(_VERB_FORMS, key=lambda value: (-len(value), value)))})"
    rf"\s+(?P<object>{_ENTITY})\??",
    re.IGNORECASE,
)
_QUESTION_WHO_OBJECT_VERB = re.compile(
    rf"(?P<wh>who|what)\s+does\s+(?P<subject>{_ENTITY})\s+"
    rf"(?P<negation>not\s+)?(?P<verb>{_BASE_VERBS})\??",
    re.IGNORECASE,
)
_QUESTION_DO_VERB = re.compile(
    rf"does\s+(?P<subject>{_ENTITY})\s+(?P<negation>not\s+)?"
    rf"(?P<verb>{_BASE_VERBS})"
    rf"\s+(?P<object>{_ENTITY})\??",
    re.IGNORECASE,
)
_QUESTION_WHY_VERB = re.compile(
    rf"why\s+does\s+(?P<subject>{_ENTITY})\s+(?P<negation>not\s+)?"
    rf"(?P<verb>{_BASE_VERBS})\s+(?P<object>{_ENTITY})\??",
    re.IGNORECASE,
)
_QUESTION_WHO_RELATION = re.compile(
    rf"who\s+is\s+(?P<negation>not\s+)?(?:(?:a|an|the)\s+)?"
    rf"(?P<relation>{_WORD})\s+of\s+(?P<object>{_ENTITY})\??",
    re.IGNORECASE,
)
_QUESTION_IS_RELATION = re.compile(
    rf"is\s+(?P<subject>{_ENTITY})\s+(?P<negation>not\s+)?"
    rf"(?:(?:a|an|the)\s+)?(?P<relation>{_WORD})\s+of\s+(?P<object>{_ENTITY})\??",
    re.IGNORECASE,
)
_QUESTION_WHY_RELATION = re.compile(
    rf"why\s+is\s+(?P<subject>{_ENTITY})\s+(?P<negation>not\s+)?"
    rf"(?:(?:a|an|the)\s+)?(?P<relation>{_WORD})\s+of\s+(?P<object>{_ENTITY})\??",
    re.IGNORECASE,
)
_QUESTION_IS_QUALITY = re.compile(
    rf"is\s+(?P<subject>{_ENTITY})\s+(?P<negation>not\s+)?"
    rf"(?:(?:very|quite|really|slightly|at\s+all)\s+)*"
    rf"(?P<quality>{_WORD})\??",
    re.IGNORECASE,
)
_QUESTION_WHY_QUALITY = re.compile(
    rf"why\s+is\s+(?P<subject>{_ENTITY})\s+(?P<negation>not\s+)?"
    rf"(?:(?:very|quite|really|slightly|at\s+all)\s+)*"
    rf"(?P<quality>{_WORD})\??",
    re.IGNORECASE,
)


def interpret_question(question: str) -> dict[str, str]:
    """Translate one supported simple-English question to a symbolic query."""
    if not isinstance(question, str) or not question.strip():
        raise ReasoningInputError("question must be a non-empty string")
    if len(question) > MAX_TEXT_LENGTH:
        raise ReasoningInputError(f"question exceeds {MAX_TEXT_LENGTH} characters")
    text = question.strip()
    template = ""
    match = _QUESTION_WHO_VERB.fullmatch(text)
    if match:
        predicate = _VERB_FORMS[match.group("verb").casefold()]
        negated = bool(re.search(r"\b(?:does\s+not|doesn't)\b", text, re.IGNORECASE))
        if negated and match.group("verb").casefold() != predicate:
            raise ReasoningInputError("Unsupported question grammar after 'does not'")
        query = f"{'not ' if negated else ''}{predicate}(?who, {_constant(match.group('object'))})"
        template = "wh_subject_transitive_verb"
    else:
        match = _QUESTION_WHO_OBJECT_VERB.fullmatch(text)
        if match:
            predicate = _VERB_FORMS[match.group("verb").casefold()]
            negated = bool(match.group("negation"))
            variable = "?who" if match.group("wh").casefold() == "who" else "?what"
            query = (
                f"{'not ' if negated else ''}{predicate}"
                f"({_constant(match.group('subject'))}, {variable})"
            )
            template = "wh_object_transitive_verb"
        else:
            match = _QUESTION_DO_VERB.fullmatch(text)
        if match and not template:
            predicate = _VERB_FORMS[match.group("verb").casefold()]
            negated = bool(match.group("negation"))
            query = (
                f"{'not ' if negated else ''}{predicate}"
                f"({_constant(match.group('subject'))}, {_constant(match.group('object'))})"
            )
            template = "yes_no_transitive_verb"
        elif not match:
            match = _QUESTION_WHY_VERB.fullmatch(text)
            if match:
                predicate = _VERB_FORMS[match.group("verb").casefold()]
                negated = bool(match.group("negation"))
                query = (
                    f"{'not ' if negated else ''}{predicate}"
                    f"({_constant(match.group('subject'))}, {_constant(match.group('object'))})"
                )
                template = "why_transitive_verb"
            else:
                match = _QUESTION_WHY_RELATION.fullmatch(text)
                if match:
                    negated = bool(match.group("negation"))
                    query = (
                        f"{'not ' if negated else ''}{_constant(match.group('relation'))}"
                        f"({_constant(match.group('subject'))}, {_constant(match.group('object'))})"
                    )
                    template = "why_relation"
            if not match:
                match = _QUESTION_WHY_QUALITY.fullmatch(text)
                if match:
                    negated = bool(match.group("negation"))
                    query = (
                        f"{'not ' if negated else ''}is_{_constant(match.group('quality'))}"
                        f"({_constant(match.group('subject'))})"
                    )
                    template = "why_quality"
            if not match:
                match = _QUESTION_WHO_RELATION.fullmatch(text)
            if match:
                if not template:
                    negated = bool(match.group("negation"))
                    query = (
                        f"{'not ' if negated else ''}{_constant(match.group('relation'))}"
                        f"(?who, {_constant(match.group('object'))})"
                    )
                    template = "wh_subject_relation"
            else:
                match = _QUESTION_IS_RELATION.fullmatch(text)
                if match:
                    negated = bool(match.group("negation"))
                    query = (
                        f"{'not ' if negated else ''}{_constant(match.group('relation'))}"
                        f"({_constant(match.group('subject'))}, {_constant(match.group('object'))})"
                    )
                    template = "yes_no_relation"
                else:
                    match = _QUESTION_IS_QUALITY.fullmatch(text)
                    if match:
                        negated = bool(match.group("negation"))
                        query = (
                            f"{'not ' if negated else ''}is_{_constant(match.group('quality'))}"
                            f"({_constant(match.group('subject'))})"
                        )
                        template = "yes_no_quality"
                    else:
                        raise ReasoningInputError(
                            "Unsupported question. Try 'Who helps Bob?', "
                            "'Who does Alice help?', "
                            "'Does Alice help Bob?', or 'Is Alice a parent of Bob?'"
                        )
    parse_literal(query)
    return {
        "question": text,
        "query": query,
        "template": template,
        "method": "bounded_english_question_pattern_v1",
    }


def _constant(text: str) -> str:
    return text.casefold().replace("-", "_")


def _entities(text: str) -> tuple[str, ...]:
    return tuple(_constant(part.strip()) for part in re.split(r"\s+and\s+", text, flags=re.IGNORECASE))


def interpret_text(text: str) -> dict[str, object]:
    """Propose atoms from a small explicit grammar without asserting or storing them."""
    if not isinstance(text, str) or not text.strip():
        raise ReasoningInputError("text must be a non-empty string")
    if len(text) > MAX_TEXT_LENGTH:
        raise ReasoningInputError(f"text exceeds {MAX_TEXT_LENGTH} characters")

    proposals: list[dict[str, object]] = []
    unsupported: list[dict[str, object]] = []
    segments = list(_SEGMENT.finditer(text))
    if len(segments) > MAX_SEGMENTS:
        raise ReasoningInputError(f"text contains more than {MAX_SEGMENTS} sentence segments")

    for segment in segments:
        source_text = segment.group().strip()
        if not source_text:
            continue
        leading_space = len(segment.group()) - len(segment.group().lstrip())
        start = segment.start() + leading_space
        end = start + len(source_text)
        candidate_text = source_text.rstrip(".!?;").strip()
        facts: list[str] = []
        template = ""

        match = _COORDINATED_VERBS.fullmatch(candidate_text)
        if match:
            subject = _constant(match.group("subject"))
            first_predicate = _VERB_FORMS[match.group("first_verb").casefold()]
            second_predicate = _VERB_FORMS[match.group("second_verb").casefold()]
            facts = [
                (
                    "not " if match.group("first_negation") else ""
                ) + f"{first_predicate}({subject}, {_constant(match.group('first_object'))})",
                (
                    "not " if match.group("second_negation") else ""
                ) + f"{second_predicate}({subject}, {_constant(match.group('second_object'))})",
            ]
            template = "shared_subject_coordination"

        if not match:
            match = _RELATION.fullmatch(candidate_text)
        if match:
            if template != "shared_subject_coordination":
                predicate = _constant(match.group("relation"))
                subjects = _entities(match.group("subject"))
                objects = _entities(match.group("object"))
                negated = bool(match.group("negation"))
                template = "copular_relation"
                if len(subjects) > 1 and match.group("copula").casefold() != "are":
                    match = None
        if not match:
            match = _PASSIVE.fullmatch(candidate_text)
            if match:
                predicate = _PASSIVE_FORMS[match.group("verb").casefold()]
                subjects = _entities(match.group("subject"))
                objects = _entities(match.group("object"))
                negated = bool(match.group("negation"))
                template = "passive_verb"
                auxiliary = match.group("auxiliary").casefold()
                plural_patient = len(objects) > 1
                plural_auxiliary = auxiliary in {"are", "were"}
                if plural_patient != plural_auxiliary:
                    match = None
        if not match:
            match = _VERB.fullmatch(candidate_text)
            if match:
                predicate = _VERB_FORMS[match.group("verb").casefold()]
                subjects = _entities(match.group("subject"))
                objects = _entities(match.group("object"))
                negated = bool(re.search(r"\b(?:does\s+not|doesn't)\b", candidate_text, re.IGNORECASE))
                template = "transitive_verb"
                verb_form = match.group("verb").casefold()
                if len(subjects) > 1 and verb_form != predicate:
                    match = None
        if not match:
            match = _QUALITY_COORDINATION.fullmatch(candidate_text)
            if match:
                subjects = _entities(match.group("subject"))
                template = "copular_quality_coordination"
                first_quality = _constant(match.group("first_quality"))
                second_quality = _constant(match.group("second_quality"))
                first_negated = bool(match.group("first_negation"))
                second_negated = bool(match.group("second_negation"))
                if (len(subjects) > 1) != (match.group("copula").casefold() == "are"):
                    match = None
                else:
                    facts = []
                    for subject in subjects:
                        facts.append(f"{'not ' if first_negated else ''}is_{first_quality}({subject})")
                        facts.append(f"{'not ' if second_negated else ''}is_{second_quality}({subject})")
        if not match:
            match = _QUALITY.fullmatch(candidate_text)
            if match:
                predicate = f"is_{_constant(match.group('quality'))}"
                subjects = _entities(match.group("subject"))
                objects = ()
                negated = bool(match.group("negation"))
                template = "copular_quality"
                if (len(subjects) > 1) != (match.group("copula").casefold() == "are"):
                    match = None
        if match:
            if template not in {"shared_subject_coordination", "copular_quality_coordination"}:
                arguments_list = (
                    [(subject, object_) for subject in subjects for object_ in objects]
                    if objects else [(subject,) for subject in subjects]
                )
                for arguments in arguments_list:
                    atom = f"{predicate}({', '.join(arguments)})"
                    facts.append(f"not {atom}" if negated else atom)
            try:
                for fact in facts:
                    parse_literal(fact)
            except ReasoningInputError:
                unsupported.append({
                    "text": source_text,
                    "start": start,
                    "end": end,
                    "reason": "normalized_terms_out_of_supported_range",
                })
                continue
            proposals.extend(
                {
                    "fact": fact,
                    "text": source_text,
                    "start": start,
                    "end": end,
                    "template": template,
                    "status": "review_required",
                }
                for fact in facts
            )
        else:
            unsupported.append({
                "text": source_text,
                "start": start,
                "end": end,
                "reason": "no_supported_pattern",
            })

    return {
        "proposals": proposals,
        "unsupported": unsupported,
        "stored": False,
        "requires_human_review": True,
        "method": "bounded_english_pattern_match_v2",
        "claim_boundary": (
            "Proposals are pattern matches, not verified facts or a general language "
            "interpretation. Review the source text before storing any fact."
        ),
        "supported_patterns": [
            "Subject is [not] relation of object",
            "One-to-three coordinated subjects/objects joined by 'and'",
            "Subject [does not] verb object (small supported verb list)",
            "Object is [not] past-participle by subject (small supported verb list)",
            "Subject is [not] [simple modifiers] quality",
        ],
    }
