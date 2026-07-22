"""The gold corpus: passages whose correct preparation is already known.

A reference-free benchmark can only measure how *much* a model changed a
passage, never whether it changed the right things. Measured that way, a model
that quietly turned "relational" into "non-relational" scored 99.7% lexical
retention — indistinguishable from the model that got everything right. The
only fix is ground truth, so each case here carries the passage, the edits it
needs, and the exact text those edits produce.

Two properties make that ground truth trustworthy, and both are enforced by
:func:`lint_case` rather than left to an author's care:

* A case is written in the form the pipeline would actually send. Normalization
  already repairs line-broken hyphens and collapses runs of spaces, so a case
  containing them would test text no model ever sees.
* A case's gold answer is reachable through the production applier — the same
  anchoring, the same per-edit size limit, the same retention floor. A gold
  edit the applier would refuse is not a gold edit; it is an unfair question.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from ..preparation import PreparationEdit, apply_edits, normalize_text
from ..preparation.adaptation.spans import sentence_spans
from ..preparation.segmentation import segment_text


CORPUS_SCHEMA_VERSION = 1
DEFAULT_CORPUS_DIR = Path(__file__).parent / "cases"

# Tiers answer different questions and are reported separately, because a model
# that scores well by never editing anything would otherwise hide behind the
# cases where doing nothing is correct.
TIERS = ("core", "noop", "trap", "robustness")

# The vocabulary an edit's `category` is drawn from, plus `no_edit` for the
# cases whose correct answer is an empty list. Kept closed so a typo cannot
# quietly create a category that is reported with a single case in it.
CATEGORIES = (
    "bibliographic_citation",
    "reference_marker",
    "visual_notation",
    "list_punctuation",
    "extraction_artifact",
    "no_edit",
)

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CorpusError(ValueError):
    """Raised when a case cannot be trusted as ground truth."""

    def __init__(self, case_id: str, issues: Sequence[str]) -> None:
        self.case_id = case_id
        self.issues = list(issues)
        super().__init__(
            f"Benchmark case '{case_id}' is not valid ground truth: " + "; ".join(issues)
        )


@dataclass(frozen=True)
class ExpectedEdit:
    """One change the passage requires, and every wording that satisfies it.

    ``accept`` exists because more than one answer is genuinely correct: "§5"
    may be spoken as "section 5" or "section five", and a benchmark that
    insisted on one of them would be measuring the model's luck. The first
    variant is canonical and is what ``BenchmarkCase.prepared`` contains.
    """

    anchor: str
    accept: tuple[str, ...]
    category: str
    why: str = ""
    # Filled in at load time by locating ``anchor`` in the source.
    start: int = -1
    end: int = -1
    sentence: int = 0

    def as_edit(self, replacement: str | None = None) -> PreparationEdit:
        """This expectation as the edit a perfect model would have returned."""

        return PreparationEdit(
            category=self.category,
            original=self.anchor,
            replacement=self.accept[0] if replacement is None else replacement,
            reason=self.why,
            sentence=self.sentence,
        )


@dataclass(frozen=True)
class Trap:
    """A region the passage dares a model to touch, and a name for the dare.

    Traps do not detect anything: any unrequested change is already caught by
    diffing the output against the source. They exist so that a failure reads
    as "changed a historical date" instead of "changed characters 214-218".
    """

    span: str
    label: str
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class BenchmarkCase:
    """One passage, the edits it needs, and the text those edits produce."""

    id: str
    tier: str
    categories: tuple[str, ...]
    source: str
    prepared: str
    expect: tuple[ExpectedEdit, ...] = ()
    traps: tuple[Trap, ...] = ()
    previous_context: str = ""
    following_context: str = ""
    chapter_title: str = "Chapter"
    notes: str = ""
    path: Path | None = field(default=None, compare=False)

    @property
    def is_noop(self) -> bool:
        return not self.expect

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "tier": self.tier,
            "categories": list(self.categories),
            "chapter_title": self.chapter_title,
            "source": self.source,
            "expect": [
                {
                    "anchor": item.anchor,
                    "accept": list(item.accept),
                    "category": item.category,
                    "why": item.why,
                }
                for item in self.expect
            ],
            "traps": [{"span": item.span, "label": item.label} for item in self.traps],
            "prepared": self.prepared,
        }
        if self.previous_context:
            payload["previous_context"] = self.previous_context
        if self.following_context:
            payload["following_context"] = self.following_context
        if self.notes:
            payload["notes"] = self.notes
        return payload


def _sentence_of(spans: Sequence[tuple[int, int]], start: int, end: int) -> int:
    """The 1-based numbered-view line an anchor sits in, or 0 if it straddles."""

    for number, (span_start, span_end) in enumerate(spans, start=1):
        if span_start <= start and end <= span_end:
            return number
    return 0


def _locate_unique(source: str, needle: str) -> tuple[int, int] | str:
    """Where ``needle`` sits in ``source``, or why it cannot be addressed."""

    if not needle:
        return "it is empty"
    occurrences = source.count(needle)
    if occurrences == 0:
        return "it does not occur in the source"
    if occurrences > 1:
        return f"it occurs {occurrences} times in the source, so it is ambiguous"
    index = source.index(needle)
    return (index, index + len(needle))


def case_from_dict(payload: dict[str, Any], *, path: Path | None = None) -> BenchmarkCase:
    """Read one case, resolving every anchor to a span in the source."""

    case_id = str(payload.get("id", path.stem if path else "unknown"))
    issues: list[str] = []
    source = str(payload.get("source", ""))
    spans = sentence_spans(source)

    expected: list[ExpectedEdit] = []
    for index, item in enumerate(payload.get("expect", [])):
        anchor = str(item.get("anchor", ""))
        accept = tuple(str(value) for value in item.get("accept", []))
        located = _locate_unique(source, anchor)
        if isinstance(located, str):
            issues.append(f"expect[{index}] anchor cannot be placed: {located}")
            continue
        start, end = located
        expected.append(
            ExpectedEdit(
                anchor=anchor,
                accept=accept,
                category=str(item.get("category", "unspecified")),
                why=str(item.get("why", "")),
                start=start,
                end=end,
                sentence=_sentence_of(spans, start, end),
            )
        )

    traps: list[Trap] = []
    for index, item in enumerate(payload.get("traps", [])):
        span = str(item.get("span", ""))
        located = _locate_unique(source, span)
        if isinstance(located, str):
            issues.append(f"traps[{index}] span cannot be placed: {located}")
            continue
        start, end = located
        traps.append(
            Trap(
                span=span,
                label=str(item.get("label", "unlabelled")),
                start=start,
                end=end,
            )
        )

    if issues:
        raise CorpusError(case_id, issues)

    return BenchmarkCase(
        id=case_id,
        tier=str(payload.get("tier", "core")),
        categories=tuple(str(value) for value in payload.get("categories", [])),
        source=source,
        prepared=str(payload.get("prepared", "")),
        expect=tuple(expected),
        traps=tuple(traps),
        previous_context=str(payload.get("previous_context", "")),
        following_context=str(payload.get("following_context", "")),
        chapter_title=str(payload.get("chapter_title", "Chapter")),
        notes=str(payload.get("notes", "")),
        path=path,
    )


def lint_case(case: BenchmarkCase) -> list[str]:
    """Every reason this case would be unfair or unusable, or an empty list.

    The expensive check is the last one: the gold edits are run through the
    production applier, which proves in one step that each anchor resolves,
    that no edit exceeds the per-edit size limit, that none empties a
    paragraph or crosses a paragraph break, that they do not overlap, that
    together they clear the retention floor — and that what comes out is
    exactly the ``prepared`` text this case claims.
    """

    issues: list[str] = []

    if not _ID_RE.fullmatch(case.id):
        issues.append(f"id '{case.id}' is not lowercase-kebab-case")
    if case.path is not None and case.path.stem != case.id:
        issues.append(f"id does not match filename '{case.path.stem}'")
    if case.tier not in TIERS:
        issues.append(f"tier '{case.tier}' is not one of {', '.join(TIERS)}")
    if case.tier == "trap" and not (case.expect and case.traps):
        # A trap case must carry real work as well as bait. Without the work, a
        # model that never edits anything would score full marks on the tier
        # built to catch over-editing.
        issues.append("a trap case needs at least one expected edit and one trap")
    if not case.categories:
        issues.append("categories is empty")
    for category in case.categories:
        if category not in CATEGORIES:
            issues.append(f"category '{category}' is not a known category")
    if not case.source.strip():
        issues.append("source is blank")
        return issues

    # Written as the pipeline would send it: anything normalization repairs is
    # work no model is ever asked to do.
    normalized = normalize_text(case.source)
    if normalized != case.source:
        issues.append(
            "source is not in normalized form; normalization would change it "
            "before any model saw it"
        )
    units = segment_text(case.source)
    if len(units) != 1 or units[0].kind != "prose":
        kinds = ", ".join(unit.kind for unit in units) or "nothing"
        issues.append(
            f"source does not segment to a single prose unit (got {kinds}), so "
            "a provider would never receive it as written"
        )

    for index, item in enumerate(case.expect):
        if not item.accept:
            issues.append(f"expect[{index}] lists no acceptable replacement")
        if item.category not in CATEGORIES or item.category == "no_edit":
            issues.append(f"expect[{index}] has category '{item.category}'")
        if item.sentence == 0:
            issues.append(
                f"expect[{index}] anchor straddles a sentence boundary, which "
                "the edit contract cannot express"
            )
    for first_index, first in enumerate(case.expect):
        for second in case.expect[first_index + 1 :]:
            if first.start < second.end and second.start < first.end:
                issues.append(f"expected edits '{first.anchor}' and '{second.anchor}' overlap")
    for trap in case.traps:
        for item in case.expect:
            if trap.start < item.end and item.start < trap.end:
                issues.append(
                    f"trap '{trap.label}' overlaps expected edit '{item.anchor}', "
                    "so the case asks for a change it also forbids"
                )

    if case.is_noop:
        if case.prepared != case.source:
            issues.append("a case with no expected edits must prepare to its source")
        if "no_edit" not in case.categories:
            issues.append("a case with no expected edits must be categorised no_edit")
        if case.tier not in ("noop", "robustness"):
            # Doing nothing is the right answer in both of these tiers. It is
            # never the right answer in `trap`, which is why that tier is
            # required to carry real work as well as bait.
            issues.append(
                f"a case with no expected edits belongs in the noop or "
                f"robustness tier, not '{case.tier}'"
            )
    else:
        if "no_edit" in case.categories:
            issues.append("a case with expected edits must not be categorised no_edit")
        if case.prepared == case.source:
            issues.append("prepared text is identical to the source")

    if issues:
        return issues

    # Each acceptable variant must be individually applicable: a variant the
    # applier would refuse would mark a correct model wrong.
    for index, item in enumerate(case.expect):
        for variant in item.accept:
            _text, applied, refusals = apply_edits(case.source, [item.as_edit(variant)])
            if refusals or len(applied) != 1:
                reason = refusals[0] if refusals else "it was silently dropped"
                issues.append(f"expect[{index}] variant {variant!r} is not applicable: {reason}")

    prepared, applied, refusals = apply_edits(case.source, [item.as_edit() for item in case.expect])
    if refusals:
        issues.extend(f"the gold edits are not all applicable: {item}" for item in refusals)
    if len(applied) != len(case.expect):
        issues.append(f"only {len(applied)} of {len(case.expect)} gold edits survived the applier")
    if prepared.strip() != case.prepared:
        issues.append(
            "applying the gold edits does not reproduce the prepared text; "
            f"expected {case.prepared!r} but got {prepared.strip()!r}"
        )
    return issues


def load_case(path: Path) -> BenchmarkCase:
    """Read and validate one case file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    case = case_from_dict(payload, path=path)
    issues = lint_case(case)
    if issues:
        raise CorpusError(case.id, issues)
    return case


def load_corpus(
    directory: Path | None = None,
    *,
    tiers: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    ids: Iterable[str] | None = None,
    limit_per_tier: int | None = None,
) -> list[BenchmarkCase]:
    """Every valid case in ``directory``, filtered and ordered by id.

    Validation is not optional here. A corpus is only worth running if every
    case in it is known-good, and a silently skipped case would quietly change
    what a score means between two runs.
    """

    directory = directory or DEFAULT_CORPUS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Benchmark corpus directory not found: {directory}")

    cases = [load_case(path) for path in sorted(directory.glob("*.json"))]
    if not cases:
        raise FileNotFoundError(f"Benchmark corpus is empty: {directory}")

    wanted_tiers = set(tiers) if tiers else None
    wanted_categories = set(categories) if categories else None
    wanted_ids = set(ids) if ids else None
    selected = [
        case
        for case in cases
        if (wanted_tiers is None or case.tier in wanted_tiers)
        and (wanted_categories is None or set(case.categories) & wanted_categories)
        and (wanted_ids is None or case.id in wanted_ids)
    ]
    if limit_per_tier is not None:
        # A smoke subset has to stay balanced across tiers: three citation cases
        # would tell you a provider is reachable and nothing about whether the
        # model leaves clean prose alone.
        seen: Counter[str] = Counter()
        kept: list[BenchmarkCase] = []
        for case in selected:
            if seen[case.tier] < limit_per_tier:
                seen[case.tier] += 1
                kept.append(case)
        selected = kept
    if not selected:
        raise ValueError("No benchmark cases matched the requested filters")
    return selected


__all__ = [
    "CATEGORIES",
    "CORPUS_SCHEMA_VERSION",
    "DEFAULT_CORPUS_DIR",
    "TIERS",
    "BenchmarkCase",
    "CorpusError",
    "ExpectedEdit",
    "Trap",
    "case_from_dict",
    "lint_case",
    "load_case",
    "load_corpus",
]
