"""Did the tags you asked for actually land in the image?

A deterministic pass/fail gate for booru-tag image pipelines. You asked for
``stirrup legwear``; a multi-label classifier looked at the render and either
saw it or did not. Same image, same model, same threshold, same answer, every
time.

What this is
------------
A **regression gate**, not a benchmark. It answers "does this still produce
what it used to" cheaply and reproducibly. It does not tell you whether an
image is good, and it is not a claim to measure prompt adherence better than
anything else -- see ``Prior art`` in the README, which names the work that
came first and the direction the field actually moved.

The one honest advantage over asking a vision-language model is not accuracy.
It is that **a classifier cannot narrate a false positive.** Asked whether an
image contains ``stirrup legwear``, a VLM can answer "yes, you can see the
strap under the arch" about an image with no such thing -- a specific,
confident, invented detail. A classifier emits a wrong float. Both are wrong;
only one of them argues with you, and only one of them is the same wrong every
run.

Three verdicts, and the score always comes with them
----------------------------------------------------
``LANDED`` / ``MISSING`` / ``NOT_JUDGEABLE``. A fourth "partial" bucket was
considered and rejected: it stacked a second arbitrary threshold (word
coverage) on top of the first (confidence), and a reader then has to learn how
much to trust two different numbers. The signal it carried is kept as a *note*
on a ``MISSING`` result instead.

``NOT_JUDGEABLE`` is different in kind from the other two. It means the
requested phrase is not in the classifier's vocabulary at all, so its absence
from the output is evidence of nothing. That is a structural fact about the
model, not a threshold decision, which is why it is a verdict rather than a
failure.

Every result carries its raw score. A tag that missed at 0.34 and one that
missed at 0.02 are different facts, and a binary throws that difference away.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

__all__ = [
    "compare", "normalise", "judgeable", "Verdict", "TagResult",
    "LANDED", "MISSING", "NOT_JUDGEABLE",
    "DEFAULT_THRESHOLD", "STRUCTURAL_EXCLUSIONS",
]
__version__ = "0.1.0"

#: The classifier saw it at or above the threshold.
LANDED = "LANDED"

#: The classifier did not see it. The score says how close it came.
MISSING = "MISSING"

#: The phrase is not something this classifier can emit, so its absence is not
#: evidence. Free text, a phrase with no tag form, a vocabulary the model was
#: never trained on.
NOT_JUDGEABLE = "NOT_JUDGEABLE"

#: The prototype's default, kept for continuity and stated plainly for what it
#: is: one person's calibration against a handful of images, not a validated
#: sweep. Move it for your own pipeline and say so in your own docs.
DEFAULT_THRESHOLD = 0.35

#: Excluded for a *structural* reason, not for taste. WD14-family taggers put
#: rating words on a separate output head from general tags, so a rating word
#: requested in a prompt would read as MISSING on every image no matter what
#: the picture contains. Excluding them prevents a guaranteed false failure.
#:
#: Anything you add here is an assumption you are making, so the report prints
#: what it excluded even when it excludes it.
STRUCTURAL_EXCLUSIONS = frozenset({
    "general", "sensitive", "questionable", "explicit", "safe", "nsfw",
})

_WEIGHT = re.compile(r"^\((.*):\s*-?[\d.]+\s*\)$")
#: Deliberately permissive. Real danbooru tags carry colons
#: (``re:zero kara hajimeru isekai seikatsu``), ampersands, plus signs and
#: escaped parentheses, and copyright tags run long. Every character this
#: rejects turns a real miss into a shrug, which is the more expensive
#: mistake -- so it rejects almost nothing.
#: Anything without a control character and not starting with whitespace.
#: Earlier versions enumerated allowed characters and kept being wrong: real
#: danbooru tags carry colons (``re:zero kara hajimeru isekai seikatsu``),
#: symbols (``^_^``), and non-ASCII (``café``, CJK artist names). Every
#: character an allowlist forgets turns a real miss into a shrug, so shape is
#: judged by length alone and the vocabulary check is the real answer.
_TAGLIKE = re.compile(r"^[^\s\x00-\x1f][^\x00-\x1f]*$")

#: A prompt fragment longer than this is prose, not a label. Danbooru's
#: longest real tags are well inside it.
_MAX_TAG_WORDS = 10
_MAX_TAG_CHARS = 70


class TagResult(object):
    """One requested tag and what became of it."""

    __slots__ = ("tag", "verdict", "score", "note")

    def __init__(self, tag: str, verdict: str, score: Optional[float] = None,
                 note: str = ""):
        self.tag = tag
        self.verdict = verdict
        #: The classifier's confidence. ``None`` only when NOT_JUDGEABLE --
        #: there is no score for a label the model cannot emit.
        self.score = score
        #: Why this result might not mean what it looks like.
        self.note = note

    def __repr__(self) -> str:
        return "TagResult(%r, %s, score=%r)" % (self.tag, self.verdict,
                                                self.score)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TagResult):
            return NotImplemented
        return (self.tag, self.verdict, self.score) == (
            other.tag, other.verdict, other.score)


class Verdict(object):
    """The whole comparison. Falsy when any requested tag is missing."""

    __slots__ = ("results", "extra_detected", "excluded", "threshold")

    def __init__(self, results: Sequence[TagResult],
                 extra_detected: Sequence[str] = (),
                 excluded: Sequence[str] = (),
                 threshold: float = DEFAULT_THRESHOLD):
        self.results = list(results)
        #: Tags the classifier saw that nobody asked for. Reported, never
        #: scored: treating them as failures would mean inferring a
        #: prohibition from silence, which is a guess. If you want to forbid
        #: something, say so with a forbidden list.
        self.extra_detected = list(extra_detected)
        #: Requested tags dropped before judging, and why -- printed so the
        #: assumption is visible rather than quietly applied.
        self.excluded = list(excluded)
        self.threshold = threshold

    @property
    def missing(self) -> List[TagResult]:
        return [r for r in self.results if r.verdict == MISSING]

    @property
    def landed(self) -> List[TagResult]:
        return [r for r in self.results if r.verdict == LANDED]

    @property
    def not_judgeable(self) -> List[TagResult]:
        return [r for r in self.results if r.verdict == NOT_JUDGEABLE]

    @property
    def passed(self) -> bool:
        """True when nothing requested is missing.

        ``NOT_JUDGEABLE`` does not fail the gate. The classifier cannot speak
        to it either way, and failing on "I cannot tell" would make the gate
        unusable the first time someone writes a phrase outside the
        vocabulary.
        """
        return not self.missing

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        return "Verdict(landed=%d, missing=%d, not_judgeable=%d)" % (
            len(self.landed), len(self.missing), len(self.not_judgeable))


def normalise(tag: str) -> str:
    """One spelling for one tag.

    Danbooru writes ``long_hair``; prompts are typed ``long hair``; taggers
    emit either depending on the export. Left alone, the same tag fails to
    match itself and every comparison reports a false miss. Attention weights
    are stripped for the same reason.
    """
    t = str(tag).strip().lower()
    m = _WEIGHT.match(t)
    if m:
        t = m.group(1).strip()
    t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def judgeable(tag: str, vocabulary: Optional[Set[str]] = None) -> bool:
    """Can this classifier be asked about this phrase at all?

    With a ``vocabulary`` this is exact: in it or not. Without one, it falls
    back to shape -- and the fallback is deliberately generous, because
    wrongly calling something unjudgeable hides a real miss.
    """
    t = normalise(tag)
    if not t:
        return False
    if vocabulary is not None:
        return t in vocabulary
    if len(t) > _MAX_TAG_CHARS or t.count(" ") + 1 > _MAX_TAG_WORDS:
        return False
    return bool(_TAGLIKE.match(t))


def _tokens(tag: str) -> Set[str]:
    return {w for w in normalise(tag).split(" ") if len(w) > 2}


def _split_candidates(tag: str, detected: Dict[str, float]) -> List[str]:
    """Detected tags that plausibly *are* the requested one, spelled apart.

    Two rules, and both had to be tightened after they misfired:

    A requested tag is a candidate's compound only if every word of the
    request is present -- ``long black hair`` against ``black hair`` plus
    ``long hair``. An earlier version also accepted any two shared words,
    which produced ``missing as written, but 'dark blue background' was
    detected`` for a request of ``dark blue eyes``: a fabricated
    compound-split story about two unrelated tags that happen to share
    ``dark`` and ``blue``.

    And the gate on multi-word requests was wrong in the other direction.
    ``twintails`` against a detected ``twin tails`` is the cleanest split
    there is, and a single-word request was skipped entirely -- the exact
    case this note exists for, silently absent.
    """
    want = normalise(tag)
    squashed = want.replace(" ", "")
    parts = _tokens(want)

    exact = [d for d in detected
             if d != want and d.replace(" ", "") == squashed]
    if exact:
        return sorted(exact)[:3]

    if len(parts) < 2:
        return []
    covered = [d for d in detected
               if d != want and parts & _tokens(d) == parts]
    if covered:
        return sorted(covered)[:3]

    # Every word of the request accounted for across several detected tags,
    # each of which contributes at least one. Weaker than the above, and the
    # reason it is a note rather than a verdict.
    pieces = [d for d in detected if d != want and parts & _tokens(d)]
    if pieces and set().union(*(parts & _tokens(d) for d in pieces)) == parts:
        return sorted(pieces)[:3]
    return []


def compare(requested: Iterable[str],
            detected: Dict[str, float],
            threshold: float = DEFAULT_THRESHOLD,
            exclude: Iterable[str] = STRUCTURAL_EXCLUSIONS,
            vocabulary: Optional[Set[str]] = None) -> Verdict:
    """Judge each requested tag against what the classifier reported.

    ``detected`` maps tag to confidence -- the raw scores, not a
    pre-thresholded list. Thresholding is a decision and belongs here, in the
    open, not inside whichever backend produced the numbers.
    """
    # A bare string would iterate as characters and silently judge every
    # letter as its own tag.
    if isinstance(requested, str):
        requested = [t for t in requested.split(",") if t.strip()]
    requested = list(requested)
    if not any(normalise(r) for r in requested):
        raise ValueError(
            "nothing was requested, so there is nothing to check. An empty "
            "comparison that reports success is the failure this package "
            "exists to prevent, so it is an error rather than a pass.")

    det = {normalise(k): float(v) for k, v in dict(detected).items()}
    excl = {normalise(x) for x in exclude}

    seen_scores = {t: s for t, s in det.items() if s >= threshold}
    results: List[TagResult] = []
    excluded: List[str] = []
    asked: Set[str] = set()

    for raw in requested:
        tag = normalise(raw)
        if not tag or tag in asked:
            continue
        asked.add(tag)
        if tag in excl:
            excluded.append(tag)
            continue
        if not judgeable(tag, vocabulary):
            results.append(TagResult(
                tag, NOT_JUDGEABLE, None,
                "not a label this classifier can emit, so its absence is not "
                "evidence"))
            continue

        score = det.get(tag)
        if score is not None and score >= threshold:
            results.append(TagResult(tag, LANDED, score))
            continue

        # A multi-word tag can be missing as written while its parts are
        # plainly present -- taggers split compounds differently than prompts
        # do. That is worth saying, and it is not worth a fourth verdict: the
        # requested tag still did not land.
        note = ""
        near = _split_candidates(tag, seen_scores)
        if near:
            note = ("missing as written, but %s was detected -- the "
                    "classifier may split this compound differently"
                    % ", ".join(repr(n) for n in near))
        results.append(TagResult(tag, MISSING, score if score is not None
                                 else 0.0, note))

    extra = sorted(t for t in seen_scores
                   if t not in asked and t not in excl)
    return Verdict(results, extra, sorted(excluded), threshold)
