# Design notes

Why the verdict vocabulary is three words, where the dependency line sits, and
what was rejected. The README says what the package does; this says why.

## Decision: three verdicts, not four, and the score always comes with them

`LANDED` / `MISSING` / `NOT_JUDGEABLE`.

The prototype had a fourth, `PARTIAL`, driven by word-coverage ≥ 0.5. It was
removed because it stacked a second arbitrary threshold on top of the first
(confidence ≥ 0.35), and a reader then has to learn how much to trust two
different numbers. The signal it carried survives as a **note** on a `MISSING`
result — the requested tag still did not land, and now the report says why that
might be misleading.

`NOT_JUDGEABLE` stays because it is different **in kind**. It is a structural
fact about the classifier's vocabulary, not a threshold decision: the model
cannot emit this label, so its absence is evidence of nothing. It does not fail
the gate, because failing on "I cannot tell" makes the tool unusable the first
time someone writes a phrase outside the tag space.

Every result carries its raw score. A tag that missed at 0.34 and one that
missed at 0.02 are different facts, and a binary throws that difference away.

## Decision: the judgeability fallback is deliberately generous

With a `vocabulary=` the verdict is exact — membership decides it. Without one
the fallback is length only, no character allowlist, because **wrongly calling
something unjudgeable hides a real miss**, and that is the more expensive
mistake.

Earlier versions enumerated allowed characters and kept being wrong. Real
danbooru tags carry colons (`re:zero kara hajimeru isekai seikatsu`), symbols
(`^_^`), and non-ASCII (`café`, CJK artist names). Every character an allowlist
forgets turns a real miss into a shrug.

The practical consequence, stated in the README: short prose is judged and
reported `MISSING` at 0.00 rather than shrugged off. Only conspicuous prose
trips the length fallback.

## Decision: three layers, one dependency seam

1. **core** — normalisation, verdicts, comparison. Plain dicts in and out;
   testable with no model, no GPU, no ComfyUI.
2. **workflow** — recovering requested tags from a PNG. Chunks are parsed by
   hand, so no Pillow.
3. **backends** — the only place an optional dependency could live. The shipped
   `ComfyTagger` speaks HTTP over `urllib`, so it costs nothing extra.

**"Zero dependencies" describes the Python package. It does not describe the
setup.** The shipped backend needs a running ComfyUI, the tagger custom node,
and several hundred megabytes of weights. The README says so rather than
implying it away.

Thresholding belongs in `compare()`, in the open, not inside whichever process
produced the numbers — which is why a backend returns `{tag: score}` and never
a pre-thresholded list.

## Decision: the exclusion list must be structural, visible, and replaceable

Rating words (`general`, `sensitive`, `explicit`, …) are excluded because
WD14-family taggers put ratings on a **separate output head** from general
tags. A rating word requested in a prompt would read as `MISSING` on every
image regardless of content — a guaranteed false failure, not an inconvenient
one.

Three properties keep this from becoming the hatch that empties the gate: the
report prints what it excluded, the reason is architectural rather than
editorial, and the list is a parameter you can replace. An opaque, ever-growing
exclusion list is exactly how a gate stops meaning anything.

## Decision: never pass because there was nothing to check

The script this grew from returned an empty list when a PNG had no embedded
workflow. The comparison then reported *"0 requested, 0 landed"* and exited 0.
A green light with no evidence behind it is worse than no gate, because nothing
downstream re-reads it.

Every such path is now an error — exit 3 at the CLI, `ValueError` from
`compare()`.

## Decision: do not guess which node holds the prompt

The prototype hardcoded node `1130`, which held because one person's one
workflow never changed. ComfyUI renumbers nodes on save and another graph has
different ids entirely.

Two honest options: name the node with `--node`, or walk back from the
sampler's `positive` input — which **refuses loudly** when the chain passes
through a node whose output is not in the file (a LoRA manager injecting
trigger words at run time, a prompt upsampler generating text with a language
model). Returning the half that is in the file would check a fraction of the
prompt and call it a pass.

## Decision: do not grade what was not requested

Detected-but-unrequested tags are printed, never scored. Failing on them would
mean inferring a prohibition from silence. If you want to forbid something, the
honest form is an explicit forbidden list, not an inference from what nobody
asked for.

## The positioning, and what must not be claimed

Discriminative per-attribute verification of text-to-image output is published
work — GenEval, NeurIPS 2023 — and the field moved **away** from it: GenEval 2
replaced the detector pipeline with a VQA method reporting better human
agreement.

So the claim is not accuracy. It is that **a classifier cannot narrate a false
positive**. A VLM asked whether an image contains `stirrup legwear` can answer
"yes, you can see the strap under the arch" about an image with no such thing —
specific, confident, invented. A classifier emits a wrong float. Both are
wrong; only one argues with you, and only one is the same wrong every run.

Do not claim: better accuracy than VLM judging, ground truth, or novelty.
Taggers are not accurate in any absolute sense — Camie-tagger reports 50.6%
macro-F1 across 70,527 tags — and the errors concentrate in exactly the
long-tail attributes a prompt is most likely to request.

The one genuinely different thing is a **domain** claim: in a booru-tag
pipeline the prompt is already written in the classifier's own label space, so
the comparison is exact — no question generation, no scene-graph parsing, no
paraphrase in the loop.

## Before you change anything

**Claims in the README are runnable.** The console example matches real output.
It has been wrong once already: the phrase it showed as `NOT_JUDGEABLE` was
short enough to be judged, so the real output said `MISSING` at 0.00.

**One known gap, honest and still open.** `backends.py`'s HTTP path has never
been exercised end to end — only `_parse` has unit coverage.

**`--vocabulary` refuses two files rather than loading them.** An empty one,
and a csv whose header did not name a `name` column. Both would produce a
vocabulary that matches nothing, and since an unjudgeable verdict does not fail
the gate, the run would exit 0 having judged none of what it was asked about —
the same vacuous pass that `compare()` raises on for an empty request. The
comma check is structural rather than a guess: a comma is the prompt separator,
so a tag cannot contain one.

**The compound-split note has misfired in both directions**, so change it
carefully. It once fabricated a split story for `dark blue eyes` against a
detected `dark blue background` (two shared words, unrelated tags), and it once
stayed silent on `twintails` against `twin tails` — the cleanest split there
is — because it was gated on multi-word requests.

**Zero third-party dependencies is a hard constraint.**

```console
python -I -c "
import sys; sys.path.insert(0, '.')
before = set(sys.modules)
import tagassert, tagassert.workflow, tagassert.backends
new = [m for m in set(sys.modules) - before
       if 'site-packages' in str(getattr(sys.modules[m], '__file__', '') or '')]
print('third-party:', new or 'none')"
```

**The default threshold and model are one person's calibration, not a sweep.**
A smaller `convnext` tagger missed `stirrup legwear` at every threshold tried
and labelled thigh-high socks as bare feet, while `eva02-large` agreed with a
human at 0.35. That is one careful comparison on one pipeline. Treat both as a
starting point and say so in your own docs.
