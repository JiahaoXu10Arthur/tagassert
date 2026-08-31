# tagassert

[![test](https://github.com/JiahaoXu10Arthur/tagassert/actions/workflows/test.yml/badge.svg)](https://github.com/JiahaoXu10Arthur/tagassert/actions/workflows/test.yml)

Did the tags you asked for actually land in the image?

You asked for `stirrup legwear`. A multi-label classifier looked at the render
and either saw it or did not. Same image, same model, same threshold, same
answer, every time.

```console
$ tagassert check render_00042.png
render_00042.png
  ok    0.99  1girl
  ok    0.96  solo
  MISS  0.11  stirrup legwear
  -        -  a soft painterly atmosphere over rolling hills at golden hour with mist
            not a label this classifier can emit, so its absence is not evidence
  excluded (structural): sensitive
  also detected, not requested: barefoot, indoors, sitting

  4 requested: 2 landed, 1 missing, 1 not judgeable (threshold 0.35)
$ echo $?
1
```

```python
from tagassert import compare
from tagassert.workflow import requested_tags

v = compare(requested_tags("render.png"), detected)   # detected: {tag: score}
if not v:
    print("missing:", [r.tag for r in v.missing])
```

No dependencies, stdlib only. 38 tests, CI on Python 3.9, 3.11 and 3.13.

## What this is, and is not

It is a **regression gate**: does this pipeline still produce what it used to,
answered cheaply, offline, and identically on every run. Exit 0 means every
requested tag landed.

It is **not** a semantic-accuracy benchmark, and it is not a claim to measure
prompt adherence better than anything else. See [Prior art](#prior-art) — that
ground is occupied, and the field has moved in the opposite direction from the
method used here.

The honest advantage over asking a vision-language model is not accuracy. It
is that **a classifier cannot narrate a false positive.** Asked whether an
image contains `stirrup legwear`, a VLM can answer *"yes, you can see the strap
under the arch"* about an image containing no such thing — a specific,
confident, invented detail that reads as evidence. A classifier emits a wrong
float. Both are wrong; only one of them argues with you, and only one is the
same wrong every run.

Taggers are not accurate in any absolute sense. Camie-tagger reports 67.3%
micro-F1 and 50.6% macro-F1 across 70,527 tags, and the errors concentrate in
exactly the long-tail specific attributes a prompt is most likely to ask for.
What you get is determinism, not truth.

## Three verdicts, and the score always comes with them

| verdict | meaning |
|---|---|
| `LANDED` | detected at or above the threshold |
| `MISSING` | not detected — the score says how close it came |
| `NOT_JUDGEABLE` | not a label this classifier can emit, so its absence is not evidence |

`NOT_JUDGEABLE` is different in kind. It is a structural fact about the model's
vocabulary, not a threshold decision, and it does not fail the gate — failing
on *"I cannot tell"* would make the tool unusable the first time someone writes
a phrase outside the tag space.

**It is only exact when you supply a vocabulary.** Pass `vocabulary=` a set of
the labels your classifier can actually emit and the verdict is decided by
membership. Without one it falls back to shape, which is deliberately generous
— length only, no character allowlist — because wrongly calling something
unjudgeable hides a real miss. In practice that means short prose is judged and
reported `MISSING` at 0.00 rather than shrugged off; only conspicuous prose
trips the fallback. The CLI has no vocabulary flag yet, so `tagassert check`
gets the generous version.

A fourth `PARTIAL` bucket existed in the prototype and was removed. It stacked
a second arbitrary threshold (word coverage ≥ 0.5) on top of the first
(confidence ≥ 0.35), and a reader then has to learn how much to trust two
different numbers. The signal it carried survives as a note:

```
  MISS  0.08  long black hair
            missing as written, but 'black hair', 'long hair' was detected --
            the classifier may split this compound differently
```

Every result carries its raw score, because a tag that missed at 0.34 and one
that missed at 0.02 are different facts.

## What it refuses to do

**It never passes because it found nothing to check.** The script this grew
from returned an empty list when a PNG had no embedded workflow; the comparison
then reported *"0 requested, 0 landed"* and exited 0. A green light with no
evidence behind it is worse than no gate, because nothing downstream re-reads
it. Every such case is now an error, exit 3.

**It does not guess which node holds your prompt.** The prototype hardcoded
node `1130`, which worked because one person's one workflow always put the
handwritten tags there. ComfyUI renumbers nodes on save and someone else's
graph has different ids entirely. Two honest options: name the node with
`--node`, or let it walk back from the sampler's `positive` input — which
refuses loudly when the chain passes through a node whose output is not in the
file at all (a LoRA manager injecting trigger words at run time, a prompt
upsampler generating text with a language model).

**It does not grade what you did not ask for.** Detected-but-unrequested tags
are printed, never scored. Failing on them would mean inferring a prohibition
from silence.

**Its exclusions are structural and visible.** Rating words (`general`,
`sensitive`, `explicit`, …) are dropped because WD14-family taggers put ratings
on a separate output head from general tags — a rating word requested in a
prompt would read as `MISSING` on every image regardless of content. The report
prints what it excluded, and the list is a parameter you can replace. An
opaque, ever-growing exclusion list is how a gate stops meaning anything.

## Prior art

Discriminative per-attribute binary verification of text-to-image output is
published, peer-reviewed work: **[GenEval][geneval]** (NeurIPS 2023) uses an
object detector and colour classifier rather than a VLM, for the same
interpretability reason argued here.

And the field moved away from it. **GenEval 2** (Dec 2025) replaced that
detector pipeline with a VQA method, reporting better agreement with human
annotators. **[TIFA][tifa]** reports captioning-based round-trip approaches
correlating with humans at ρ=0.34 against its own 0.60. If you want the
best-aligned automatic judgement of whether an image matches a prompt, the
literature says use a VLM, and this tool is not it.

The narrow thing that is different here: those benchmarks target open-vocabulary
natural-language prompts, where a fixed tag vocabulary is a poor fit and VQA
legitimately wins. In a booru-tag pipeline **the prompt is already written in
the classifier's own label space**, so the comparison is exact — no question
generation, no scene-graph parsing, no paraphrase in the loop. That is a domain
claim, not a method claim.

In the ComfyUI ecosystem specifically, WD14 tagger nodes are interrogators —
captioning, dataset preparation, prompt generation. The only verification-role
use of a tagger I found is an NSFW safety gate. Comfy-Org's own
[test framework][cotf] has assertion nodes, but its image assertion is dHash
perceptual hashing: it answers *are these the same pixels*, never *does this
contain what was asked for*.

## Setup

"No dependencies" describes the Python package. It does not describe the setup.
The shipped backend drives a tagger through a running ComfyUI over HTTP, which
costs no extra Python dependency — it is JSON over `urllib` — but it does need:

- ComfyUI running (`--url`, default `http://127.0.0.1:8188`)
- the `WD14Tagger|pysssss` custom node installed
- the tagger model on disk (`--model`, default `wd-eva02-large-tagger-v3`)

The default model was chosen by one careful hand comparison, not a sweep: a
smaller `convnext` tagger missed `stirrup legwear` at every threshold tried and
labelled thigh-high socks as bare feet, while `eva02-large` agreed with a human
at 0.35. Treat both the model and the 0.35 as a starting point for your
pipeline, not a validated default.

If you get your tags some other way, skip all of it — `compare()` takes two
plain arguments and has no idea where they came from.

## Install

Not on PyPI. From a clone:

```console
pip install .          # or -e ".[test]" to run the suite
pytest -q
```

## License

MIT

[geneval]: https://arxiv.org/abs/2310.11513
[tifa]: https://arxiv.org/abs/2303.11897
[cotf]: https://github.com/Comfy-Org/ComfyUI-test-framework
