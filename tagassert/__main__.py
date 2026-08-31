"""Command line entry point.

Exit codes::

    0   every requested tag landed
    1   at least one requested tag is missing
    3   the check could not run (no workflow, unreachable tagger, bad usage)

3 is not a soft failure. Nothing was judged, so nothing may be concluded --
which is the opposite of the exit 0 that an earlier version produced when it
found no prompt to check.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import DEFAULT_THRESHOLD, LANDED, MISSING, NOT_JUDGEABLE, compare
from .backends import BackendError, ComfyTagger
from .workflow import WorkflowError, requested_tags

USAGE = """\
tagassert -- did the tags you asked for actually land?

  tagassert check IMAGE.png [options]
      Read the requested tags from the image's own embedded workflow, run a
      tagger over the image, and compare.

  tagassert compare IMAGE.png --tags "a, b, c"
      Same, but you say what was requested.

Options:
  --node ID          which workflow node holds the prompt (default: walk back
                     from the sampler's positive input)
  --tags "a, b"      requested tags, instead of reading the workflow
  --threshold F      confidence at or above which a tag counts as landed
                     (default %.2f)
  --url URL          ComfyUI base url (default http://127.0.0.1:8188)
  --model NAME       tagger model (default wd-eva02-large-tagger-v3)
  --input-dir PATH   ComfyUI's input directory, if the image must be staged
  --quiet            print only failures

Exit: 0 all landed, 1 something missing, 3 could not run.
""" % DEFAULT_THRESHOLD

_MARK = {LANDED: "ok  ", MISSING: "MISS", NOT_JUDGEABLE: "-   "}


def _fail(msg: str, *extra: str) -> int:
    print("error: %s" % msg, file=sys.stderr)
    for line in extra:
        print("       %s" % line, file=sys.stderr)
    return 3


def _opts(args):
    out = {"node": None, "tags": None, "threshold": DEFAULT_THRESHOLD,
           "url": "http://127.0.0.1:8188",
           "model": "wd-eva02-large-tagger-v3", "input_dir": None,
           "quiet": False}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--quiet":
            out["quiet"] = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise ValueError("%s needs a value" % a)
        v = args[i + 1]
        if a == "--node":
            out["node"] = v
        elif a == "--tags":
            out["tags"] = [t.strip() for t in v.split(",") if t.strip()]
        elif a == "--threshold":
            out["threshold"] = float(v)
        elif a == "--url":
            out["url"] = v
        elif a == "--model":
            out["model"] = v
        elif a == "--input-dir":
            out["input_dir"] = v
        else:
            raise ValueError("unrecognised option %r" % a)
        i += 2
    return out


def _report(v, quiet: bool) -> None:
    for r in v.results:
        if quiet and r.verdict != MISSING:
            continue
        score = "    -" if r.score is None else "%5.2f" % r.score
        print("  %s %s  %s" % (_MARK[r.verdict], score, r.tag))
        if r.note:
            print("            %s" % r.note)
    if v.excluded and not quiet:
        print("  excluded (structural): %s" % ", ".join(v.excluded))
    if v.extra_detected and not quiet:
        shown = ", ".join(v.extra_detected[:12])
        more = " ... (%d more)" % (len(v.extra_detected) - 12) \
            if len(v.extra_detected) > 12 else ""
        print("  also detected, not requested: %s%s" % (shown, more))
    print()
    print("  %d requested: %d landed, %d missing, %d not judgeable "
          "(threshold %.2f)"
          % (len(v.results), len(v.landed), len(v.missing),
             len(v.not_judgeable), v.threshold))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in ("check", "compare"):
        return _fail("unknown command %r" % cmd, "try --help")
    if not rest:
        return _fail("%s needs an image" % cmd)

    image = Path(rest[0])
    try:
        o = _opts(rest[1:])
    except ValueError as e:
        return _fail(str(e))

    try:
        asked = o["tags"] if o["tags"] is not None else requested_tags(
            image, o["node"])
    except WorkflowError as e:
        return _fail(str(e),
                     "pass the requested tags explicitly with --tags if you "
                     "know them.")
    except OSError as e:
        return _fail(str(e))

    if not asked:
        return _fail("no requested tags found -- nothing to check",
                     "An empty check that reports success is worse than no "
                     "check at all, so this is an error, not a pass.")

    try:
        detected = ComfyTagger(o["url"], o["model"], o["input_dir"]).tag(image)
    except BackendError as e:
        return _fail(str(e))

    v = compare(asked, detected, threshold=o["threshold"])
    print("%s" % image.name)
    _report(v, o["quiet"])
    return 0 if v.passed else 1


if __name__ == "__main__":
    sys.exit(main())
