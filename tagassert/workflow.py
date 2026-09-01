"""Recover the requested tags from a ComfyUI PNG.

ComfyUI writes the API graph it actually ran into every PNG, so the prompt is
in the file. Getting it back out is the part with sharp edges.

It raises. It never returns an empty list.
--------------------------------------------
The script this grew from returned ``[]`` when a PNG had no workflow, and the
comparison downstream then reported "0 requested, 0 landed" and **passed**. A
gate that passes because it found nothing to check is worse than no gate: it
is a green light with no evidence behind it, and nothing downstream re-reads
it. Every failure here is an exception.

Finding "the prompt" is not automatic, and pretending otherwise is the trap
-----------------------------------------------------------------------------
The original hardcoded node ``1130``, which worked because one person's one
workflow always put the handwritten tags there. That is a constant, not a
design; ComfyUI renumbers nodes on save, and someone else's graph has
different ids entirely.

Two honest ways to say which text is the prompt:

- ``node=`` names it explicitly. Exact, and yours to maintain.
- with no ``node``, the sampler's ``positive`` input is walked backwards
  through string nodes. Correct when the chain is plain string plumbing, and a
  loud refusal when it is not -- a LoRA manager that injects trigger words at
  run time, or a prompt upsampler that writes text with a language model, put
  the real prompt outside the file, and no amount of walking recovers it.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["read_workflow", "requested_tags", "WorkflowError"]

_PNG_SIG = b"\x89PNG\r\n\x1a\n"

#: Node classes whose output follows from their string inputs, so walking back
#: through them reconstructs what the encoder actually received. An allowlist,
#: because a blocklist only ever grows: every new custom node is a new way to
#: collect a filename or an enum value and call it a prompt.
_TRAVERSABLE = (
    "CLIPTextEncode", "TextEncode", "StringConcatenate", "JoinStringMulti",
    "JoinString", "PrimitiveString", "StringLiteral", "CR Text",
    "Text Multiline", "ShowText", "StringConstantMultiline",
)

_SAMPLERS = ("KSampler", "SamplerCustom", "KSamplerAdvanced")

_SKIP_KEYS = frozenset({
    "delimiter", "separator", "type", "device", "clip", "vae", "model",
    "ckpt_name", "lora_name", "clip_name", "unet_name", "sampler_name",
    "scheduler", "filename_prefix", "image", "latent_image",
})

_MAX_DEPTH = 24


class WorkflowError(Exception):
    """The PNG carried no usable prompt, or one that cannot be recovered."""


#: A chunk that is present but did not decode. It stands in the result where
#: the text would have been, so "I could not decompress your prompt" cannot
#: arrive as "this image has no prompt".
class _Undecodable(object):
    __slots__ = ("why",)

    def __init__(self, why: str):
        self.why = why

    def __repr__(self) -> str:
        return "<undecodable: %s>" % self.why


def _chunks(png: Path) -> Dict[str, str]:
    data = Path(png).read_bytes()
    if not data.startswith(_PNG_SIG):
        raise WorkflowError("%s is not a PNG" % Path(png).name)
    out: Dict[str, str] = {}
    pos = len(_PNG_SIG)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IEND":
            break
        try:
            if ctype == b"tEXt":
                k, _, v = body.partition(b"\x00")
                out[k.decode("latin-1")] = v.decode("utf-8", "replace")
            elif ctype == b"zTXt":
                k, _, rest = body.partition(b"\x00")
                out[k.decode("latin-1")] = zlib.decompress(
                    rest[1:]).decode("utf-8", "replace")
            elif ctype == b"iTXt":
                # This copy had no iTXt branch, so a prompt written there was
                # never looked at -- and an unread chunk reports exactly like
                # an absent one.
                k, _, rest = body.partition(b"\x00")
                flag = rest[0] if rest else 0
                rest = rest[2:]                        # compression flag, method
                _, _, rest = rest.partition(b"\x00")   # language tag
                _, _, rest = rest.partition(b"\x00")   # translated keyword
                out[k.decode("latin-1")] = (
                    zlib.decompress(rest) if flag else rest
                ).decode("utf-8", "replace")
        except Exception as exc:
            # A malformed chunk must not abandon the whole file, and must not
            # vanish from it either. Leaving no trace made a prompt that failed
            # to decompress read as "no embedded ComfyUI workflow", with an
            # invented cause and a chunk list missing the chunk that failed.
            try:
                k = body.partition(b"\x00")[0].decode("latin-1")
            except Exception:
                continue
            out.setdefault(k, _Undecodable(
                "%s chunk: %s" % (ctype.decode("latin-1"), exc)))
            continue
    return out


def read_workflow(png) -> dict:
    """The API graph ComfyUI embedded in ``png``.

    Only the ``prompt`` chunk, which is what actually ran. The ``workflow``
    chunk is the editor's copy and can differ.
    """
    chunks = _chunks(png)
    raw = chunks.get("prompt")
    if isinstance(raw, _Undecodable):
        raise WorkflowError(
            "%s carries a prompt chunk that did not decode (%s). The workflow "
            "is there and unreadable, which is a different problem from an "
            "image that never had one." % (Path(png).name, raw.why))
    if not raw:
        have = ", ".join(sorted(chunks)) or "none"
        raise WorkflowError(
            "%s has no embedded ComfyUI workflow (text chunks: %s). An image "
            "re-saved by an editor loses it." % (Path(png).name, have))
    try:
        api = json.loads(raw)
    except ValueError as e:
        raise WorkflowError("%s: unreadable workflow (%s)"
                            % (Path(png).name, e))
    if not isinstance(api, dict) or not api:
        raise WorkflowError("%s: empty workflow" % Path(png).name)
    return api


def _resolve(api: dict, value, depth: int = 0) -> List[str]:
    if depth > _MAX_DEPTH:
        raise WorkflowError("prompt chain deeper than %d nodes, or a cycle"
                            % _MAX_DEPTH)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and len(value) == 2:
        nid = str(value[0])
        node = api.get(nid)
        if not isinstance(node, dict):
            # A link always points at a node. A two-element list that does
            # not is a literal widget value -- ``["masterpiece", "best
            # quality"]`` -- and treating it as a dangling link dropped it
            # from the prompt with no error. The overall result stayed
            # non-empty, so the "never silently empty" guard never fired: a
            # silent *partial* loss, which nothing downstream can notice.
            if all(isinstance(x, str) for x in value):
                return list(value)
            return []
        cls = str(node.get("class_type") or "")
        if not any(s in cls for s in _TRAVERSABLE):
            raise WorkflowError(
                "cannot recover the prompt: node %s is a %r, which is not a "
                "string node whose output follows from its inputs. If it "
                "injects text at run time or generates it, the text is not in "
                "this file. Name the node you mean with node=, or pass the "
                "tags directly." % (nid, cls))
        out: List[str] = []
        for k, v in (node.get("inputs") or {}).items():
            if k in _SKIP_KEYS or isinstance(v, (int, float, bool)):
                continue
            out.extend(_resolve(api, v, depth + 1))
        return out
    return []


def prompt_text(api: dict, node: Optional[str] = None) -> str:
    """The positive prompt, either from a named node or by walking back."""
    if node is not None:
        nid = str(node)
        if nid not in api:
            raise WorkflowError(
                "node %s is not in this workflow (it has %d nodes). ComfyUI "
                "renumbers nodes when a graph is saved, so a node id from one "
                "file may not exist in another." % (nid, len(api)))
        found = [p for p in _resolve(api, [nid, 0]) if p.strip()]
        if not found:
            raise WorkflowError("node %s holds no text" % nid)
        return ", ".join(found)

    samplers = [n for n in api
                if any(s in str((api[n] or {}).get("class_type") or "")
                       for s in _SAMPLERS)]
    if not samplers:
        raise WorkflowError(
            "no sampler node found, so the positive conditioning cannot be "
            "located. Name the prompt node with node=.")
    texts = []
    for n in sorted(samplers):
        pos = (api[n].get("inputs") or {}).get("positive")
        if pos is None:
            continue
        t = ", ".join(p for p in _resolve(api, pos) if p.strip())
        if t.strip():
            texts.append(t)
    uniq = sorted(set(texts))
    if not uniq:
        raise WorkflowError("the sampler's positive input resolved to no text")
    if len(uniq) > 1:
        raise WorkflowError(
            "this workflow has %d samplers with different prompts, so there "
            "is no single prompt to check. Name one with node=." % len(uniq))
    return uniq[0]


def requested_tags(png, node: Optional[str] = None) -> List[str]:
    """The tags a PNG's own workflow says were asked for."""
    text = prompt_text(read_workflow(png), node)
    return [t.strip() for t in text.split(",") if t.strip()]
