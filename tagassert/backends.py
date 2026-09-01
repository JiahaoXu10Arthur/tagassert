"""Where the detected tags come from.

A backend takes an image and returns ``{tag: confidence}`` -- raw scores, not
a thresholded list. Thresholding is a decision, and it belongs in
:func:`tagassert.compare` where it is visible and adjustable, not buried in
whichever process produced the numbers.

The one shipped backend drives a WD14-family tagger through a running
ComfyUI's HTTP API. That costs no extra dependency -- it is JSON over
``urllib`` -- but it is honest to say what it does cost: a running ComfyUI,
the tagger custom node installed, and the model weights on disk. "Zero
dependencies" describes the Python package. It does not describe the setup.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, Optional

__all__ = ["ComfyTagger", "BackendError"]


class BackendError(Exception):
    """The tagger could not be reached, or did not answer usefully."""


class ComfyTagger(object):
    """Run a WD14-family tagger on an image through a running ComfyUI.

    Requires the tagger custom node (``WD14Tagger|pysssss``) and its model.
    The default model is the one the prototype settled on after a hand
    comparison: a smaller ``convnext`` tagger missed ``stirrup legwear`` at
    every threshold tried and called thigh-high socks bare feet, while
    ``eva02-large`` agreed with a human at 0.35. That is one careful
    comparison on one pipeline, not a sweep -- treat it as a starting point.
    """

    def __init__(self, url: str = "http://127.0.0.1:8188",
                 model: str = "wd-eva02-large-tagger-v3",
                 input_dir: Optional[Path] = None, timeout: int = 120):
        self.url = url.rstrip("/")
        self.model = model
        self.input_dir = Path(input_dir) if input_dir else None
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=self.timeout))

    def _get(self, path: str) -> dict:
        return json.load(urllib.request.urlopen(self.url + path,
                                                timeout=self.timeout))

    def check(self) -> None:
        """Fail early with something a human can act on.

        A bare ``WinError 10061`` a hundred lines into a batch is not
        actionable; this is.
        """
        try:
            urllib.request.urlopen(self.url + "/system_stats", timeout=5).read(1)
        except Exception as e:
            raise BackendError(
                "cannot reach ComfyUI at %s (%s). Start it, or point --url at "
                "the right host and port." % (self.url, type(e).__name__))

    def tag(self, image: Path) -> Dict[str, float]:
        """Return ``{tag: confidence}`` for one image."""
        self.check()
        image = Path(image)
        if not image.exists():
            raise BackendError("%s does not exist" % image)

        name = image.name
        if self.input_dir:
            staged = Path(self.input_dir) / ("tagassert_%s_%s"
                                             % (uuid.uuid4().hex[:8], name))
            shutil.copy2(image, staged)
            name = staged.name

        graph = {
            "1": {"class_type": "LoadImage", "inputs": {"image": name}},
            "2": {"class_type": "WD14Tagger|pysssss",
                  "inputs": {"image": ["1", 0], "model": self.model,
                             "threshold": 0.01, "character_threshold": 0.01,
                             "replace_underscore": True, "trailing_comma": False,
                             "exclude_tags": ""}},
            "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
        }
        cid = uuid.uuid4().hex
        try:
            self._post("/prompt", {"prompt": graph, "client_id": cid})
        except Exception as e:
            raise BackendError(
                "the tagger node rejected the request (%s). Is "
                "WD14Tagger|pysssss installed and is %r a model it has?"
                % (e, self.model))

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = self._get("/history")
            for entry in hist.values():
                outs = entry.get("outputs") or {}
                if "3" in outs or "2" in outs:
                    return self._parse(outs)
            time.sleep(0.4)
        raise BackendError("the tagger did not answer within %ds"
                           % self.timeout)

    @staticmethod
    def _parse(outputs: dict) -> Dict[str, float]:
        """Pull ``{tag: score}`` out of whatever shape the node returned.

        The node has emitted a bare string, a list of strings, and a mapping
        across versions. A comma-joined string has already thrown the scores
        away, so those tags come back at 1.0 with no way to recover the
        original confidence -- worth knowing when a report shows every score
        as exactly 1.0.
        """
        def from_text(text: str) -> Dict[str, float]:
            return {t.strip(): 1.0 for t in text.split(",") if t.strip()}

        # Every element of every field of every node, never the first one
        # found. Returning early dropped tags three ways -- the tail of a
        # dict list, a second output field, a second output node -- and a
        # dropped detection does not reach the caller as "I could not tell".
        # It reaches it as a confident MISSING at 0.00, which is the same
        # thing the model genuinely not seeing it looks like.
        #
        # The second field is not hypothetical: WD14 puts ratings on a head
        # separate from the general tags, which is the structural fact the
        # rating exclusion in ``compare`` rests on. Walk that head first and
        # every general tag on every image reads MISSING, so the gate fails
        # continuously while looking like it works.
        out: Dict[str, float] = {}
        for node_out in outputs.values():
            for value in (node_out or {}).values():
                if isinstance(value, dict):
                    out.update((str(k), float(v)) for k, v in value.items())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            out.update((str(k), float(v))
                                       for k, v in item.items())
                        elif isinstance(item, str):
                            out.update(from_text(item))
                elif isinstance(value, str):
                    out.update(from_text(value))
        if not out:
            raise BackendError("the tagger returned nothing this can read")
        return out
