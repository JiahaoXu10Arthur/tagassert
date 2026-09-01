import json
import struct
import zlib

import pytest

from tagassert import (LANDED, MISSING, NOT_JUDGEABLE, compare, judgeable,
                       normalise)
from tagassert.workflow import (WorkflowError, read_workflow, requested_tags)


# ------------------------------------------------------------- normalisation

def test_underscores_and_spaces_are_the_same_tag():
    """Danbooru writes long_hair, prompts are typed long hair, taggers emit
    either. Left alone the same tag fails to match itself."""
    assert normalise("long_hair") == normalise("long hair") == "long hair"


def test_case_and_padding_do_not_make_two_tags():
    assert normalise("  Blue   Eyes ") == "blue eyes"


def test_attention_weights_are_stripped():
    assert normalise("(stirrup legwear:1.3)") == "stirrup legwear"


def test_underscore_mismatch_is_not_reported_as_missing():
    v = compare(["long_hair"], {"long hair": 0.9})
    assert v.results[0].verdict == LANDED


# ------------------------------------------------------------------ verdicts

def test_a_tag_above_threshold_landed():
    v = compare(["smile"], {"smile": 0.8})
    assert v.results[0].verdict == LANDED
    assert v.results[0].score == 0.8
    assert v.passed and bool(v)


def test_a_tag_below_threshold_is_missing_and_keeps_its_score():
    """0.34 and 0.02 are different facts; a binary throws that away."""
    v = compare(["smile"], {"smile": 0.34}, threshold=0.35)
    assert v.results[0].verdict == MISSING
    assert v.results[0].score == 0.34
    assert not v.passed


def test_a_tag_the_classifier_never_mentioned_is_missing_at_zero():
    v = compare(["smile"], {"frown": 0.9})
    assert v.results[0].verdict == MISSING
    assert v.results[0].score == 0.0


def test_free_text_is_not_judgeable_rather_than_missing():
    """The classifier cannot emit this, so its absence is evidence of
    nothing. Failing on it would make the gate unusable."""
    phrase = "a sweeping cinematic vista of rolling hills at golden hour ok"
    v = compare([phrase], {"smile": 0.9})
    assert v.results[0].verdict == NOT_JUDGEABLE
    assert v.results[0].score is None
    assert v.passed


def test_not_judgeable_does_not_fail_the_gate():
    v = compare(["smile", "x" * 80], {"smile": 0.9})
    assert v.passed
    assert len(v.not_judgeable) == 1


def test_a_vocabulary_makes_judgeability_exact():
    v = compare(["stirrup legwear"], {"smile": 0.9}, vocabulary={"smile"})
    assert v.results[0].verdict == NOT_JUDGEABLE
    v2 = compare(["stirrup legwear"], {"smile": 0.9},
                 vocabulary={"smile", "stirrup legwear"})
    assert v2.results[0].verdict == MISSING


# ------------------------------------------------------------------- notes

def test_a_split_compound_is_a_note_not_a_fourth_verdict():
    """Taggers split compounds differently than prompts do. Worth saying;
    not worth a bucket whose threshold a reader must separately learn."""
    v = compare(["long black hair"], {"black hair": 0.9, "long hair": 0.8})
    r = v.results[0]
    assert r.verdict == MISSING
    assert "may split this compound" in r.note
    assert not v.passed


def test_a_single_word_miss_gets_no_compound_note():
    v = compare(["smile"], {"frown": 0.9})
    assert v.results[0].note == ""


# -------------------------------------------------------------- exclusions

def test_rating_words_are_excluded_for_a_structural_reason():
    """WD14 puts ratings on a separate head from general tags, so a rating
    word would read as MISSING on every image no matter its content."""
    v = compare(["general", "smile"], {"smile": 0.9})
    assert [r.tag for r in v.results] == ["smile"]
    assert v.excluded == ["general"]
    assert v.passed


def test_exclusions_are_reported_not_silently_dropped():
    v = compare(["explicit"], {"smile": 0.9})
    assert v.excluded == ["explicit"]


def test_the_exclusion_list_is_replaceable():
    v = compare(["general"], {"general": 0.9}, exclude=set())
    assert v.results[0].verdict == LANDED


# ------------------------------------------------- unrequested detected tags

def test_extra_tags_are_reported_but_never_scored():
    """Grading what nobody asked for means inferring a prohibition from
    silence, which is a guess."""
    v = compare(["smile"], {"smile": 0.9, "outdoors": 0.8})
    assert v.extra_detected == ["outdoors"]
    assert v.passed


def test_below_threshold_extras_are_not_reported():
    v = compare(["smile"], {"smile": 0.9, "outdoors": 0.1})
    assert v.extra_detected == []


def test_duplicate_requests_are_judged_once():
    v = compare(["smile", "Smile", "smile"], {"smile": 0.9})
    assert len(v.results) == 1


# ------------------------------------------------------------------ workflow

def _png(chunks):
    out = bytearray(b"\x89PNG\r\n\x1a\n")
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    for ctype, payload in [(b"IHDR", ihdr)] + list(chunks) + [(b"IEND", b"")]:
        out += struct.pack(">I", len(payload)) + ctype + payload
        out += struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF)
    return bytes(out)


def _text(key, value):
    return (b"tEXt", key.encode() + b"\x00" + value.encode())


WF = {
    "6": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "1girl, solo, smile", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality"}},
    "4": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "m.safetensors"}},
    "3": {"class_type": "KSampler",
          "inputs": {"seed": 1, "positive": ["6", 0], "negative": ["7", 0]}},
}


def test_requested_tags_walk_back_from_the_sampler(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png([_text("prompt", json.dumps(WF))]))
    assert requested_tags(p) == ["1girl", "solo", "smile"]


def test_the_negative_prompt_is_not_collected(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png([_text("prompt", json.dumps(WF))]))
    assert "worst quality" not in requested_tags(p)


def test_a_named_node_is_used_when_given(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png([_text("prompt", json.dumps(WF))]))
    assert requested_tags(p, node="6") == ["1girl", "solo", "smile"]


def test_a_missing_workflow_raises_instead_of_passing_vacuously(tmp_path):
    """The bug this guards: an earlier version returned [] here, the
    comparison then reported '0 requested, 0 landed' and PASSED. A green
    light with no evidence behind it is worse than no gate."""
    p = tmp_path / "plain.png"
    p.write_bytes(_png([]))
    with pytest.raises(WorkflowError):
        requested_tags(p)


def test_a_node_id_that_does_not_exist_raises_and_says_why(tmp_path):
    """ComfyUI renumbers nodes on save, so a hardcoded id from one file may
    simply not be in another."""
    p = tmp_path / "a.png"
    p.write_bytes(_png([_text("prompt", json.dumps(WF))]))
    with pytest.raises(WorkflowError) as e:
        requested_tags(p, node="1130")
    assert "renumbers" in str(e.value)


def test_a_runtime_injecting_node_is_refused_not_partially_read():
    """A LoRA manager emits trigger words it read off disk; the graph records
    the toggle, not the words. Returning the half that is in the file would
    check a fraction of the prompt and call it a pass."""
    api = {
        "1078": {"class_type": "TriggerWord Toggle (LoraManager)",
                 "inputs": {"group_mode": True}},
        "1043": {"class_type": "StringConcatenate",
                 "inputs": {"string_a": "masterpiece",
                            "string_b": ["1078", 0], "delimiter": ", "}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["1043", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    from tagassert.workflow import prompt_text
    with pytest.raises(WorkflowError) as e:
        prompt_text(api)
    assert "TriggerWord Toggle" in str(e.value)


def test_several_samplers_with_different_prompts_are_refused():
    api = dict(WF)
    api["8"] = {"class_type": "CLIPTextEncode",
                "inputs": {"text": "something else entirely"}}
    api["9"] = {"class_type": "KSampler", "inputs": {"positive": ["8", 0]}}
    from tagassert.workflow import prompt_text
    with pytest.raises(WorkflowError):
        prompt_text(api)


def test_not_a_png(tmp_path):
    p = tmp_path / "nope.png"
    p.write_bytes(b"hello")
    with pytest.raises(WorkflowError):
        read_workflow(p)


# ----------------------------------------------------------------- end to end

def test_end_to_end_from_a_png(tmp_path):
    p = tmp_path / "a.png"
    p.write_bytes(_png([_text("prompt", json.dumps(WF))]))
    asked = requested_tags(p)
    landed = compare(asked, {"1girl": 0.99, "solo": 0.95, "smile": 0.7})
    assert landed.passed
    missed = compare(asked, {"1girl": 0.99, "solo": 0.95, "smile": 0.1})
    assert not missed.passed
    assert [r.tag for r in missed.missing] == ["smile"]


def test_judgeable_shape_fallback_is_generous():
    """Wrongly calling something unjudgeable hides a real miss, so the
    fallback errs toward judging."""
    assert judgeable("stirrup legwear")
    assert judgeable("hand on own hip")
    assert judgeable("re:zero kara hajimeru isekai seikatsu")
    assert not judgeable("")
    assert not judgeable("x" * 80)


def test_symbol_tags_are_judgeable():
    """`^_^` and friends are real danbooru expression tags. Rejecting them
    for not starting with a letter turns a real miss into a shrug."""
    assert judgeable("^_^")
    assert judgeable(">_<")
    assert compare(["^_^"], {"^ ^": 0.9}).results[0].verdict == LANDED


# ------------------------------------- holes found in adversarial verification

def test_a_single_word_split_gets_the_note():
    """`twintails` against a detected `twin tails` is the cleanest split
    there is, and gating the note on multi-word requests skipped it -- the
    exact case the note exists for, silently absent."""
    r = compare(["twintails"], {"twin tails": 0.9}).results[0]
    assert r.verdict == MISSING
    assert "twin tails" in r.note


def test_two_shared_words_do_not_fabricate_a_split_story():
    """`dark blue eyes` vs a detected `dark blue background` shares two
    words and means something else entirely. The old fallback printed a
    compound-split explanation for it."""
    r = compare(["dark blue eyes"], {"dark blue background": 0.9}).results[0]
    assert r.verdict == MISSING
    assert r.note == ""


def test_non_ascii_tags_are_judgeable():
    """Accented and CJK tags are real. An ASCII allowlist made every one of
    them a shrug."""
    assert judgeable("café") and judgeable("巫女")
    assert compare(["巫女"], {"巫女": 0.9}).results[0].verdict == LANDED


def test_prose_is_still_not_judgeable_by_length():
    assert not judgeable("a sweeping cinematic vista of rolling hills at "
                         "golden hour with mist")


def test_compare_refuses_an_empty_request():
    """An empty comparison that reports success is the failure this package
    exists to prevent."""
    with pytest.raises(ValueError):
        compare([], {"smile": 0.9})


def test_a_bare_string_is_split_not_iterated_by_character():
    v = compare("smile, 1girl", {"smile": 0.9, "1girl": 0.9})
    assert [r.tag for r in v.results] == ["smile", "1girl"]


def test_backend_parse_keeps_every_tag_in_a_list_of_strings():
    """Reading only value[0] dropped the rest, and every lost tag came back
    to the caller as a confident MISSING."""
    from tagassert.backends import ComfyTagger
    got = ComfyTagger._parse({"3": {"tags": ["1girl", "solo", "long hair"]}})
    assert set(got) == {"1girl", "solo", "long hair"}


def test_backend_parse_handles_the_documented_shapes():
    from tagassert.backends import ComfyTagger, BackendError
    assert ComfyTagger._parse({"2": {"t": {"smile": 0.8}}}) == {"smile": 0.8}
    assert ComfyTagger._parse({"2": {"t": [{"smile": 0.8}]}}) == {"smile": 0.8}
    assert ComfyTagger._parse({"2": {"t": "a, b"}}) == {"a": 1.0, "b": 1.0}
    with pytest.raises(BackendError):
        ComfyTagger._parse({"2": {"t": 17}})


def test_a_literal_two_string_list_is_not_mistaken_for_a_link():
    """A widget holding ["masterpiece", "best quality"] has the same shape as
    a [node_id, slot] link. Treating it as a dangling link dropped it from
    the prompt with no error -- a silent partial loss."""
    from tagassert.workflow import prompt_text
    api = {
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "1girl, solo",
                         "style_tags": ["masterpiece", "best quality"]}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0]}},
    }
    assert "masterpiece" in prompt_text(api)


def _realistic_png(text_chunks, idat_count=5, idat_size=2048):
    """A PNG shaped like the ones ComfyUI actually writes -- IHDR, one or two
    tEXt chunks, several IDAT, IEND. The synthetic ones above have no IDAT,
    which is the easy case for a chunk walker. No pixels; IDAT is filler."""
    out = bytearray(b"\x89PNG\r\n\x1a\n")

    def put(ctype, payload):
        out.extend(struct.pack(">I", len(payload)))
        out.extend(ctype + payload)
        out.extend(struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))

    put(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    for key, value in text_chunks:
        put(b"tEXt", key.encode() + b"\x00" + value.encode())
    for _ in range(idat_count):
        put(b"IDAT", bytes(idat_size))
    put(b"IEND", b"")
    return bytes(out)


def test_requested_tags_found_past_many_idat(tmp_path):
    p = tmp_path / "real.png"
    p.write_bytes(_realistic_png([("prompt", json.dumps(WF))], idat_count=30))
    assert requested_tags(p) == ["1girl", "solo", "smile"]


def test_the_editor_copy_does_not_win_over_the_prompt(tmp_path):
    p = tmp_path / "both.png"
    p.write_bytes(_realistic_png([
        ("workflow", json.dumps({"nodes": [{"id": 1, "type": "Decoy"}]})),
        ("prompt", json.dumps(WF)),
    ]))
    assert requested_tags(p) == ["1girl", "solo", "smile"]


# --------------------------------------------------------- vocabulary loading

def test_a_plain_list_is_one_tag_per_line(tmp_path):
    p = tmp_path / "tags.txt"
    p.write_text("1girl\nsolo\nstirrup legwear\n")
    from tagassert.__main__ import read_vocabulary
    assert read_vocabulary(p) == {"1girl", "solo", "stirrup legwear"}


def test_a_wd14_selected_tags_csv_is_read_from_its_name_column(tmp_path):
    p = tmp_path / "selected_tags.csv"
    p.write_text("tag_id,name,category,count\n"
                 "470575,1girl,0,5352272\n"
                 "212816,solo,0,4498549\n"
                 "13200,general,9,1000\n")
    from tagassert.__main__ import read_vocabulary
    assert read_vocabulary(p) == {"1girl", "solo", "general"}


def test_vocabulary_entries_are_normalised_like_requests_are(tmp_path):
    # WD14 ships underscores; a prompt is typed with spaces. Comparing them
    # raw makes every underscored tag fall outside its own vocabulary, and an
    # unjudgeable verdict does not fail the gate -- so a real miss would be
    # silently downgraded to a shrug.
    p = tmp_path / "selected_tags.csv"
    p.write_text("tag_id,name,category,count\n470575,long_hair,0,1\n")
    from tagassert.__main__ import read_vocabulary
    vocab = read_vocabulary(p)
    assert vocab == {"long hair"}
    v = compare(["long hair"], {}, vocabulary=vocab)
    assert v.results[0].verdict == MISSING


def test_an_empty_vocabulary_file_is_an_error_not_an_empty_set(tmp_path):
    # An empty set makes every tag unjudgeable, and unjudgeable does not fail
    # the gate -- so the run would exit 0 having checked nothing. That is the
    # vacuous pass this package exists to refuse.
    p = tmp_path / "empty.txt"
    p.write_text("\n\n  \n")
    from tagassert.__main__ import read_vocabulary
    with pytest.raises(ValueError, match="no tags"):
        read_vocabulary(p)


def test_blank_lines_and_surrounding_space_are_ignored(tmp_path):
    p = tmp_path / "tags.txt"
    p.write_text("\n  1girl  \n\nsolo\n\n")
    from tagassert.__main__ import read_vocabulary
    assert read_vocabulary(p) == {"1girl", "solo"}


def test_an_unrecognised_csv_is_refused_rather_than_matching_nothing(tmp_path):
    # No header naming a column, so every line would become one "tag" full of
    # commas -- a vocabulary that matches nothing, reports every request
    # unjudgeable, and exits 0 having judged none of them.
    p = tmp_path / "headerless.csv"
    p.write_text("470575,1girl,0,5352272\n212816,solo,0,4498549\n")
    from tagassert.__main__ import read_vocabulary
    with pytest.raises(ValueError, match="comma"):
        read_vocabulary(p)


def test_a_bad_vocabulary_stops_the_run_instead_of_checking_nothing(tmp_path):
    from tagassert.__main__ import main
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert main(["check", str(tmp_path / "x.png"),
                 "--vocabulary", str(p)]) == 3


def test_a_missing_vocabulary_file_exits_3_rather_than_traceback(tmp_path):
    # 3 is the documented "could not run" code. An unhandled OSError here
    # would leave the caller reading a traceback for a plain bad path.
    from tagassert.__main__ import main
    assert main(["check", str(tmp_path / "x.png"),
                 "--vocabulary", str(tmp_path / "nope.txt")]) == 3


def test_backend_parse_keeps_every_tag_in_a_list_of_dicts():
    """The sibling of the string-list case above, three lines away in the same
    function and still reading value[0]. A dropped detection does not read as
    "I could not tell" -- it reads as a confident MISSING at 0.00."""
    from tagassert.backends import ComfyTagger
    got = ComfyTagger._parse(
        {"3": {"t": [{"1girl": 0.9}, {"solo": 0.8, "smile": 0.7}]}})
    assert got == {"1girl": 0.9, "solo": 0.8, "smile": 0.7}


def test_backend_parse_reads_every_output_field_on_a_node():
    """WD14 puts ratings on an output head separate from the general tags --
    the structural fact ADR-2 decision 4 rests on. Returning at the first
    readable field means that if the rating head is walked first, every
    general tag on every image comes back MISSING and the gate fails
    continuously while looking like it works."""
    from tagassert.backends import ComfyTagger
    got = ComfyTagger._parse(
        {"3": {"rating": {"general": 0.9}, "tags": {"1girl": 0.9, "solo": 0.8}}})
    assert got == {"general": 0.9, "1girl": 0.9, "solo": 0.8}


def test_backend_parse_reads_every_output_node():
    from tagassert.backends import ComfyTagger
    got = ComfyTagger._parse({"2": {"t": {"general": 0.9}},
                              "3": {"t": {"1girl": 0.9}}})
    assert got == {"general": 0.9, "1girl": 0.9}


def _broken_ztxt(key):
    return (b"zTXt", key.encode() + b"\x00" + b"\x00" + b"not-actually-zlib")


def _itxt(key, value):
    """An uncompressed iTXt chunk: key, flag, method, language, translated."""
    return (b"iTXt", key.encode() + b"\x00" + b"\x00\x00" + b"\x00" + b"\x00"
            + value.encode())


def test_an_undecodable_prompt_chunk_is_not_reported_as_a_missing_one(tmp_path):
    """A prompt chunk that failed to decompress left no trace, so the file
    read as "has no embedded ComfyUI workflow" -- with an invented cause and a
    chunk list that omitted the chunk that failed."""
    p = tmp_path / "x.png"
    p.write_bytes(_png([_broken_ztxt("prompt")]))
    with pytest.raises(WorkflowError) as e:
        read_workflow(p)
    msg = str(e.value)
    assert "prompt" in msg
    assert "has no embedded ComfyUI workflow" not in msg
    assert "re-saved by an editor" not in msg


def test_one_broken_chunk_does_not_lose_a_readable_prompt(tmp_path):
    p = tmp_path / "y.png"
    p.write_bytes(_png([_broken_ztxt("junk"), _text("prompt", json.dumps(WF))]))
    assert read_workflow(p)["6"]["class_type"] == "CLIPTextEncode"


def test_a_prompt_in_an_itxt_chunk_is_read(tmp_path):
    """attribution-gate's copy of this walker handles iTXt; this one did not,
    so a prompt written there was never looked at and reported as absent --
    the same collapse as the undecodable case, one step earlier."""
    p = tmp_path / "z.png"
    p.write_bytes(_png([_itxt("prompt", json.dumps(WF))]))
    assert read_workflow(p)["6"]["class_type"] == "CLIPTextEncode"
