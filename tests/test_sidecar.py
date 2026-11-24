from __future__ import annotations

import copy

import pytest

from nla_audit.nla.sidecar import (
    NLASidecar,
    av_prompt_ids,
    injection_position,
    verify_ar_tokenizer,
    verify_av_tokenizer,
)


def test_parse_released_av_sidecar(av_meta):
    s = NLASidecar.from_dict(av_meta)
    assert s.role == "av"
    assert s.d_model == 3584
    assert s.extraction_layer_index == 20
    assert s.injection_scale == 150.0
    assert s.mse_scale == pytest.approx(3584**0.5)
    assert s.injection_token_id == 149705
    assert "{injection_char}" in s.av_template
    assert "{explanation}" in s.ar_template


def test_parse_released_ar_sidecar(ar_meta):
    s = NLASidecar.from_dict(ar_meta)
    assert s.role == "ar"
    assert s.injection_scale is None
    assert s.critic_suffix_ids == (1318, 29, 366, 1708, 29)
    assert s.extraction_layer_index == 20


def test_av_sidecar_requires_injection_scale(av_meta):
    m = copy.deepcopy(av_meta)
    m["extraction"]["injection_scale"] = None
    with pytest.raises(ValueError, match="injection_scale"):
        NLASidecar.from_dict(m)


def test_rejects_non_nla(av_meta):
    m = copy.deepcopy(av_meta)
    m["kind"] = "something_else"
    with pytest.raises(ValueError):
        NLASidecar.from_dict(m)


@pytest.mark.network
def test_av_tokenizer_agrees_with_sidecar(av_meta, qwen_tokenizer):
    s = NLASidecar.from_dict(av_meta)
    pos = verify_av_tokenizer(s, qwen_tokenizer)
    ids = av_prompt_ids(s, qwen_tokenizer)
    assert ids[pos] == s.injection_token_id
    assert ids[pos - 1] == s.injection_left_neighbor_id
    assert ids[pos + 1] == s.injection_right_neighbor_id


@pytest.mark.network
def test_template_drift_is_caught(av_meta, qwen_tokenizer):
    m = copy.deepcopy(av_meta)
    m["prompt_templates"]["av"] = m["prompt_templates"]["av"].replace(
        "<concept>{injection_char}</concept>", "<concept> {injection_char} </concept>"
    )
    s = NLASidecar.from_dict(m)
    with pytest.raises(AssertionError):
        verify_av_tokenizer(s, qwen_tokenizer)


@pytest.mark.network
def test_ar_suffix_matches(ar_meta, qwen_tokenizer):
    verify_ar_tokenizer(NLASidecar.from_dict(ar_meta), qwen_tokenizer)


def test_injection_position_requires_neighbours(av_meta):
    s = NLASidecar.from_dict(av_meta)
    inj, left, right = (
        s.injection_token_id,
        s.injection_left_neighbor_id,
        s.injection_right_neighbor_id,
    )
    assert injection_position(s, [1, left, inj, right, 2]) == 2
    with pytest.raises(AssertionError):
        injection_position(s, [1, 7, inj, right, 2])  # wrong left neighbour
    with pytest.raises(AssertionError):
        injection_position(s, [left, inj, right, left, inj, right])  # two sites
