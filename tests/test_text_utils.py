import json

import pytest

from newscaster.text_utils import extract_json


def test_extract_json_object_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_array_plain():
    assert extract_json('[{"a": 1}]', want=list) == [{"a": 1}]


def test_extract_json_strips_code_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_pulls_object_out_of_prose():
    # Robust to a model that wraps the JSON in chatter.
    assert extract_json('Sure! Here it is: {"a": 1} -- done') == {"a": 1}


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_extract_json_propagates_decode_error_on_malformed():
    with pytest.raises(json.JSONDecodeError):
        extract_json('{"a": }')
