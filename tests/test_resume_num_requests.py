"""Regression tests for legacy resume request-count reconstruction."""

import pytest

from agent_cap.utils.resume import recover_num_requests


def test_explicit_num_requests_is_preserved():
    assert recover_num_requests(
        {"num_requests": 4},
        [{"request_index": 0}],
        context="test row",
    ) == 4


def test_missing_num_requests_is_recovered_from_request_indexes():
    assert recover_num_requests(
        {},
        [
            {"request_index": 0},
            {"request_index": 1},
            {"request_index": 2},
        ],
        context="test row",
    ) == 3


def test_latest_request_index_handles_a_gap_in_detailed_rows():
    assert recover_num_requests(
        {},
        [
            {"request_index": 0},
            {"request_index": 2},
        ],
        context="test row",
    ) == 3


def test_missing_num_requests_without_details_does_not_default_to_one():
    with pytest.raises(RuntimeError, match="Refusing to default the count to 1"):
        recover_num_requests({}, [], context="test row")


def test_missing_request_index_does_not_get_silently_counted():
    with pytest.raises(RuntimeError, match="has no request_index"):
        recover_num_requests({}, [{}], context="test row")
