"""Regression tests for legacy resume request-count reconstruction."""

import pytest

from agent_cap.utils.resume import recover_num_requests, require_nonnegative_number


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


@pytest.mark.parametrize("value", [1.9, 0.5, True, -1, "01", "1.0"])
def test_resume_integer_fields_reject_lossy_or_invalid_values(value):
    with pytest.raises(RuntimeError, match="invalid|negative"):
        recover_num_requests(
            {"num_requests": value},
            [],
            context="test row",
        )


@pytest.mark.parametrize("value", [True, -0.1, float("nan"), float("inf"), "1"])
def test_resume_numeric_fields_require_finite_nonnegative_json_numbers(value):
    with pytest.raises(RuntimeError, match="invalid"):
        require_nonnegative_number(value, context="test value")
