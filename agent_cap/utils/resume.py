"""Helpers for reconstructing metrics when resuming legacy experiment output."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


_NONNEGATIVE_INTEGER = re.compile(r"0|[1-9][0-9]*")


def require_nonnegative_int(value: Any, *, context: str) -> int:
    """Parse a persisted integer without accepting lossy numeric coercions."""
    if isinstance(value, bool):
        raise RuntimeError(f"{context} has invalid integer value {value!r}.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and _NONNEGATIVE_INTEGER.fullmatch(value):
        parsed = int(value)
    else:
        raise RuntimeError(f"{context} has invalid integer value {value!r}.")
    if parsed < 0:
        raise RuntimeError(f"{context} has negative integer value {parsed}.")
    return parsed


def require_nonnegative_number(value: Any, *, context: str) -> float:
    """Validate a persisted finite, nonnegative JSON number."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RuntimeError(f"{context} has invalid numeric value {value!r}.")
    return float(value)


def recover_num_requests(
    output_row: Mapping[str, Any],
    detailed_rows: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> int:
    """Return a trustworthy request count for one resumed task.

    Current output rows contain ``num_requests`` directly. Older rows may not,
    but their detailed request records contain zero-based ``request_index``
    values. Refuse to guess when neither source can establish the count.
    """

    raw_num_requests = output_row.get("num_requests")
    if raw_num_requests is not None:
        return require_nonnegative_int(
            raw_num_requests,
            context=f"{context} num_requests",
        )

    if not detailed_rows:
        raise RuntimeError(
            f"{context} is missing num_requests and has no detailed request rows "
            "from which to recover it. Refusing to default the count to 1 because "
            "that would corrupt avg_num_requests."
        )

    request_indexes = []
    for detail_position, detail_row in enumerate(detailed_rows):
        raw_request_index = detail_row.get("request_index")
        if raw_request_index is None:
            raise RuntimeError(
                f"{context} is missing num_requests, and detailed request row "
                f"{detail_position} has no request_index. Refusing to guess the "
                "request count."
            )
        request_index = require_nonnegative_int(
            raw_request_index,
            context=f"{context} detailed request row {detail_position} request_index",
        )
        request_indexes.append(request_index)

    return max(request_indexes) + 1
