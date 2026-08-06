"""Helpers for reconstructing metrics when resuming legacy experiment output."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


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
        try:
            num_requests = int(raw_num_requests)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{context} has invalid num_requests={raw_num_requests!r}."
            ) from exc

        if num_requests < 0:
            raise RuntimeError(
                f"{context} has negative num_requests={num_requests}."
            )
        return num_requests

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
        try:
            request_index = int(raw_request_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{context} has invalid request_index={raw_request_index!r} in "
                f"detailed request row {detail_position}."
            ) from exc
        if request_index < 0:
            raise RuntimeError(
                f"{context} has negative request_index={request_index} in "
                f"detailed request row {detail_position}."
            )
        request_indexes.append(request_index)

    return max(request_indexes) + 1
