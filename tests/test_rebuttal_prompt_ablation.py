from types import SimpleNamespace

from scripts.aggregate_rebuttal_prompt_ablation import exact_mcnemar, paired_bootstrap_delta
from scripts.rebuttal_prompt_ablation import (
    LANGCHAIN_EXEC,
    LANGCHAIN_PLAN,
    load_task_indices,
    prompt_pair,
    prompt_provenance,
    is_mcp_read_only,
    select_stratified,
    select_stratified_window,
    stable_hash,
    task_stratum,
)


def task(task_id, question_type="A", tools=()):
    return SimpleNamespace(
        task_id=task_id,
        eval_config={"question_type": question_type},
        enabled_tools=list(tools),
    )


def test_selection_is_deterministic_and_seeded():
    tasks = [task(f"t{i}", "A" if i % 2 else "B") for i in range(20)]
    first = [t.task_id for t in select_stratified(tasks, "financebench", 10, 17)]
    second = [t.task_id for t in select_stratified(list(reversed(tasks)), "financebench", 10, 17)]
    other = [t.task_id for t in select_stratified(tasks, "financebench", 10, 18)]
    assert first == second
    assert first != other
    assert len(first) == len(set(first)) == 10


def test_finance_selection_round_robins_question_types():
    tasks = [task(f"a{i}", "A") for i in range(10)] + [task(f"b{i}", "B") for i in range(10)]
    selected = select_stratified(tasks, "financebench", 8, 7)
    counts = {"A": 0, "B": 0}
    for item in selected:
        counts[task_stratum("financebench", item)] += 1
    assert counts == {"A": 4, "B": 4}


def test_stratified_holdout_window_is_deterministic_and_disjoint():
    tasks = [task(f"{kind}{i}", kind) for kind in "ABC" for i in range(20)]
    first = select_stratified_window(tasks, "financebench", n=12, seed=19, skip=0)
    holdout = select_stratified_window(tasks, "financebench", n=24, seed=19, skip=12)
    repeated = select_stratified_window(
        list(reversed(tasks)), "financebench", n=24, seed=19, skip=12
    )

    first_ids = [x.task_id for x in first]
    holdout_ids = [x.task_id for x in holdout]
    assert set(first_ids).isdisjoint(holdout_ids)
    assert holdout_ids == [x.task_id for x in repeated]
    assert len(holdout_ids) == len(set(holdout_ids)) == 24
    assert {
        kind: sum(task_stratum("financebench", x) == kind for x in holdout)
        for kind in "ABC"
    } == {"A": 8, "B": 8, "C": 8}


def test_explicit_task_indices_preserve_order_and_reject_duplicates(tmp_path):
    manifest = tmp_path / "indices.json"
    manifest.write_text('{"indices": [9, 2, 17]}')
    indices, digest = load_task_indices(manifest)
    assert indices == [9, 2, 17]
    assert len(digest) == 64

    manifest.write_text('{"indices": [9, 2, 9]}')
    try:
        load_task_indices(manifest)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate indices must be rejected")


def test_mcp_stratum_uses_tool_server_set():
    item = task("m1", tools=["wikipedia_search", "calculator_calculate", "wikipedia_get_page"])
    assert task_stratum("mcp-atlas", item) == "calculator+wikipedia"
    assert is_mcp_read_only(item)
    assert not is_mcp_read_only(task("m2", tools=["github_search_repositories"]))


def test_imo_stratum_uses_math_category():
    item = task("i1")
    item.eval_config = {"category": "geometry"}
    assert task_stratum("imo-answerbench", item) == "geometry"


def test_stable_hash_is_sha256_hex():
    value = stable_hash("abc")
    assert len(value) == 64
    assert value == stable_hash("abc")


def test_paired_bootstrap_delta_respects_pairing_and_scale():
    lo, hi = paired_bootstrap_delta([1, 1, 0, 0], [0, 0, 0, 0], draws=2000)
    assert 0.0 <= lo <= 50.0
    assert 50.0 <= hi <= 100.0
    tie_lo, tie_hi = paired_bootstrap_delta([1, 0], [1, 0], draws=500)
    assert tie_lo == tie_hi == 0.0


def test_exact_mcnemar_reports_directional_transitions():
    result = exact_mcnemar([1, 1, 1], [0, 0, 0])
    assert result == {
        "both_pass": 0,
        "a_only": 3,
        "b_only": 0,
        "both_fail": 0,
        "discordant": 3,
        "exact_two_sided_p": 0.25,
    }

    tied = exact_mcnemar([1, 1, 0, 0], [0, 1, 1, 0])
    assert tied["a_only"] == tied["b_only"] == 1
    assert tied["exact_two_sided_p"] == 1.0


def test_langchain_prompt_control_is_pinned_and_protocol_only_adapted():
    planner, executor = prompt_pair("langchain-v0.0.354-native")
    assert planner == LANGCHAIN_PLAN
    assert executor == LANGCHAIN_EXEC
    assert planner.startswith("Let's first understand the problem and devise a plan")
    assert planner.endswith("'<END_OF_PLAN>'")
    assert executor == (
        "Respond to the human as helpfully and accurately as possible. "
        "You have access to the following tools:"
    )

    source = prompt_provenance("langchain-v0.0.354-native")
    assert source["upstream"] == "langchain-ai/langchain"
    assert source["version"] == "v0.0.354"
    assert source["license"] == "MIT"
    assert source["adaptation"] == (
        "StructuredChatAgent textual JSON tool-call serialization omitted; "
        "the runner supplies the same tool schemas through native function calling. "
        "Task-to-plan and plan-to-executor message boundaries are unchanged across arms."
    )


def test_original_prompt_pair_remains_the_production_scaffold():
    planner, executor = prompt_pair("original")
    assert planner
    assert executor
    assert planner != LANGCHAIN_PLAN
    assert executor != LANGCHAIN_EXEC
