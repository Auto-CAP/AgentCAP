from types import SimpleNamespace

from scripts.aggregate_rebuttal_prompt_ablation import paired_bootstrap_delta
from scripts.rebuttal_prompt_ablation import (
    is_mcp_read_only,
    select_stratified,
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
