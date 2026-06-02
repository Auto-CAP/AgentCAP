"""CLI for the multi-agent module.

Examples:

    # Quickest smoke test (no API key required):
    python -m agent_cap.agents --mock --strategy plan-execute \
        --task "What is 2 + 3 * 4?"

    # Real LLMs:
    python -m agent_cap.agents --strategy plan-execute \
        --agent planner=name=gpt-4o,base_url=https://api.openai.com/v1,api_key=$OAI_KEY \
        --agent executor=name=gpt-4o-mini,base_url=https://api.openai.com/v1,api_key=$OAI_KEY \
        --task "Compute (12 + 8) / 4"

    # YAML config:
    python -m agent_cap.agents --config configs/agents_demo.yaml

    # List built-in strategies:
    python -m agent_cap.agents --list-strategies

    # Plug in a custom strategy module:
    python -m agent_cap.agents --load-module mypkg.my_strategies \
        --strategy my-strategy --task "..."
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

import agent_cap.agents.strategies as _builtin_strategies  # noqa: F401 - side effect registers
from agent_cap.agents.agent import Agent
from agent_cap.agents.evaluators import get_evaluator, list_evaluators
from agent_cap.agents.llm import MockLLMClient, RealLLMClient, make_client, resolve_protocol_name
from agent_cap.agents.registry import (
    get_strategy,
    list_strategies,
    load_modules,
)
from agent_cap.agents.strategies import SequentialStrategy
from agent_cap.agents.tools import LocalToolRegistry, build_demo_tools
from agent_cap.agents.types import AgentSpec, ModelEndpoint, Task


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_cap.agents",
        description="AgentCAP multi-agent runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--strategy", default="plan-execute",
                   help="Strategy name (see --list-strategies)")
    p.add_argument("--list-strategies", action="store_true",
                   help="Print available strategies and exit")
    p.add_argument("--list-protocols", action="store_true",
                   help="Print available LLM protocols and exit")
    p.add_argument("--load-module", action="append", default=[],
                   metavar="MOD", help="Import this dotted module before resolving "
                                       "the strategy (repeatable). Use to load "
                                       "@register_strategy code.")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config with `agents:` and optional `strategy:` and `task:`")
    p.add_argument("--agent", action="append", default=[],
                   metavar="ROLE=k=v,k=v",
                   help="Inline agent spec. Repeatable. "
                        "Example: planner=name=gpt-4o,base_url=...,api_key=...")

    quick = p.add_argument_group(
        "single-agent shortcuts",
        "Convenience flags that build one agent role named 'agent' when "
        "--agent / --role / --config do not already define it. Anything set "
        "via --agent overrides these.",
    )
    quick.add_argument("--model", default=None, help="Model name (alias for name=)")
    quick.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    quick.add_argument("--api-key", default=None, help="Bearer API key (use EMPTY for self-hosted)")
    quick.add_argument("--max-tokens", type=int, default=None, help="Per-agent max_tokens")
    quick.add_argument("--temperature", type=float, default=None, help="Per-agent temperature")
    quick.add_argument("--top-p", type=float, default=None, help="Per-agent top_p")
    quick.add_argument("--seed", type=int, default=None, help="Per-agent sampling seed (harmony only)")
    quick.add_argument("--engine", default=None,
                       help="Per-agent serving engine variant (harmony: vllm | sglang)")
    quick.add_argument("--protocol", default=None,
                       help="Per-agent LLM protocol override (openai | harmony | mock | ...)")
    quick.add_argument("--system-prompt", default=None, help="Per-agent system prompt")
    quick.add_argument("--use-streaming", action="store_true", default=None,
                       help="Enable streaming on the per-agent client (sticky)")
    p.add_argument("--agents-file", action="append", default=[],
                   metavar="PATH",
                   help="YAML file containing only `agents:` (+ optional "
                        "`defaults:` / `include:`). Repeatable; later files "
                        "override earlier ones by role name.")
    p.add_argument("--agents-glob", action="append", default=[],
                   metavar="GLOB",
                   help="Glob pattern of YAML files to load as --agents-file. "
                        "e.g. 'specs/*.yaml'")
    p.add_argument("--role", action="append", default=[],
                   metavar="ROLE=AGENT",
                   help="Bind a strategy role to a named agent from the pool. "
                        "Repeatable. e.g. --role planner=gpt4o --role critic=gpt4o "
                        "(both share one endpoint definition).")
    p.add_argument("--task", type=str, default=None,
                   help="Single user prompt. Use --task-file for batch JSONL.")
    p.add_argument("--task-file", type=str, default=None,
                   help="JSONL file of tasks. Each line: {task_id, user_prompt, ...}")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Max tool-use turns per agent run (default: YAML "
                        "`max_turns` or 20 if absent — matches official mcp-atlas)")
    p.add_argument("--sequence", type=str, default=None,
                   help="For --strategy sequential: comma-separated role order")
    p.add_argument("--mock", action="store_true",
                   help="Use deterministic offline LLM (no network, no API key)")
    p.add_argument("--demo-tools", action="store_true",
                   help="Expose a small built-in calc/echo tool registry to agents")
    p.add_argument("--no-tools", action="store_true",
                   help="Disable all tools even if --demo-tools or --config provides them")
    p.add_argument("--tool-backend", default=None,
                   help="Real tool backend: 'mcp' (more to come). "
                        "Use with --mcp-server-url")
    p.add_argument("--mcp-server-url", default=None,
                   help="MCP server URL for --tool-backend mcp")
    p.add_argument("--dataset", default=None,
                   help="Dataset name passed to unified_runner._load_dataset_tasks")
    p.add_argument("--num-tasks", type=int, default=0,
                   help="Cap dataset tasks at N (0 = all)")
    p.add_argument("--evaluator", default=None,
                   help=f"Evaluator name. Built-in: {{{','.join(list_evaluators())}}}")
    p.add_argument("--judge", default=None,
                   metavar="k=v,k=v",
                   help="LLM judge config for --evaluator llm-judge "
                        "(or as fallback for evaluators that support it). "
                        "Example: name=gpt-4o,base_url=https://api.openai.com/v1,"
                        "api_key=$OPENAI_API_KEY")
    p.add_argument("--output-dir", default=None,
                   help="Write JSONL outputs and enable resume mode")
    p.add_argument("--resume", action="store_true",
                   help="Skip task_ids already present in <output-dir>/results.jsonl")
    p.add_argument("--output", type=str, default=None,
                   help="Write JSON result(s) here. Otherwise printed to stdout.")
    p.add_argument("--verbose", "-v", action="count", default=0,
                   help="-v prints per-task result, -vv also dumps turns")
    return p


def _parse_agent_spec(spec: str) -> Tuple[str, Dict[str, Any]]:
    role, sep, rest = spec.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"--agent expects ROLE=k=v,k=v; got: {spec!r}"
        )
    fields: Dict[str, Any] = {}
    for chunk in rest.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, _, v = chunk.partition("=")
        fields[k.strip()] = os.path.expandvars(v.strip())
    return role.strip(), fields


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return _expand_env(data)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _collect_quick_agent_fields(args: argparse.Namespace) -> Dict[str, Any]:
    """Build per-agent fields from single-agent shortcut flags."""
    mapping = [
        ("model", "name"),
        ("base_url", "base_url"),
        ("api_key", "api_key"),
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("seed", "seed"),
        ("engine", "engine"),
        ("protocol", "protocol"),
        ("system_prompt", "system_prompt"),
        ("use_streaming", "use_streaming"),
    ]
    fields: Dict[str, Any] = {}
    for cli_attr, endpoint_key in mapping:
        v = getattr(args, cli_attr, None)
        if v is None:
            continue
        if endpoint_key == "base_url" or endpoint_key == "api_key" or endpoint_key == "name":
            v = os.path.expandvars(str(v))
        fields[endpoint_key] = v
    return fields


def _build_agent_specs(args: argparse.Namespace) -> Dict[str, AgentSpec]:
    """Return role -> AgentSpec.

    Resolves the two YAML layouts:

    A) Inline (legacy / simple): `agents:` keys ARE role names.

    B) Pool + mapping (decoupled): `agents:` is a pool of endpoint defs (keys
       are arbitrary names); `roles:` maps strategy roles to agent names.
       Multiple roles can point to the same agent for endpoint sharing —
       each role still gets its own Agent instance with its own state.
    """
    pool: Dict[str, Dict[str, Any]] = {}
    roles_map: Dict[str, str] = {}

    def merge_layout(cfg: Dict[str, Any], base_dir: Path) -> None:
        nonlocal pool, roles_map
        expanded, this_roles = _resolve_layout(cfg, base_dir)
        pool.update(expanded)
        roles_map.update(this_roles)

    if args.config:
        merge_layout(_load_yaml(args.config), Path(args.config).parent)

    files: List[str] = list(args.agents_file or [])
    for pat in (args.agents_glob or []):
        files.extend(sorted(glob.glob(pat)))
    for path in files:
        merge_layout(_load_yaml(path), Path(path).parent)

    for raw_spec in args.agent:
        role, fields = _parse_agent_spec(raw_spec)
        pool[role] = fields
        roles_map[role] = role

    quick_fields = _collect_quick_agent_fields(args)
    if quick_fields and "agent" not in pool:
        pool["agent"] = quick_fields
        roles_map.setdefault("agent", "agent")
    elif quick_fields and "agent" in pool:
        for k, v in quick_fields.items():
            pool["agent"].setdefault(k, v)

    for raw in args.role:
        role, sep, agent_name = raw.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--role expects ROLE=AGENT; got {raw!r}")
        roles_map[role.strip()] = agent_name.strip()

    if not roles_map:
        roles_map = {name: name for name in pool}

    specs: Dict[str, AgentSpec] = {}
    for role, agent_name in roles_map.items():
        if agent_name not in pool:
            raise ValueError(
                f"role '{role}' references unknown agent '{agent_name}'. "
                f"Pool: {sorted(pool)}"
            )
        specs[role] = AgentSpec.from_dict(role, pool[agent_name])

    return specs


def _resolve_layout(
    cfg: Dict[str, Any], base_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    expanded = _expand_agents_block(cfg, base_dir)
    roles_cfg = cfg.get("roles")
    roles_map: Dict[str, str] = {}
    if isinstance(roles_cfg, dict):
        for role, value in roles_cfg.items():
            if isinstance(value, str):
                roles_map[str(role)] = value
            elif isinstance(value, dict) and "agent" in value:
                replicas = int(value.get("replicas", 1) or 1)
                agent_tpl = str(value["agent"])
                if replicas <= 1:
                    roles_map[str(role)] = agent_tpl
                else:
                    for i in range(replicas):
                        role_i = role.format(i=i) if "{i}" in str(role) else f"{role}-{i}"
                        agent_i = agent_tpl.format(i=i) if "{i}" in agent_tpl else agent_tpl
                        roles_map[role_i] = agent_i
            else:
                raise ValueError(
                    f"`roles.{role}` must be a string agent name or "
                    f"{{agent: name, replicas: N}}"
                )
    elif roles_cfg is not None:
        raise ValueError("`roles:` must be a mapping role->agent_name")

    return expanded, roles_map


def _expand_agents_block(cfg: Dict[str, Any], base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Resolve `include:`, apply `defaults:`, and expand `replicas:`.

    Returns: {role_name: merged_agent_dict}. Roles from later includes override
    earlier ones; the top-level `agents:` block wins over included files.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for inc in (cfg.get("include") or []):
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = base_dir / inc_path
        inc_cfg = _load_yaml(str(inc_path))
        for role, raw in _expand_agents_block(inc_cfg, inc_path.parent).items():
            out[role] = raw

    defaults = cfg.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("`defaults:` must be a mapping")

    agents_cfg = cfg.get("agents") or {}
    if not isinstance(agents_cfg, dict):
        raise ValueError("`agents:` must be a mapping role->{...}")

    for role, raw in agents_cfg.items():
        if not isinstance(raw, dict):
            raise ValueError(f"agent '{role}' must be a mapping")
        replicas = int(raw.get("replicas", 1) or 1)
        clean = {k: v for k, v in raw.items() if k != "replicas"}
        if replicas <= 1:
            out[str(role)] = _merge_dicts(defaults, clean)
        else:
            for i in range(replicas):
                rolei = f"{role}-{i}"
                inst = copy.deepcopy(clean)
                if "name" in inst and "{i}" in str(inst["name"]):
                    inst["name"] = str(inst["name"]).format(i=i)
                if "base_url" in inst and "{i}" in str(inst["base_url"]):
                    inst["base_url"] = str(inst["base_url"]).format(i=i)
                out[rolei] = _merge_dicts(defaults, inst)

    return out


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        merged[k] = v
    return merged


def _load_tasks(args: argparse.Namespace, config: Dict[str, Any]) -> List[Task]:
    tasks: List[Task] = []
    if args.task_file:
        with open(args.task_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tasks.append(Task.from_dict(json.loads(line)))
    if args.task:
        tasks.append(Task(task_id="cli-task", user_prompt=args.task))

    dataset = args.dataset or config.get("dataset")
    if dataset:
        from agent_cap.agents.adapters import load_dataset_as_tasks

        n = int(args.num_tasks or config.get("num_tasks") or 0)
        tasks.extend(Task.from_dict(d) for d in load_dataset_as_tasks(str(dataset), n))

    if not tasks and config.get("task"):
        tasks.append(Task.from_dict({"user_prompt": config["task"], "task_id": "config-task"}))
    if not tasks and config.get("tasks"):
        for i, t in enumerate(config["tasks"]):
            if isinstance(t, str):
                tasks.append(Task(task_id=f"task-{i}", user_prompt=t))
            elif isinstance(t, dict):
                tasks.append(Task.from_dict(t))
    if not tasks:
        tasks.append(Task(task_id="default", user_prompt="Say hello and stop."))
    return tasks


def _instantiate_strategy(args: argparse.Namespace):
    cls = get_strategy(args.strategy)
    if cls is SequentialStrategy:
        seq = [r.strip() for r in (args.sequence or "").split(",") if r.strip()]
        return SequentialStrategy(sequence=seq or None)
    return cls()


async def _run_async(args: argparse.Namespace) -> int:
    load_modules(args.load_module)

    if args.list_strategies:
        for name in list_strategies():
            print(name)
        return 0

    if args.list_protocols:
        from agent_cap.agents.llm import list_protocols
        for name in list_protocols():
            print(name)
        return 0

    config_data: Dict[str, Any] = _load_yaml(args.config) if args.config else {}
    if config_data.get("strategy") and args.strategy == "plan-execute":
        args.strategy = str(config_data["strategy"])

    specs = _build_agent_specs(args)
    if not specs:
        if args.mock:
            specs = _default_mock_specs(args.strategy)
        else:
            print("ERROR: no agents provided. Use --agent ... or --config or --mock.",
                  file=sys.stderr)
            return 2

    strategy = _instantiate_strategy(args)
    strategy.max_turns = int(args.max_turns if args.max_turns is not None else config_data.get("max_turns", 20))
    tasks = _load_tasks(args, config_data)

    evaluator_name = args.evaluator or config_data.get("evaluator")
    judge_cfg = _parse_judge_config(args.judge, config_data.get("judge"))
    evaluator: Any = (
        get_evaluator(evaluator_name, **judge_cfg) if evaluator_name else None
    )
    if evaluator_name and evaluator is None:
        print(f"ERROR: unknown evaluator '{evaluator_name}'. "
              f"Available: {list_evaluators()}", file=sys.stderr)
        return 2

    out_dir_value = args.output_dir or config_data.get("output_dir")
    out_dir = Path(out_dir_value).resolve() if out_dir_value else None
    done: Dict[str, Dict[str, Any]] = {}
    results_path: Optional[Path] = None
    output_data_path: Optional[Path] = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "results.jsonl"
        output_data_path = out_dir / "output-data.jsonl"
        if args.resume and results_path.exists():
            done = _load_resume(results_path)
            print(f"resume: {len(done)} task(s) already complete in {results_path}",
                  file=sys.stderr)

    async def run_all(llm, session=None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        sess = session if session is not None else getattr(llm, "_session", None)
        tools: Any = await _resolve_tools(args, config_data, session=sess)
        try:
            res_f = results_path.open("a", encoding="utf-8") if results_path else None
            out_f = output_data_path.open("w", encoding="utf-8") if output_data_path else None
            try:
                for i, task in enumerate(tasks):
                    if task.task_id in done:
                        results.append(done[task.task_id])
                        _emit_progress(i, task, done[task.task_id])
                        continue
                    if hasattr(tools, "set_task_allowlist"):
                        unified = (task.metadata or {}).get("_unified_task")
                        et = getattr(unified, "enabled_tools", None) if unified else None
                        tools.set_task_allowlist(et)
                    agents = _build_agents(
                        specs, llm, tools,
                        session=sess, force_mock=args.mock,
                    )
                    task_sys = (task.metadata or {}).get("system_prompt") or ""
                    if task_sys:
                        for ag in agents.values():
                            if not ag.state.messages or ag.state.messages[0].get("role") != "system":
                                ag.state.messages.insert(0, {"role": "system", "content": task_sys})
                            else:
                                ag.state.messages[0] = {"role": "system", "content": task_sys}
                    try:
                        run_res = await strategy.run(task, agents, tools)
                        row = _serialize_result(run_res, args.verbose)
                    except Exception as exc:
                        row = {
                            "task_id": task.task_id,
                            "strategy": args.strategy,
                            "output_text": "",
                            "e2e_latency_s": 0.0,
                            "errors": [f"{type(exc).__name__}: {exc}"[:500]],
                            "num_turns": 0,
                        }
                    if evaluator is not None and not row.get("errors"):
                        ev = evaluator.evaluate(_eval_meta(task), row.get("output_text", "") or "")
                        row["eval_passed"] = ev.passed
                        row["eval_score"] = ev.score
                        row["eval_details"] = ev.details
                    elif row.get("errors"):
                        row["eval_passed"] = False
                        row["eval_score"] = 0.0
                        row["eval_details"] = {"evaluator": "skipped", "reason": "task errored"}
                    if res_f:
                        res_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        res_f.flush()
                    if out_f:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                    results.append(row)
                    _emit_progress(i, task, row)
            finally:
                if res_f:
                    res_f.close()
                if out_f:
                    out_f.close()
        finally:
            if tools is not None and hasattr(tools, "teardown"):
                try:
                    await tools.teardown()
                except Exception:
                    pass
        return results

    if args.mock:
        results = await run_all(MockLLMClient())
    else:
        connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
        timeout = aiohttp.ClientTimeout(total=1800, connect=60, sock_connect=60, sock_read=1800)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            results = await run_all(None, session=session)

    _print_summary(results, evaluator_name)

    if out_dir is None:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _eval_meta(task: Task) -> Dict[str, Any]:
    meta = dict(task.metadata or {})
    src = meta.get("_unified_task")
    if src is not None:
        eval_cfg = getattr(src, "eval_config", None) or {}
        if isinstance(eval_cfg, dict):
            meta.update(eval_cfg)
            meta["eval_config"] = eval_cfg
    return meta


def _load_resume(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            tid = row.get("task_id")
            if isinstance(tid, str) and tid:
                out[tid] = row
    return out


def _emit_progress(i: int, task: Task, row: Dict[str, Any]) -> None:
    score = row.get("eval_score")
    score_s = f" score={float(score):.3f}" if isinstance(score, (int, float)) else ""
    passed = row.get("eval_passed")
    passed_s = f" passed={bool(passed)}" if passed is not None else ""
    print(f"[{i}] {task.task_id}: {row.get('e2e_latency_s', 0):.2f}s"
          f"{score_s}{passed_s}", file=sys.stderr)


def _print_summary(results: List[Dict[str, Any]], evaluator_name: Optional[str]) -> None:
    n = len(results)
    if n == 0:
        return
    scores = [float(r.get("eval_score") or 0.0) for r in results if r.get("eval_score") is not None]
    passes = [bool(r.get("eval_passed")) for r in results if r.get("eval_passed") is not None]
    line = f"FINAL: n={n}"
    if scores:
        line += f"  acc={sum(scores) / max(len(scores), 1):.3f}"
    if passes:
        line += f"  task_coverage={sum(passes) / max(len(passes), 1):.3f}"
    if evaluator_name:
        line += f"  evaluator={evaluator_name}"
    print(line, file=sys.stderr)


async def _resolve_tools(
    args: argparse.Namespace,
    config: Dict[str, Any],
    session: Optional[aiohttp.ClientSession],
) -> Optional[object]:
    if args.no_tools:
        return None
    backend_name = (args.tool_backend or config.get("tool_backend") or "").strip().lower()
    if backend_name in ("mcp", "mcp-atlas"):
        if session is None:
            raise RuntimeError("MCP backend requires --mock disabled (real LLM session)")
        from agent_cap.agents.adapters import MCPProviderAdapter

        url = args.mcp_server_url or config.get("mcp_server_url") or "http://localhost:1984"
        return MCPProviderAdapter(session=session, mcp_server_url=str(url))
    if backend_name in ("math-python", "math_python", "mathpython", "python"):
        from agent_cap.agents.adapters import MathPythonProviderAdapter

        return MathPythonProviderAdapter()
    if backend_name and backend_name != "demo":
        raise ValueError(
            f"Unknown --tool-backend '{backend_name}'. "
            "Supported: mcp, math-python, demo."
        )
    return _build_tools(args, config)


def _parse_judge_config(cli_value, yaml_value) -> Dict[str, Any]:
    """Combine `--judge k=v,k=v` and YAML `judge:` block. CLI overrides YAML."""
    cfg: Dict[str, Any] = {}
    if isinstance(yaml_value, dict):
        cfg.update(yaml_value)
    if cli_value:
        for chunk in str(cli_value).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            k, _, v = chunk.partition("=")
            cfg[k.strip()] = os.path.expandvars(v.strip())
    return cfg


def _build_agents(specs, llm, tools, session=None, force_mock=False):
    """Build one Agent per spec, auto-routing protocol per endpoint.

    `llm` is used when force_mock=True (single shared MockLLMClient).
    Otherwise, each spec's endpoint is routed through the protocol registry,
    so e.g. a gpt-oss endpoint gets Harmony while a Qwen endpoint gets OpenAI.
    """
    client_cache: Dict[str, Any] = {}

    def _client_for(spec) -> Any:
        if force_mock:
            return llm
        proto = resolve_protocol_name(spec.endpoint)
        if proto not in client_cache:
            client_cache[proto] = make_client(spec.endpoint, session=session)
        return client_cache[proto]

    return {role: Agent(spec, _client_for(spec), tools) for role, spec in specs.items()}


def _build_tools(args: argparse.Namespace, config: Dict[str, Any]) -> Optional[LocalToolRegistry]:
    if args.demo_tools or config.get("demo_tools") or args.mock:
        return build_demo_tools()
    return None


def _default_mock_specs(strategy_name: str) -> Dict[str, AgentSpec]:
    endpoint = ModelEndpoint(name="mock-model")
    if strategy_name == "single":
        return {"agent": AgentSpec(role="agent", endpoint=endpoint)}
    if strategy_name == "supervisor":
        return {
            "supervisor": AgentSpec(role="supervisor", endpoint=ModelEndpoint(name="mock-planner")),
            "worker": AgentSpec(role="worker", endpoint=endpoint),
        }
    if strategy_name == "sequential":
        return {
            "writer": AgentSpec(role="writer", endpoint=endpoint),
            "reviewer": AgentSpec(role="reviewer", endpoint=ModelEndpoint(name="mock-reviewer")),
        }
    return {
        "planner": AgentSpec(role="planner", endpoint=ModelEndpoint(name="mock-planner")),
        "executor": AgentSpec(role="executor", endpoint=endpoint),
    }


def _serialize_result(res, verbose: int) -> Dict[str, Any]:
    payload = res.to_dict()
    if verbose >= 2:
        payload["turns"] = [
            {
                "role": t.role,
                "model": t.model,
                "assistant_content": t.assistant.get("content"),
                "tool_calls": [{"name": (tc.get("function") or {}).get("name"),
                                "arguments": (tc.get("function") or {}).get("arguments")}
                               for tc in t.tool_calls],
                "tool_results": t.tool_results,
                "latency_s": round(t.latency_s, 4),
            }
            for t in res.turns
        ]
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        return asyncio.run(_run_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
