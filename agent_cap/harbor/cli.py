"""``python -m agent_cap.harbor``: run any agent harness on Harbor and export TEAS files.

Commands::

    adapt imo-answerbench --out datasets/imo-answerbench [--limit 100]
    adapt nl2repo --source path/to/NL2RepoBench --out datasets/nl2repo [--tasks math-verify]
    job   --benchmark nl2repo --dataset datasets/nl2repo --harness codex --model openai/gpt-5.4 --out job.yaml
    run   (same options as job)  -> writes job.yaml, runs `harbor run -c`, exports
    export --job-dir jobs/<name> --dataset nl2repo --out results/<name>

Every harness takes Harbor's model flags (--model, --base-url, --api-key-name,
--api-format, --effort): --harness capagent | dsh | codex | claude-code | ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_cap.harbor import jobspec


HARNESS_HELP = (
    "capagent (default): an AgentCAP strategy on the host. dsh: DeepSeek Harness. "
    "Or any Harbor agent: codex, claude-code, openhands, mini-swe-agent, gemini-cli, opencode, ... "
    "Or an import path module:Class."
)


def _add_job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark", required=True, choices=sorted(jobspec.PRESETS))
    p.add_argument("--dataset", required=True, help="local Harbor dataset dir from `adapt`")
    p.add_argument("--harness", default="capagent", help=HARNESS_HELP)

    model = p.add_argument_group("model (every harness; Harbor's connection flags)")
    model.add_argument("--model", help="model, e.g. openai/gpt-5.4, anthropic/claude-sonnet-4-5, deepseek/deepseek-chat, "
                                       "or an id served at --base-url (also the TEAS model name)")
    model.add_argument("--base-url", help="inference endpoint; for harnesses in the container it must be reachable from there")
    model.add_argument("--api-key-name", "--api-key-env", dest="api_key_name",
                       help="name of the env var holding the API key (the value is never written to files)")
    model.add_argument("--api-format", help="openai_chat_completions, openai_responses, anthropic_messages, ...")
    model.add_argument("--effort", help="reasoning effort, for harnesses that support it")
    model.add_argument("--ak", "--agent-kwarg", dest="agent_kwargs", action="append", default=[], metavar="KEY=VALUE",
                       help="harness option, e.g. --ak teammates=2 (dsh), --ak version=0.130.0 (repeatable; YAML values)")
    model.add_argument("--ae", "--agent-env", dest="agent_env", action="append", default=[], metavar="KEY=VALUE",
                       help="env var for the harness; ${VAR} is resolved by Harbor at run time (repeatable)")
    meter = model.add_mutually_exclusive_group()
    meter.add_argument("--meter", dest="meter", action="store_true", default=None,
                       help="record every model request through AgentCAP's host proxy "
                            "(default for Harbor's agents; off for capagent and dsh, which meter themselves)")
    meter.add_argument("--no-meter", dest="meter", action="store_false")
    model.add_argument("--meter-host", help="address of this host as seen from the task container "
                                            "(default: the container's gateway, docker only)")

    team = p.add_argument_group("capagent team (instead of --model)")
    team_x = team.add_mutually_exclusive_group()
    team_x.add_argument("--teas-agent-yaml", help="TEASBench agent YAML (pipeline/configs/agents/*.yaml)")
    team_x.add_argument("--role", action="append", help="ROLE=name=...,base_url=...,api_key=... (repeatable)")
    team.add_argument("--strategy", help="override the preset strategy (single, plan-execute, ...)")
    team.add_argument("--max-turns", type=int)
    team.add_argument("--task-timeout-s", type=float, help="per-task deadline inside the agent")

    dsh = p.add_argument_group("DeepSeek Harness (--harness dsh); --model/--base-url/--api-key-name also work")
    dsh.add_argument("--dsh-model", help="model id accepted by the endpoint")
    dsh.add_argument("--dsh-base-url", help="API base URL, reachable from inside the task container")
    dsh.add_argument("--dsh-api-key-env", default="DSH_API_KEY",
                     help="name of the env var holding the API key (default DSH_API_KEY); the value is not written to files")
    dsh.add_argument("--dsh-protocol", choices=["openai-completions", "openai-responses", "anthropic-messages", "deepseek-messages"],
                     help="default: from --api-format, else openai-completions")
    dsh.add_argument("--dsh-teammates", type=int, default=0, help="0 = one agent; K = Agent Team with K teammates")
    dsh.add_argument("--dsh-context-window", type=int, default=131072)
    dsh.add_argument("--dsh-max-tokens", type=int, default=16384)
    dsh.add_argument("--dsh-reasoning-effort", help="provider effort id, e.g. high (same as --effort)")
    dsh.add_argument("--dsh-version", default="0.2.0-rc.2", help="@deepseek-ai/dsh version installed in the container")

    run = p.add_argument_group("Harbor job")
    run.add_argument("--env", default="docker", help="Harbor environment type (docker, modal, ...)")
    run.add_argument("--env-import-path", help="custom Harbor environment class (module:Class)")
    run.add_argument("--concurrency", type=int, default=1, help="concurrent trials (TEAS concurrency)")
    run.add_argument("--attempts", type=int, default=1, help="attempts per task (seeds)")
    run.add_argument("--task-names", help="comma-separated task names or globs")
    run.add_argument("--n-tasks", type=int)
    run.add_argument("--job-name", default=None)
    run.add_argument("--jobs-dir", default="jobs")


def _kv(items: List[str], what: str, parse_values: bool) -> Dict[str, Any]:
    import yaml

    out: Dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"{what} must be KEY=VALUE; got {item!r}")
        out[key.strip()] = yaml.safe_load(value) if parse_values and value else value
    return out


def _harness_kwargs(args: argparse.Namespace):
    """Preset + harness-specific flags -> (agent kwargs, verifier env)."""
    harness = args.harness
    verifier_env: Dict[str, str] = {}
    kwargs = jobspec.harness_kwargs(harness, args.benchmark)
    if harness == "capagent":
        overrides: Dict[str, Any] = {
            "strategy": args.strategy,
            "max_turns": args.max_turns,
            "task_timeout_s": args.task_timeout_s,
        }
        roles: Dict[str, Any] = {}
        if args.teas_agent_yaml:
            teas_kwargs, verifier_env = jobspec.load_teas_agent_yaml(Path(args.teas_agent_yaml))
            roles = teas_kwargs.pop("roles")
            overrides = {**teas_kwargs, **{k: v for k, v in overrides.items() if v is not None}}
        elif args.role:
            roles = jobspec.roles_from_specs(args.role)
        elif not args.model:
            raise SystemExit("--harness capagent needs --model (with --base-url), --role ... or --teas-agent-yaml")
        if roles:
            kwargs["roles"] = {str(r): dict(v) for r, v in roles.items()}
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    elif harness == "dsh":
        if args.dsh_model or args.dsh_base_url:  # the original --dsh-* interface
            if not (args.dsh_model and args.dsh_base_url):
                raise SystemExit("--dsh-model and --dsh-base-url go together (or use --model/--base-url)")
            kwargs.update(model=args.dsh_model, endpoint_url=args.dsh_base_url, api_key_env=args.dsh_api_key_env)
            if not os.environ.get(args.dsh_api_key_env):
                print(f"note: ${args.dsh_api_key_env} is not set; DSH will send the key 'EMPTY'", file=sys.stderr)
        elif not args.model:
            raise SystemExit("--harness dsh needs --model (or --dsh-model and --dsh-base-url)")
        kwargs.update(
            teammates=args.dsh_teammates,
            context_window=args.dsh_context_window,
            max_tokens=args.dsh_max_tokens,
            dsh_version=args.dsh_version,
        )
        if args.dsh_protocol:
            kwargs["protocol"] = args.dsh_protocol
        if args.dsh_reasoning_effort:
            kwargs["reasoning_effort"] = args.dsh_reasoning_effort
    else:
        if args.role or args.teas_agent_yaml or args.strategy:
            raise SystemExit(f"--role/--teas-agent-yaml/--strategy are capagent options; {harness} takes --model")
        if args.benchmark == "imo-answerbench" and not _dataset_asks_for_answer_file(Path(args.dataset)):
            print("note: Harbor's agents do not know to write /workspace/answer.txt; adapt IMO-AnswerBench with "
                  "--answer-file-instruction for them, or every task scores 0", file=sys.stderr)
    if args.effort:
        kwargs["reasoning_effort"] = args.effort
    kwargs.update(_kv(args.agent_kwargs, "--ak", parse_values=True))
    return kwargs, verifier_env


def _dataset_asks_for_answer_file(dataset: Path) -> bool:
    try:
        meta = json.loads((dataset / "agentcard_dataset.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(meta.get("answer_file_instruction"))


def _job_from_args(args: argparse.Namespace):
    try:
        jobspec.check_harness(args.harness)
    except ValueError as exc:
        raise SystemExit(str(exc))
    kwargs, verifier_env = _harness_kwargs(args)
    meter = args.meter
    if meter is None:  # CAPAgent and DSH meter themselves; Harbor's agents go through the proxy
        meter = args.harness not in jobspec.OWN_HARNESSES
    if meter and args.harness == "capagent":
        print("note: capagent meters every request on the host already; --meter ignored", file=sys.stderr)
        meter = False
    if meter and args.meter_host:
        kwargs["meter_host"] = args.meter_host
    agent = jobspec.agent_entry(
        args.harness,
        kwargs=kwargs,
        model=args.model,
        base_url=args.base_url,
        api_key_name=args.api_key_name,
        api_format=args.api_format,
        env=_kv(args.agent_env, "--ae", parse_values=False),
        meter=meter,
    )
    short = args.harness.rsplit(":", 1)[-1].lower()
    if args.harness == "dsh" and args.dsh_teammates:
        short = f"dsh-team{args.dsh_teammates}"
    job_name = args.job_name or f"{args.benchmark}-{short}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = jobspec.build_job_config(
        dataset_path=Path(args.dataset).resolve(),
        agent_kwargs=kwargs,
        job_name=job_name,
        jobs_dir=Path(args.jobs_dir).resolve(),
        environment=args.env,
        environment_import_path=args.env_import_path,
        n_concurrent=args.concurrency,
        n_attempts=args.attempts,
        task_names=[t for t in (args.task_names or "").split(",") if t] or None,
        n_tasks=args.n_tasks,
        agent=agent,
    )
    if args.api_key_name and not os.environ.get(args.api_key_name):
        print(f"note: ${args.api_key_name} is not set in this shell", file=sys.stderr)
    return config, verifier_env


def cmd_adapt(args: argparse.Namespace) -> int:
    if args.benchmark == "imo-answerbench":
        from agent_cap.harbor.adapters import imo_answerbench

        indices = [int(x) for x in args.indices.split(",") if x.strip()] if args.indices else None
        dirs = imo_answerbench.generate(
            Path(args.out),
            source=args.source,
            limit=args.limit,
            indices=indices,
            agent_timeout_sec=args.agent_timeout,
            answer_file_instruction=args.answer_file_instruction,
        )
    else:
        from agent_cap.harbor.adapters import nl2repo

        if not args.source:
            print("adapt nl2repo needs --source (an NL2RepoBench checkout)", file=sys.stderr)
            return 2
        tasks = [t for t in (args.tasks or "").split(",") if t] or None
        dirs = nl2repo.generate(
            Path(args.source),
            Path(args.out),
            tasks=tasks,
            agent_timeout_sec=args.agent_timeout,
            require_commit=args.require_commit,
        )
    print(f"wrote {len(dirs)} task(s) to {args.out}")
    return 0


def cmd_job(args: argparse.Namespace) -> int:
    config, verifier_env = _job_from_args(args)
    jobspec.validate_job_config(config)
    path = jobspec.write_job_config(config, Path(args.out))
    print(f"wrote {path}")
    if verifier_env:
        names = ", ".join(sorted(verifier_env))
        print(f"verifier reads {names} from the environment of `harbor run`")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config, verifier_env = _job_from_args(args)
    jobspec.validate_job_config(config)
    job_dir = Path(config["jobs_dir"]) / config["job_name"]
    job_yaml = jobspec.write_job_config(config, Path(args.out) if args.out else job_dir.with_suffix(".yaml"))
    harbor = shutil.which("harbor")
    if not harbor:
        print("`harbor` CLI not found; install with: pip install 'agent-cap[harbor]'", file=sys.stderr)
        return 2
    env = dict(os.environ)
    for key, value in verifier_env.items():
        env.setdefault(key, os.path.expandvars(value))
    # --yes: Harbor otherwise asks before loading [verifier.env] variables.
    print(f"harbor run -c {job_yaml} --yes", file=sys.stderr)
    rc = subprocess.run([harbor, "run", "-c", str(job_yaml), "--yes"], env=env).returncode
    if args.no_export:
        return rc
    if not job_dir.is_dir():
        print(f"no job directory at {job_dir}; nothing to export", file=sys.stderr)
        return rc or 1
    out_dir = Path(args.export_dir) if args.export_dir else Path("results") / "harbor" / config["job_name"]
    metrics, results = _export(job_dir, out_dir, dataset=args.teas_dataset or args.benchmark,
                               model=args.teas_model or args.model)
    print(f"results: {results}" + (f"\nTEAS metrics: {metrics}" if metrics else ""))
    return rc


def _export(job_dir: Path, out_dir: Path, *, dataset: str, model: Optional[str]):
    from agent_cap.harbor.export_teas import export_job

    return export_job(job_dir, out_dir, dataset=dataset, model_name=model)


def cmd_export(args: argparse.Namespace) -> int:
    metrics, results = _export(Path(args.job_dir), Path(args.out), dataset=args.dataset, model=args.model)
    print(f"results: {results}")
    if metrics:
        print(f"TEAS metrics: {metrics}")
        print(json.dumps(json.loads(metrics.read_text())["quality"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_cap.harbor", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("adapt", help="convert a benchmark into Harbor task directories")
    p.add_argument("benchmark", choices=["imo-answerbench", "nl2repo"])
    p.add_argument("--out", required=True)
    p.add_argument("--source", help="imo: local .jsonl/.json or HF dataset id; nl2repo: NL2RepoBench checkout")
    p.add_argument("--limit", type=int, default=0, help="imo: first N problems (AgentCAP order)")
    p.add_argument("--indices", help="imo: comma-separated dataset indices")
    p.add_argument("--tasks", help="nl2repo: comma-separated task names (default: all)")
    p.add_argument("--require-commit", help="nl2repo: fail unless the checkout is at this commit")
    p.add_argument("--agent-timeout", type=int, default=7200)
    p.add_argument("--answer-file-instruction", action="store_true",
                   help="imo: tell the agent to write /workspace/answer.txt (for non-CAPAgent agents)")
    p.set_defaults(func=cmd_adapt)

    p = sub.add_parser("job", help="write a Harbor job.yaml for any harness")
    _add_job_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_job)

    p = sub.add_parser("run", help="write job.yaml, run it with Harbor, export results")
    _add_job_args(p)
    p.add_argument("--out", help="where to write job.yaml (default: <jobs-dir>/<job-name>.yaml)")
    p.add_argument("--export-dir")
    p.add_argument("--teas-dataset", help="dataset name for TEAS files (default: --benchmark)")
    p.add_argument("--teas-model", help="model name for TEAS metadata (default: --model, else the first role's model)")
    p.add_argument("--no-export", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("export", help="Harbor job dir -> results.jsonl + TEAS files")
    p.add_argument("--job-dir", required=True)
    p.add_argument("--dataset", required=True, help="e.g. imo-answerbench, nl2repo")
    p.add_argument("--out", required=True)
    p.add_argument("--model")
    p.set_defaults(func=cmd_export)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
