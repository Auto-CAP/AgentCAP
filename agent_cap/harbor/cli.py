"""``python -m agent_cap.harbor``: run AgentCAP configurations on Harbor.

Commands::

    adapt imo-answerbench --out datasets/imo-answerbench [--limit 100]
    adapt nl2repo --source path/to/NL2RepoBench --out datasets/nl2repo [--tasks math-verify]
    job   --benchmark imo-answerbench --dataset datasets/imo-answerbench \\
          --teas-agent-yaml imo-answerbench_gptoss_sglang.yaml --out job.yaml
    run   (same options as job)  -> writes job.yaml, runs `harbor run -c`, exports
    export --job-dir jobs/<name> --dataset imo-answerbench --out results/<name>
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


def _add_job_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark", required=True, choices=sorted(jobspec.PRESETS))
    p.add_argument("--dataset", required=True, help="local Harbor dataset dir from `adapt`")
    p.add_argument("--harness", choices=["capagent", "dsh"], default="capagent",
                   help="capagent: AgentCAP strategies on the host; dsh: DeepSeek Harness in the container")
    team = p.add_mutually_exclusive_group()
    team.add_argument("--teas-agent-yaml", help="capagent: TEASBench agent YAML (pipeline/configs/agents/*.yaml)")
    team.add_argument("--role", action="append", help="capagent: ROLE=name=...,base_url=...,api_key=... (repeatable)")
    dsh = p.add_argument_group("DeepSeek Harness (--harness dsh)")
    dsh.add_argument("--dsh-model", help="model id accepted by the endpoint")
    dsh.add_argument("--dsh-base-url", help="API base URL, reachable from inside the task container")
    dsh.add_argument("--dsh-api-key-env", default="DSH_API_KEY",
                     help="name of the env var holding the API key (default DSH_API_KEY); the value is not written to files")
    dsh.add_argument("--dsh-protocol", default="openai-completions",
                     choices=["openai-completions", "openai-responses", "anthropic-messages", "deepseek-messages"])
    dsh.add_argument("--dsh-teammates", type=int, default=0, help="0 = one agent; K = Agent Team with K teammates")
    dsh.add_argument("--dsh-context-window", type=int, default=131072)
    dsh.add_argument("--dsh-max-tokens", type=int, default=16384)
    dsh.add_argument("--dsh-reasoning-effort", help="provider effort id, e.g. high")
    dsh.add_argument("--dsh-version", default="0.2.0-rc.2", help="@deepseek-ai/dsh version installed in the container")
    p.add_argument("--strategy", help="override the preset strategy (single, plan-execute, ...)")
    p.add_argument("--max-turns", type=int)
    p.add_argument("--task-timeout-s", type=float, help="per-task deadline inside the agent")
    p.add_argument("--env", default="docker", help="Harbor environment type (docker, modal, ...)")
    p.add_argument("--env-import-path", help="custom Harbor environment class (module:Class)")
    p.add_argument("--concurrency", type=int, default=1, help="concurrent trials (TEAS concurrency)")
    p.add_argument("--attempts", type=int, default=1, help="attempts per task (seeds)")
    p.add_argument("--task-names", help="comma-separated task names or globs")
    p.add_argument("--n-tasks", type=int)
    p.add_argument("--job-name", default=None)
    p.add_argument("--jobs-dir", default="jobs")


def _job_from_args(args: argparse.Namespace):
    if args.harness == "dsh":
        return _dsh_job_from_args(args)
    if not (args.teas_agent_yaml or args.role):
        raise SystemExit("--harness capagent needs --role ... or --teas-agent-yaml")
    overrides: Dict[str, Any] = {
        "strategy": args.strategy,
        "max_turns": args.max_turns,
        "task_timeout_s": args.task_timeout_s,
    }
    verifier_env: Dict[str, str] = {}
    if args.teas_agent_yaml:
        teas_kwargs, verifier_env = jobspec.load_teas_agent_yaml(Path(args.teas_agent_yaml))
        roles = teas_kwargs.pop("roles")
        overrides = {**teas_kwargs, **{k: v for k, v in overrides.items() if v is not None}}
    else:
        roles = jobspec.roles_from_specs(args.role)
    kwargs = jobspec.capagent_kwargs(args.benchmark, roles, overrides)
    job_name = args.job_name or f"{args.benchmark}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    )
    return config, verifier_env


def _dsh_job_from_args(args: argparse.Namespace):
    if not (args.dsh_model and args.dsh_base_url):
        raise SystemExit("--harness dsh needs --dsh-model and --dsh-base-url")
    kwargs = jobspec.dsh_kwargs(
        args.benchmark,
        {
            "model": args.dsh_model,
            "endpoint_url": args.dsh_base_url,
            "api_key_env": args.dsh_api_key_env,
            "protocol": args.dsh_protocol,
            "teammates": args.dsh_teammates,
            "context_window": args.dsh_context_window,
            "max_tokens": args.dsh_max_tokens,
            "reasoning_effort": args.dsh_reasoning_effort,
            "dsh_version": args.dsh_version,
        },
    )
    mode = f"dsh-team{args.dsh_teammates}" if args.dsh_teammates else "dsh"
    job_name = args.job_name or f"{args.benchmark}-{mode}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        agent_import_path=jobspec.DSH_IMPORT_PATH,
    )
    if not os.environ.get(args.dsh_api_key_env):
        print(f"note: ${args.dsh_api_key_env} is not set; DSH will send the key 'EMPTY'", file=sys.stderr)
    return config, {}


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
    metrics, results = _export(job_dir, out_dir, dataset=args.teas_dataset or args.benchmark, model=args.model)
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

    p = sub.add_parser("job", help="write a Harbor job.yaml for CAPAgent")
    _add_job_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_job)

    p = sub.add_parser("run", help="write job.yaml, run it with Harbor, export results")
    _add_job_args(p)
    p.add_argument("--out", help="where to write job.yaml (default: <jobs-dir>/<job-name>.yaml)")
    p.add_argument("--export-dir")
    p.add_argument("--teas-dataset", help="dataset name for TEAS files (default: --benchmark)")
    p.add_argument("--model", help="model name for TEAS metadata (default: first role's model)")
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
