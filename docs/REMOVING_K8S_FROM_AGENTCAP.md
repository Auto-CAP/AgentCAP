# Removing the built-in k8s provisioning from AgentCAP

**Status: not yet actioned.** The k8s code described here is still present and still the
default for `--sweagent-deployment k8s`. This document is the plan for deleting it once
TEASBench's providers are proven on a real EIDF run.

## Why this code exists, and why it is now redundant

AgentCAP owns *benchmark semantics*: agent strategy, call limits, streaming metrics, official
SWE-bench grading. It consumes exactly three things from its environment:

1. an OpenAI-compatible LLM endpoint (`--base-url`),
2. per SWE-bench task, a swe-rex sandbox endpoint `{host, port, auth_token}`,
3. per SWE-bench eval, an exec container built from the official instance image.

*How* those come to exist — k8s Jobs, local docker, Modal, vast.ai — is a **deployment
scenario**, which is TEASBench's concern. The k8s implementation landed in AgentCAP only
because EIDF was the first cluster to need it and there was nowhere else to put it.

There is now somewhere else: `pipeline/k8s/lib/k8s_pod_providers/providers.py` in TEASBench, offering
`InClusterK8sProvider` (the default; talks to sandbox pod IPs directly) and
`PortForwardK8sProvider` (a faithful port of the code below, for driving a run from a login
node). AgentCAP reaches them through the dotted-path seam already added to
`get_sandbox_provider` / `get_exec_provider`:

```
--sandbox-provider k8s_pod_providers:InClusterK8sProvider
--exec-provider    k8s_pod_providers:InClusterK8sProvider
```

Kubernetes is the *only* substrate that needs a provider at all: `docker` and `modal` are
native swe-rex deployment types, which is why no equivalent removal is needed for them.

## Preconditions — do not start until all four hold

- [ ] TEASBench's `InClusterK8sProvider` has completed at least one full SWE-bench Lite
      curated-100 run on EIDF, and its `metrics_*.json` has been diffed against a reference
      run from the built-in provider (expect accuracy within run-to-run noise; expect
      per-task `e2e_latency_s` to *drop*, since the port-forward hop disappears).
- [ ] `PortForwardK8sProvider` has been smoke-tested from a login node, since it is the
      fallback if EIDF declines the RBAC in `pipeline/k8s/rbac/teasbench-runner-rbac.yaml`.
- [ ] Nothing in `scripts/` or `k8s/` still passes `--sweagent-deployment k8s` (see the
      deletion list below — those scripts go at the same time).
- [ ] No external consumer depends on `SWEBENCH_K8S_NAMESPACE`, `SWEBENCH_K8S_AUTH_TOKEN`,
      or `SWEBENCH_SANDBOX_PROVIDER`. Grep the org, not just this repo.

## What to delete

### `agent_cap/agents/sandbox_providers.py`

| Symbol | Line (at time of writing) |
|---|---|
| `_K8S_PORT_LOCK` | 45 |
| `_K8S_AUTH_TOKEN` | 46 |
| `_k8s_next_port()` | 49 |
| `_kubectl()` | 59 |
| `class _K8sSidecar` | 90 |
| `class K8sSandboxProvider` | 239 |
| `class K8sExecContainer` | 413 |
| `class K8sExecProvider` | 482 |
| the `"k8s"` entry in `_PROVIDERS` | 318 |
| the `"k8s"` entry in `_EXEC_PROVIDERS` | 513 |

**Keep** `SandboxEndpoint`, `SandboxProvider`, `ExecHandle`, `ExecProvider`,
`_resolve_dotted_path`, `get_sandbox_provider`, `get_exec_provider`, and
`HttpSandboxProvider`. `HttpSandboxProvider` is deliberately retained even though nothing
uses it: it is the escape hatch for a future substrate that the agent process cannot reach
directly, and it costs nothing to keep.

Once `_PROVIDERS` / `_EXEC_PROVIDERS` are empty, decide whether to drop the registries
entirely and make dotted paths the only non-URL form. Preferred: **keep them** as empty
dicts, so the error message can still enumerate built-ins if any are ever re-added, and so
`get_*_provider` keeps one code shape.

### `agent_cap/agents/strategies_sweagent.py`

- The `deployment == "k8s"` branch (line 115) and its `provider_name` / `provider_kwargs`
  resolution (lines 118–125), including the `SWEBENCH_SANDBOX_PROVIDER` env fallback and the
  `k8s_namespace` default. After removal, a provider must be supplied explicitly.
- The docstring line 21 listing `docker | modal | local | k8s`.
- The `f"k8s sidecar failed: {exc}"` error string (line 136) — make it substrate-neutral,
  e.g. `f"sandbox acquire failed: {exc}"`.

## The `_swebench_image()` trap — read this before deleting anything

`_swebench_image()` (line 45) does:

```python
if deployment in ("modal", "k8s"):
    return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"
```

That `_1776_` substitution and the `docker.io/` prefix are **registry naming semantics**, not
Kubernetes semantics — they describe how the official SWE-bench images are published, which
is why `modal` shares the branch. Deleting `"k8s"` from that tuple without thinking will
silently produce unpullable image names for every EIDF run.

Re-key it on the actual distinction, which is *"is this image pulled from a remote registry
or built locally?"*. Suggested shape:

```python
_REGISTRY_DEPLOYMENTS = ("modal", "remote")

def _swebench_image(instance_id: str, deployment: str, image_repo: str) -> str:
    iid = instance_id.lower().replace("/", "__")
    if deployment in _REGISTRY_DEPLOYMENTS:
        return f"docker.io/swebench/sweb.eval.x86_64.{iid.replace('__', '_1776_')}:latest"
    ...
```

Add a unit test pinning both forms for one known instance id *before* touching this, so the
change is provably behaviour-preserving.

## Renames to do at the same time

Both are user-visible; do them together so there is one migration, not two.

1. **`--sweagent-deployment k8s` → `remote`.** "remote" is the swe-rex deployment type
   AgentCAP legitimately owns (it is what actually gets passed as
   `--env.deployment.type remote`, line 139); "k8s" names a substrate AgentCAP will no longer
   know about. Accept `k8s` as a deprecated alias for one release, mapping it to `remote`
   with a `DeprecationWarning`.
2. **`swebench-k8s` evaluator → `swebench-remote`.** The alias already exists (both names are
   registered on `SWEBenchK8sEvaluator`), so this is just retiring the old name and renaming
   the class to `SWEBenchRemoteEvaluator`. Note its `finalize()` writes reports into
   `out_dir / "eval_k8s"` — rename that to `eval_remote`, and check nothing downstream
   (TEASBench's results collection, `postprocessing/`) globs for `eval_k8s`.

Also rename the surviving env knobs for consistency: `SWEBENCH_EVAL_TIMEOUT` is fine;
`SWEBENCH_K8S_NAMESPACE` becomes dead once the built-in provider goes (TEASBench uses
`TEASBENCH_K8S_NAMESPACE`).

## Scripts to delete, and what replaces each

These encode a *login-node driver* model — conda env on the head node, `kubectl port-forward`
tunnels, a hand-written queue loop. TEASBench's model is one self-contained Job per
experiment, generated from a CSV row. The functions map across; the scripts do not.

| Delete | Replaced by |
|---|---|
| `k8s/launch_llm_server.sh` | `@agentic_server_command@` in `pipeline/templates/agentic.yaml` |
| `k8s/port_forward_llm.sh` | nothing — the driver runs in-cluster, so there is no tunnel |
| `k8s/run_one_experiment.sh` | one generated k8s Job |
| `k8s/master_queue.sh` | the experiments CSV + `pipeline/generate.py` + `pipeline/submit_job.sh` |
| `k8s/mcp-atlas-sidecar.yaml` | the `sidecar_containers` block in `pipeline/configs/config.yaml` |
| `scripts/run_swebench_k8s_100.sh` | `experiments/swe-bench-lite-eidf.csv` + the generated Job |
| `scripts/run_mcpatlas_k8s_60.sh` | `experiments/mcp-atlas-eidf.csv` + the generated Job |

Keep `scripts/patch_sweagent_streaming.py` — it patches SWE-agent for streaming metrics,
which is benchmark instrumentation, not deployment. TEASBench's job template invokes it.

Keep `k8s/patch_sglang_gptoss.py` only if it is still needed by a supported engine version;
otherwise it goes with the rest.

## Verification after removal

1. `grep -rn "k8s\|kubectl" agent_cap/` returns nothing outside comments and the
   `--sweagent-deployment` alias shim.
2. `python -m pytest tests/test_sandbox_provider_resolution.py` still passes.
3. Run the agentic smoke experiment through TEASBench
   (`experiments/smoke_tests_agentic.csv`, `num_tasks: 2`) end to end on EIDF.
4. Run the full SWE-bench Lite curated-100 through TEASBench and diff `metrics_*.json`
   against the reference run captured in the first precondition. Quality (`quality.acc`)
   must match within run-to-run noise; `performance.*` may legitimately differ.
5. Confirm `agent_cap` imports cleanly in an environment with **no** `kubectl` on `$PATH` and
   no kubeconfig — that is the real proof the dependency is gone.
