"""SWE-bench evaluator for the unified agents CLI.

Per-task `evaluate(task_meta, patch)` accumulates into an in-process
predictions buffer; on the first call it returns score=0/passed=False
(pending). The CLI runs `finalize()` after the loop to call
`swebench.harness.run_evaluation` once on all collected patches, then
patches each row's eval fields in results.jsonl.

Registered name: "swebench".

`task_meta` must include `eval_config.instance_id` (set by the
unified_runner swe_bench_lite/pro loader).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

from agent_cap.agents.evaluators import EvalResult, register_evaluator


def unique_run_id(base: str, out_dir: Path) -> str:
    """A per-run swebench run_id derived from the run's output directory.

    swebench.harness caches results by (run_id, model_name_or_path, instance);
    a shared/constant run_id makes every run after the first report "already
    run, skipping" and reuse the first run's results. out_dir encodes
    model/hw/timestamp, so it disambiguates runs.
    """
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-",
                 "_".join(Path(out_dir).resolve().parts[-5:])).strip("-")
    return f"{base}__{tag}" if tag else base


@register_evaluator("swebench")
class SWEBenchEvaluator:
    """Official grading via `swebench.harness.run_evaluation` (needs a local
    docker daemon by default).

    Set env SWEBENCH_HARNESS_MODAL=1 to pass `--modal true` to the harness
    instead, which grades each instance in a Modal sandbox rather than a
    local docker container. This is what the archived Vast.ai runs did by
    hand (see TEAS_Results_Private/agentic/vastai/.../run.sh).
    """

    def __init__(
        self,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        run_id: str = "agentcap_unified",
        max_workers: int = 8,
        model_name: str = "agentcap-unified",
    ) -> None:
        self.dataset = dataset
        self.run_id = run_id
        self.max_workers = int(max_workers)
        self.model_name = str(model_name).replace("/", "-").replace(":", "-")
        self._buffer: Dict[str, str] = {}
        self._lock = threading.Lock()

    def evaluate(self, task_meta: Dict[str, Any], output_text: str) -> EvalResult:
        eval_cfg = (task_meta.get("eval_config") or {})
        iid = eval_cfg.get("instance_id") or task_meta.get("instance_id") or ""
        if not iid:
            return EvalResult(
                passed=False, score=0.0,
                details={"evaluator": "swebench", "error": "missing instance_id"},
            )
        if not output_text.strip():
            return EvalResult(
                passed=False, score=0.0,
                details={"evaluator": "swebench", "error": "empty patch", "instance_id": iid},
            )
        with self._lock:
            self._buffer[iid] = output_text
        return EvalResult(
            passed=False, score=0.0,
            details={"evaluator": "swebench", "instance_id": iid, "status": "pending"},
        )

    def finalize(self, out_dir: Path) -> Dict[str, Dict[str, Any]]:
        if not self._buffer:
            return {}
        # Unique run_id per run. swebench.harness caches results by
        # (run_id, model_name_or_path, instance) under logs/run_evaluation/,
        # so a shared/constant run_id makes every run after the first report
        # "N instances already run, skipping" and silently reuse the first
        # run's results (identical acc across models).
        run_id = unique_run_id(self.run_id, out_dir)
        preds = [
            {"instance_id": iid, "model_patch": patch, "model_name_or_path": self.model_name}
            for iid, patch in self._buffer.items()
        ]
        preds_path = out_dir / "predictions.json"
        preds_path.write_text(json.dumps(preds, indent=2))
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", self.dataset,
            "--predictions_path", str(preds_path),
            "--max_workers", str(self.max_workers),
            "--run_id", run_id,
            "--cache_level", "instance",
        ]
        if os.environ.get("SWEBENCH_HARNESS_MODAL") == "1":
            cmd += ["--modal", "true"]
        log_path = out_dir / "swebench_eval.log"
        with open(log_path, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600 * 6)
        reports_dir = Path("logs/run_evaluation") / run_id
        results: Dict[str, Dict[str, Any]] = {}
        if reports_dir.exists():
            for model_dir in reports_dir.glob("*"):
                for rep in model_dir.glob("*/report.json"):
                    iid = rep.parent.name
                    try:
                        info = json.loads(rep.read_text()).get(iid, {})
                        results[iid] = {
                            "resolved": bool(info.get("resolved")),
                            "details": info,
                        }
                    except Exception:
                        pass
        return results


@register_evaluator("swebench-remote")
@register_evaluator("swebench-k8s")
class SWEBenchK8sEvaluator(SWEBenchEvaluator):
    """Official SWE-bench grading without a docker daemon.

    Per prediction: acquire an exec container from the official instance
    image (via an exec provider — see agent_cap.agents.sandbox_providers),
    apply the model patch (git apply, then `patch --fuzz=5` fallback — same
    as the harness), execute the TestSpec eval script, pull the log back,
    and grade it locally with `swebench.harness.grading.get_eval_report`.
    Semantics match `swebench.harness.run_evaluation`.

    Registered as both "swebench-k8s" (original name) and "swebench-remote"
    (substrate-neutral alias — the exec provider need not be k8s).

    Env knobs: AGENTCAP_EXEC_PROVIDER (exec provider name/dotted-path,
    default "k8s" — see sandbox_providers.get_exec_provider),
    SWEBENCH_K8S_NAMESPACE (default eidf230ns, read by the built-in "k8s"
    exec provider), SWEBENCH_EVAL_TIMEOUT (per-instance test timeout,
    default 1800s).
    """

    def finalize(self, out_dir: Path) -> Dict[str, Dict[str, Any]]:
        if not self._buffer:
            return {}
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from datasets import load_dataset
        from swebench.harness.grading import get_eval_report
        from swebench.harness.test_spec.test_spec import make_test_spec

        eval_timeout = int(os.environ.get("SWEBENCH_EVAL_TIMEOUT", "1800"))

        preds = [
            {"instance_id": iid, "model_patch": patch, "model_name_or_path": self.model_name}
            for iid, patch in self._buffer.items()
        ]
        (out_dir / "predictions.json").write_text(json.dumps(preds, indent=2))

        split = "test"
        ds = load_dataset(self.dataset, split=split)
        inst_map = {ex["instance_id"]: ex for ex in ds}

        from agent_cap.agents.sandbox_providers import get_exec_provider

        provider = get_exec_provider(os.environ.get("AGENTCAP_EXEC_PROVIDER", "k8s"))

        def eval_one(iid: str, patch: str) -> Dict[str, Any]:
            inst = inst_map.get(iid)
            if inst is None:
                return {"resolved": False, "details": {"error": "instance not in dataset"}}
            spec = make_test_spec(inst, namespace="swebench")
            image = f"docker.io/{spec.instance_image_key}"
            inst_dir = out_dir / "eval_k8s" / iid
            inst_dir.mkdir(parents=True, exist_ok=True)

            # Reuse a completed grading of the identical patch.
            #
            # finalize() grades everything in _buffer, and a --resume pass
            # re-feeds every already-finished row into the evaluator. Without
            # this, a run that retries infrastructure failures re-grades all of
            # its passing tasks on every attempt: a fresh pod and a full test
            # suite each time, for a patch that has not changed and cannot
            # produce a different answer for any legitimate reason.
            #
            # It also stops being merely wasteful. Instances whose tests call
            # out to a public service -- psf/requests hits httpbin.org -- get
            # rate-limited into 503s, so the same patch grades resolved on one
            # attempt and unresolved on the next. Re-grading is what
            # manufactures that flapping.
            #
            # Keyed on the patch text, so a retry that produces a *different*
            # patch is always graded afresh. AGENTCAP_EVAL_NO_CACHE=1 forces a
            # full re-grade.
            #
            # The key is graded_patch.diff, NOT patch.diff. patch.diff is
            # written before grading starts, and grading can end without
            # writing a report -- a patch that fails to apply returns early,
            # and a crash or timeout can land anywhere. Keying on it would let
            # this sequence serve a stale verdict: patch A graded, retry with
            # patch B overwrites patch.diff, B fails to apply, report.json
            # still holds A's result, and the next attempt matches B against
            # B and returns A's verdict. graded_patch.diff is written only
            # after report.json, so its presence means that report describes
            # exactly this patch.
            report_path = inst_dir / "report.json"
            graded_patch_path = inst_dir / "graded_patch.diff"
            if (os.environ.get("AGENTCAP_EVAL_NO_CACHE") != "1"
                    and report_path.is_file() and graded_patch_path.is_file()):
                try:
                    if graded_patch_path.read_text() == patch:
                        info = json.loads(report_path.read_text()).get(iid, {})
                        if isinstance(info, dict) and isinstance(info.get("resolved"), bool):
                            return {"resolved": bool(info["resolved"]),
                                    "details": info, "cached": True}
                except (OSError, ValueError):
                    pass  # unreadable cache is not evidence: grade it properly

            box = None
            try:
                try:
                    box = provider.acquire_exec(image, iid)
                except Exception as exc:
                    return {"resolved": False, "details": {"error": str(exc)[:200]}}

                patch_path = inst_dir / "patch.diff"
                patch_path.write_text(patch)
                eval_sh = inst_dir / "eval.sh"
                eval_sh.write_text(spec.eval_script)
                box.cp(str(patch_path), "/tmp/patch.diff")
                box.cp(str(eval_sh), "/eval.sh")

                # Apply patch with the same fallback chain as the harness.
                apply_cmd = (
                    "cd /testbed && ("
                    "git apply -v /tmp/patch.diff || "
                    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff)"
                )
                r = box.exec(apply_cmd)
                if r.returncode != 0:
                    (inst_dir / "apply.log").write_text(r.stdout + "\n" + r.stderr)
                    return {"resolved": False,
                            "details": {"error": "patch apply failed",
                                        "stderr": r.stderr[-500:]}}

                r = box.exec(f"timeout {eval_timeout} bash /eval.sh 2>&1 || true",
                             timeout=eval_timeout + 120)
                log_path = inst_dir / "test_output.txt"
                log_path.write_text(r.stdout or "")

                report = get_eval_report(
                    test_spec=spec,
                    prediction={"instance_id": iid, "model_patch": patch,
                                "model_name_or_path": self.model_name},
                    test_log_path=str(log_path),
                    include_tests_status=True,
                )
                info = report.get(iid, {})
                (inst_dir / "report.json").write_text(json.dumps(report, indent=2))
                # Written last, and only on the path that produced a report:
                # this file is what marks the report as describing this exact
                # patch. See the cache check above.
                (inst_dir / "graded_patch.diff").write_text(patch)
                return {"resolved": bool(info.get("resolved")), "details": info}
            except Exception as exc:
                return {"resolved": False, "details": {"error": str(exc)[:300]}}
            finally:
                if box is not None:
                    provider.release_exec(box)

        results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futs = {pool.submit(eval_one, iid, patch): iid
                    for iid, patch in self._buffer.items()}
            done = 0
            for fut in as_completed(futs):
                iid = futs[fut]
                try:
                    results[iid] = fut.result()
                except Exception as exc:
                    results[iid] = {"resolved": False, "details": {"error": str(exc)[:300]}}
                done += 1
                n_res = sum(1 for v in results.values() if v.get("resolved"))
                n_cached = sum(1 for v in results.values() if v.get("cached"))
                print(f"[swebench-k8s eval] {done}/{len(futs)} graded "
                      f"({n_cached} reused), {n_res} resolved — {iid}",
                      file=sys.stderr, flush=True)
        (out_dir / "eval_k8s_results.json").write_text(json.dumps(results, indent=2))
        return results
