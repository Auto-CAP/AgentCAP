"""The k8s evaluator must not re-grade a patch it has already graded.

finalize() grades everything in its buffer, and a --resume pass re-feeds every
finished row back into the evaluator. Re-grading an unchanged patch costs a pod
and a full test suite per instance per attempt, and for instances whose tests
call a public service it also makes the verdict flap: the same patch grades
resolved on one attempt and unresolved on the next.
"""

import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

_UNSET = object()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_cap.agents.evaluators_swebench import SWEBenchK8sEvaluator  # noqa: E402


def _seed(out_dir: Path, iid: str, patch_text: str, resolved, raw_report=None,
          graded_patch=_UNSET):
    """Write what a *completed* grading of `patch_text` leaves behind.

    `graded_patch` overrides the post-report marker independently of
    patch.diff, so a test can reproduce a directory where grading started on
    one patch and the report belongs to another. `raw_report` writes the report
    file verbatim, for the malformed-cache case.
    """
    d = out_dir / "eval_k8s" / iid
    d.mkdir(parents=True)
    (d / "patch.diff").write_text(patch_text)
    (d / "report.json").write_text(
        raw_report if raw_report is not None
        else json.dumps({iid: {"resolved": resolved}}))
    marker = patch_text if graded_patch is _UNSET else graded_patch
    if marker is not None:
        (d / "graded_patch.diff").write_text(marker)


class EvalCacheTests(unittest.TestCase):
    def _run(self, buffer, seeds, env=None):
        """finalize() with the pod/dataset machinery stubbed out.

        Records which instances actually reached the grading path, which is the
        only thing these tests care about.
        """
        graded = []
        ev = SWEBenchK8sEvaluator.__new__(SWEBenchK8sEvaluator)
        ev._buffer = dict(buffer)
        ev.dataset = "ds"
        ev.model_name = "m"
        ev.max_workers = 2

        with TemporaryDirectory() as td:
            out = Path(td)
            for iid, seed in seeds.items():
                _seed(out, iid, *seed)

            ds = [{"instance_id": i} for i in buffer]
            spec = types.SimpleNamespace(instance_image_key="img", eval_script="true")

            def fake_acquire(image, iid):
                graded.append(iid)
                raise RuntimeError("stubbed: grading path reached")

            provider = types.SimpleNamespace(
                acquire_exec=fake_acquire, release_exec=lambda b: None)

            with patch.dict(sys.modules, {
                "datasets": types.SimpleNamespace(load_dataset=lambda *a, **k: ds),
                "swebench.harness.grading": types.SimpleNamespace(
                    get_eval_report=lambda **k: {}),
                "swebench.harness.test_spec.test_spec": types.SimpleNamespace(
                    make_test_spec=lambda inst, namespace=None: spec),
            }), patch("agent_cap.agents.sandbox_providers.get_exec_provider",
                      return_value=provider), \
                 patch.dict(os.environ, env or {}, clear=False):
                results = ev.finalize(out)
        return graded, results

    def test_unchanged_patch_is_not_regraded(self):
        graded, results = self._run(
            buffer={"a": "PATCH-A"}, seeds={"a": ("PATCH-A", True)})
        self.assertEqual(graded, [], "an unchanged patch must not be re-graded")
        self.assertTrue(results["a"]["resolved"])
        self.assertTrue(results["a"]["cached"])

    def test_changed_patch_is_regraded(self):
        """A retry that produces a different patch invalidates the cache -- the
        old verdict says nothing about the new patch."""
        graded, _ = self._run(
            buffer={"a": "PATCH-B"}, seeds={"a": ("PATCH-A", True)})
        self.assertEqual(graded, ["a"])

    def test_uncached_instance_is_graded(self):
        graded, _ = self._run(buffer={"a": "PATCH-A"}, seeds={})
        self.assertEqual(graded, ["a"])

    def test_no_cache_env_forces_regrade(self):
        graded, _ = self._run(
            buffer={"a": "PATCH-A"}, seeds={"a": ("PATCH-A", True)},
            env={"AGENTCAP_EVAL_NO_CACHE": "1"})
        self.assertEqual(graded, ["a"])

    def test_report_from_a_different_patch_is_not_served(self):
        """The sequence the marker exists to stop.

        Patch A grades and leaves a report. A retry with patch B overwrites
        patch.diff, then fails to apply and returns before writing a report --
        so report.json still describes A. Keying the cache on patch.diff would
        match B against B and hand back A's verdict.
        """
        graded, _ = self._run(
            buffer={"a": "PATCH-B"},
            seeds={"a": ("PATCH-B", True, None, "PATCH-A")})
        self.assertEqual(graded, ["a"], "a report for a different patch is not evidence")

    def test_report_without_the_marker_is_graded(self):
        """Grading that was interrupted after the report but before the marker
        -- or a directory written by a version that had neither -- must not be
        trusted."""
        graded, _ = self._run(
            buffer={"a": "PATCH-A"},
            seeds={"a": ("PATCH-A", True, None, None)})
        self.assertEqual(graded, ["a"])

    def test_malformed_report_is_graded_rather_than_trusted(self):
        """A cache we cannot read is not evidence of anything. Trusting it
        would silently invent a verdict for a patch nobody graded."""
        graded, _ = self._run(
            buffer={"a": "PATCH-A"}, seeds={"a": ("PATCH-A", None, "{not json")})
        self.assertEqual(graded, ["a"])

    def test_report_without_a_boolean_resolved_is_graded(self):
        graded, _ = self._run(
            buffer={"a": "PATCH-A"},
            seeds={"a": ("PATCH-A", None, json.dumps({"a": {"resolved": "yes"}}))})
        self.assertEqual(graded, ["a"])


if __name__ == "__main__":
    unittest.main()
