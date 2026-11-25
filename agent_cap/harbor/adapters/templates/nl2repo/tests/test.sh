#!/bin/bash
# Harbor verifier for one NL2Repo task. Always exits 0; the score is in
# /logs/verifier/reward.json (written by nl2repo_eval.py).
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/nl2repo_eval.py
status=$?
if [ ! -s /logs/verifier/reward.json ]; then
    echo "nl2repo_eval.py exited with status ${status} and wrote no reward; scoring 0" >&2
    echo '{"reward": 0.0, "passed": 0, "pass_all": 0.0}' > /logs/verifier/reward.json
fi
exit 0
