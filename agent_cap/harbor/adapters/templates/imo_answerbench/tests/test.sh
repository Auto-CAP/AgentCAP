#!/bin/bash
# Harbor verifier for one IMO-AnswerBench task. Always exits 0; the reward is in
# /logs/verifier/reward.json (written by grade_imo.py).
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/grade_imo.py
status=$?
if [ ! -s /logs/verifier/reward.json ]; then
    echo "grader exited with status ${status} and wrote no reward; scoring 0" >&2
    echo '{"reward": 0.0}' > /logs/verifier/reward.json
fi
exit 0
