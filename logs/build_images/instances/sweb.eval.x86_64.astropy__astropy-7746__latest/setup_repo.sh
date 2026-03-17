#!/bin/bash
set -euxo pipefail
git clone -o origin  --single-branch https://github.com/astropy/astropy /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard d5bd3f68bb6d5ce3a61bdce9883ee750d1afade5
git remote remove origin
TARGET_TIMESTAMP=$(git show -s --format=%ci d5bd3f68bb6d5ce3a61bdce9883ee750d1afade5)
git tag -l | while read tag; do TAG_COMMIT=$(git rev-list -n 1 "$tag"); TAG_TIME=$(git show -s --format=%ci "$TAG_COMMIT"); if [[ "$TAG_TIME" > "$TARGET_TIMESTAMP" ]]; then git tag -d "$tag"; fi; done
git reflog expire --expire=now --all
git gc --prune=now --aggressive
AFTER_TIMESTAMP=$(date -d "$TARGET_TIMESTAMP + 1 second" '+%Y-%m-%d %H:%M:%S')
COMMIT_COUNT=$(git log --oneline --all --since="$AFTER_TIMESTAMP" | wc -l)
[ "$COMMIT_COUNT" -eq 0 ] || exit 1
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
python -m pip install -e .[test] --verbose
git config --global user.email setup@swebench.config
git config --global user.name SWE-bench
git commit --allow-empty -am SWE-bench
