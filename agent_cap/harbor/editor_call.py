#!/usr/bin/env python3
"""Run SWE-agent's ``str_replace_editor`` with arguments read from a JSON file.

Uploaded next to ``agent_cap/backends/swe_agent_tools/str_replace_editor`` (and
its ``registry.py`` shim) into the task container by CAPAgent's ``shell`` tool
backend. Reading the arguments from a file avoids shell quoting and the
per-argument length limit of exec(2) for large ``file_text`` values; the
official script then runs in-process with an equivalent ``sys.argv``.

Usage: python3 editor_call.py ARGS.json   (Python 3.6+, standard library only)
"""

import json
import os
import runpy
import sys


def build_argv(script, args):
    argv = [script, str(args.get("command", "")), str(args.get("path", ""))]
    for key in ("file_text", "old_str", "new_str"):
        if args.get(key) is not None:
            argv += ["--" + key, str(args[key])]
    if args.get("insert_line") is not None:
        argv += ["--insert_line", str(int(args["insert_line"]))]
    view_range = args.get("view_range")
    if isinstance(view_range, (list, tuple)) and len(view_range) == 2:
        argv += ["--view_range", str(int(view_range[0])), str(int(view_range[1]))]
    return argv


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        args = json.load(f)
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "str_replace_editor")
    sys.path.insert(0, here)
    sys.argv = build_argv(script, args)
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
