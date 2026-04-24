"""Minimal registry shim for SWE-agent str_replace_editor.

The official script does `from registry import registry as REGISTRY`
and uses REGISTRY.get(key, default) / REGISTRY[key] = value.
We implement it as a simple dict with env-var overrides.
"""

import os

class _Registry(dict):
    """Dict that falls back to env vars."""
    def get(self, key, default=None):
        val = super().get(key, None)
        if val is not None:
            return val
        return os.environ.get(key, default)

registry = _Registry({
    "USE_FILEMAP": "true",
    "USE_LINTER": "false",
    "MAX_WINDOW_EXPANSION_VIEW": "0",
    "MAX_WINDOW_EXPANSION_EDIT_CONFIRM": "0",
})
