"""Playwright-based browser tools for WebArena agent.

Tools: goto, click, type, scroll, screenshot, get_page_text
"""

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "goto",
            "description": "Navigate to a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click on an element by CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into an input field identified by CSS selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the page. direction: 'up' or 'down'.",
            "parameters": {
                "type": "object",
                "properties": {"direction": {"type": "string", "enum": ["up", "down"]}},
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_text",
            "description": "Get the visible text content of the current page.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_url",
            "description": "Get the current page URL.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class BrowserToolResult:
    tool_name: str
    tool_call_id: str
    output: str
    latency_ms: float
    success: bool
    screenshot_b64: str = ""


class BrowserToolExecutor:
    def __init__(self, max_output_chars: int = 16_000):
        self.max_output_chars = max_output_chars
        self._page = None
        self._browser = None
        self._playwright = None

    def start(self):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page(viewport={"width": 1280, "height": 720})

    def stop(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None

    def execute(
        self, tool_name: str, tool_call_id: str, arguments: Dict[str, Any]
    ) -> BrowserToolResult:
        t0 = time.perf_counter()
        try:
            output = self._dispatch(tool_name, arguments)
            success = True
        except Exception as exc:
            output = f"ERROR: {type(exc).__name__}: {exc}"
            success = False
        latency_ms = (time.perf_counter() - t0) * 1000

        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + "\n... (truncated)"

        return BrowserToolResult(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            output=output,
            latency_ms=latency_ms,
            success=success,
        )

    def _dispatch(self, name: str, args: Dict[str, Any]) -> str:
        if name == "goto":
            self._page.goto(args["url"], wait_until="domcontentloaded", timeout=15000)
            return f"Navigated to {self._page.url}"
        if name == "click":
            self._page.click(args["selector"], timeout=5000)
            self._page.wait_for_load_state("domcontentloaded", timeout=5000)
            return f"Clicked {args['selector']}, now at {self._page.url}"
        if name == "type_text":
            self._page.fill(args["selector"], args["text"], timeout=5000)
            return f"Typed into {args['selector']}"
        if name == "scroll":
            delta = -300 if args["direction"] == "up" else 300
            self._page.mouse.wheel(0, delta)
            self._page.wait_for_timeout(500)
            return f"Scrolled {args['direction']}"
        if name == "get_page_text":
            text = self._page.inner_text("body")
            return text
        if name == "get_page_url":
            return self._page.url
        raise ValueError(f"Unknown tool: {name}")

    def get_page_snapshot(self) -> str:
        if not self._page:
            return ""
        try:
            return self._page.inner_text("body")[:8000]
        except Exception:
            return ""
