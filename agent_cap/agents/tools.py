"""Tool provider abstraction and a minimal local tool registry.

A `ToolProvider` exposes:
- list_tools() -> OpenAI-style tool schemas
- call(name, args) -> string result

Strategies pass these to the LLM unchanged. The CLI ships with a tiny in-process
provider that can be extended by registering Python callables. For real tool
backends (MCP, SWE-bench, etc.) wrap them with a thin adapter.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import operator
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

_SAFE_CALC_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_SAFE_CALC_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_SAFE_CALC_FUNCS = {k: getattr(math, k) for k in ("sqrt", "log", "exp", "sin", "cos")}
_SAFE_CALC_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_calc_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_calc_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_CALC_CONSTS:
        return _SAFE_CALC_CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_CALC_BINOPS:
        return _SAFE_CALC_BINOPS[type(node.op)](_safe_calc_eval(node.left), _safe_calc_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_CALC_UNARYOPS:
        return _SAFE_CALC_UNARYOPS[type(node.op)](_safe_calc_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SAFE_CALC_FUNCS
        and not node.keywords
    ):
        return _SAFE_CALC_FUNCS[node.func.id](*(_safe_calc_eval(a) for a in node.args))
    raise ValueError(f"calc: disallowed expression node {type(node).__name__}")


class ToolProvider(Protocol):
    async def list_tools(self) -> List[Dict[str, Any]]: ...
    async def call(self, name: str, arguments: Dict[str, Any]) -> str: ...


ToolFn = Callable[..., Any]


class LocalToolRegistry:
    """In-process tool provider. Register sync or async callables."""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._fns: Dict[str, ToolFn] = {}

    def register(
        self,
        name: str,
        fn: ToolFn,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not name:
            raise ValueError("tool name must be non-empty")
        self._tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or (fn.__doc__ or "").strip(),
                "parameters": parameters or {"type": "object", "properties": {}},
            },
        }
        self._fns[name] = fn

    def tool(self, name: Optional[str] = None, description: str = "",
             parameters: Optional[Dict[str, Any]] = None) -> Callable[[ToolFn], ToolFn]:
        def deco(fn: ToolFn) -> ToolFn:
            self.register(name or fn.__name__, fn, description=description, parameters=parameters)
            return fn
        return deco

    async def list_tools(self) -> List[Dict[str, Any]]:
        return list(self._tools.values())

    async def call(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self._fns:
            return f"ERROR: unknown tool '{name}'"
        fn = self._fns[name]
        try:
            result: Any
            if inspect.iscoroutinefunction(fn):
                result = await fn(**arguments)
            else:
                result = fn(**arguments)
        except Exception as exc:
            return f"ERROR: {exc}"
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)


def build_demo_tools() -> LocalToolRegistry:
    """Tiny demo toolkit: calculator + echo. Used by --mock and as an example."""
    reg = LocalToolRegistry()

    @reg.tool(
        name="calc",
        description="Evaluate a simple arithmetic expression and return the number.",
        parameters={
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    )
    def calc(expr: str) -> str:
        return str(_safe_calc_eval(ast.parse(expr, mode="eval")))

    @reg.tool(
        name="echo",
        description="Return the message verbatim.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )
    def echo(message: str) -> str:
        return message

    return reg


__all__ = ["ToolProvider", "LocalToolRegistry", "build_demo_tools"]
