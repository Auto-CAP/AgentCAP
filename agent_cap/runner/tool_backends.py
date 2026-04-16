import abc
import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse, urlunparse

import aiohttp

if TYPE_CHECKING:
    from agent_cap.runner.fhir_mock_server import FHIRMockServer


class ToolBackend(abc.ABC):
    @abc.abstractmethod
    async def setup(self, task_config: Dict[str, Any]) -> bool: ...

    @abc.abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]: ...

    @abc.abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any: ...

    @abc.abstractmethod
    async def teardown(self) -> None: ...

    async def get_patch(self) -> str:
        return ""

    def run_tests(self, timeout: int = 300) -> Dict[str, Any]:
        del timeout
        return {"passed": False, "reason": "run_tests not supported"}


class MCPToolBackend(ToolBackend):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        mcp_server_url: str,
        enabled_tools: Sequence[str] = (),
    ):
        self._session = session
        self._mcp_url = mcp_server_url
        self._enabled_tools = enabled_tools
        self._tools: List[Dict[str, Any]] = []

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        return True

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self._tools:
            return self._tools

        async with self._session.post(
            f"{self._mcp_url.rstrip('/')}/list-tools"
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"list-tools failed ({resp.status}): {body}")
            payload = await resp.json()

        from agent_cap.runner.llm_client import _fix_tool_schema

        enabled = set(self._enabled_tools)
        for tool in payload:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name", ""))
            if not name:
                continue
            if enabled and name not in enabled:
                continue
            self._tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description", "")),
                        "parameters": _fix_tool_schema(
                            tool.get("input_schema", {}), name
                        ),
                    },
                }
            )
        return self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        async with self._session.post(
            f"{self._mcp_url.rstrip('/')}/call-tool",
            json={"tool_name": name, "tool_args": arguments},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"tool call failed ({resp.status}): {body}")
            return await resp.json()

    async def teardown(self) -> None:
        pass


class SWEBenchToolBackend(ToolBackend):
    def __init__(self, runtime: str = "docker"):
        self._runtime = runtime
        self._backend: Optional[Any] = None
        self._tools: List[Dict[str, Any]] = []

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        from agent_cap.backends.swebench_backend import SWEBenchBackend

        self._backend = SWEBenchBackend(runtime=self._runtime)
        return self._backend.setup(task_config)

    async def list_tools(self) -> List[Dict[str, Any]]:
        if self._tools:
            return self._tools
        from agent_cap.backends.tool_executor import TOOL_DEFINITIONS

        self._tools = list(TOOL_DEFINITIONS)
        return self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if not self._backend:
            raise RuntimeError("Backend not set up")
        result = self._backend.execute(name, "call", arguments)
        if result.success:
            return [{"type": "text", "text": result.output}]
        raise RuntimeError(result.output)

    async def teardown(self) -> None:
        if self._backend:
            self._backend.teardown()
            self._backend.cleanup()
            self._backend = None

    async def get_patch(self) -> str:
        if self._backend:
            return self._backend.get_patch()
        return ""

    def run_tests(self, timeout: int = 300) -> Dict[str, Any]:
        if self._backend:
            return self._backend.run_tests(timeout=timeout)
        return {"passed": False, "reason": "no backend"}


class MedAgentBenchToolBackend(ToolBackend):
    """Text-protocol backend for MedAgentBench (GET/POST/FINISH)."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        fhir_base_url: Optional[str] = None,
        data_dir: Optional[str] = None,
    ):
        self._session = session
        self._external_url = fhir_base_url.rstrip("/") if fhir_base_url else None
        self._data_dir = data_dir
        self._fhir_base = ""
        self.max_round = 5
        self._mock_server: Optional["FHIRMockServer"] = None

    @property
    def prompt_api_base(self) -> str:
        return self._fhir_base

    @property
    def fhir_api_base(self) -> str:
        return self._fhir_base.rstrip("/") + "/"

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        del task_config

        if self._fhir_base:
            # Server already running, reuse it
            try:
                async with self._session.get(f"{self._fhir_base}/metadata") as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass

        if self._external_url:
            self._fhir_base = self._external_url
        else:
            from agent_cap.runner.fhir_mock_server import FHIRMockServer

            self._mock_server = FHIRMockServer(data_dir=self._data_dir)
            self._fhir_base = await self._mock_server.start()

        try:
            async with self._session.get(f"{self._fhir_base}/metadata") as resp:
                return resp.status == 200
        except Exception:
            if self._mock_server is not None:
                await self._mock_server.stop()
                self._mock_server = None
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        # Official MedAgentBench is text protocol, not function calling.
        return []

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        del name, arguments
        raise RuntimeError(
            "MedAgentBench backend does not expose function-calling tools; "
            "use parse_and_execute() with text actions."
        )

    async def parse_and_execute(self, model_output: str) -> tuple[str, bool]:
        """Parse model text output, execute FHIR request, return (response, is_finished)."""
        text = (
            (model_output or "")
            .strip()
            .replace("```tool_code", "")
            .replace("```", "")
            .strip()
        )

        if text.startswith("GET"):
            try:
                url = text[3:].strip()
                if "_format=json" not in url:
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}_format=json"
                url = self._normalize_request_url(url)
                async with self._session.get(url) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        return (
                            f"Error in sending the GET request: HTTP {resp.status} {body}",
                            False,
                        )

                    try:
                        parsed = json.loads(body)
                        payload_text = json.dumps(parsed, ensure_ascii=False)
                    except Exception:
                        payload_text = body
                return (
                    "Here is the response from the GET request:\n"
                    f"{payload_text}. Please call FINISH if you have got answers "
                    "for all the questions and finished all the requested tasks",
                    False,
                )
            except Exception as exc:
                return f"Error in sending the GET request: {exc}", False

        if text.startswith("POST"):
            lines = text.splitlines()
            if len(lines) < 2:
                return "Invalid POST request", False
            try:
                url = self._normalize_request_url(lines[0][4:].strip())
                payload = json.loads("\n".join(lines[1:]))
            except Exception:
                return "Invalid POST request", False

            try:
                async with self._session.post(url, json=payload) as resp:
                    if resp.status in (200, 201):
                        return (
                            "POST request accepted and executed successfully. "
                            "Please call FINISH if you have got answers for all "
                            "the questions and finished all the requested tasks",
                            False,
                        )
                    body = await resp.text()
                    return (
                        f"Error in sending the POST request: HTTP {resp.status} {body}",
                        False,
                    )
            except Exception as exc:
                return f"Error in sending the POST request: {exc}", False

        if text.startswith("FINISH("):
            if text.endswith(")"):
                return text[len("FINISH(") : -1], True
            return text[len("FINISH(") :], True

        return "Invalid action", True

    async def teardown(self) -> None:
        if self._mock_server is not None:
            await self._mock_server.stop()
            self._mock_server = None

    def _normalize_request_url(self, url: str) -> str:
        normalized = (url or "").strip()
        if not normalized:
            return normalized

        if "{api_base}" in normalized:
            normalized = normalized.replace("{api_base}", self._fhir_base)

        if normalized.startswith("/"):
            if normalized.startswith("/fhir/"):
                normalized = f"{self._fhir_base}{normalized[5:]}"
            else:
                normalized = f"{self._fhir_base}/{normalized.lstrip('/')}"

        if normalized.startswith(("http://", "https://")):
            source = urlparse(normalized)
            target = urlparse(self._fhir_base)
            if source.netloc != target.netloc and "/fhir" in source.path:
                idx = source.path.find("/fhir")
                tail = source.path[idx + len("/fhir") :]
                remapped_path = f"{target.path.rstrip('/')}{tail}"
                normalized = urlunparse(
                    (
                        target.scheme,
                        target.netloc,
                        remapped_path,
                        source.params,
                        source.query,
                        source.fragment,
                    )
                )
            return normalized

        return f"{self._fhir_base}/{normalized.lstrip('/')}"


class FinanceBenchToolBackend(ToolBackend):
    """Tool backend for FinanceBench: search pre-extracted SEC filing evidence + safe calculator."""

    DEFAULT_DATA_PATH = (
        Path(__file__).resolve().parent.parent.parent
        / "third_party"
        / "financebench"
        / "data"
        / "financebench_open_source.jsonl"
    )

    def __init__(self, data_path: Optional[Path] = None):
        self._data_path = data_path or self.DEFAULT_DATA_PATH
        # doc_name -> list of evidence dicts {evidence_text, evidence_page_num, evidence_text_full_page}
        self._doc_index: Dict[str, List[Dict[str, Any]]] = {}
        self._task_config: Dict[str, Any] = {}

    def _load_index(self) -> None:
        if self._doc_index:
            return
        with self._data_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for ev in item.get("evidence", []):
                    doc = ev.get("doc_name", "")
                    if doc:
                        self._doc_index.setdefault(doc, []).append(ev)

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        self._load_index()
        self._task_config = task_config
        return True

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_document",
                    "description": (
                        "Search the pre-extracted text of an SEC filing for information "
                        "relevant to the query. Returns matching evidence passages."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "doc_name": {
                                "type": "string",
                                "description": (
                                    "The document identifier, e.g. '3M_2018_10K'. "
                                    "Format: {COMPANY}_{YEAR}_{TYPE}."
                                ),
                            },
                            "query": {
                                "type": "string",
                                "description": "Keywords or phrase to search for in the document.",
                            },
                        },
                        "required": ["doc_name", "query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": (
                        "Evaluate a mathematical expression and return the numeric result. "
                        "Supports basic arithmetic, percentages, and common financial formulas. "
                        "Example: '(1577 / 32136) * 100' returns 4.906..."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "A Python-style arithmetic expression to evaluate.",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "search_document":
            return self._search_document(
                str(arguments.get("doc_name", "")),
                str(arguments.get("query", "")),
            )
        if name == "calculate":
            return self._calculate(str(arguments.get("expression", "")))
        raise ValueError(f"Unknown tool: {name}")

    def _search_document(self, doc_name: str, query: str) -> str:
        entries = self._doc_index.get(doc_name, [])
        if not entries:
            available = sorted(self._doc_index.keys())[:10]
            return (
                f"Document '{doc_name}' not found. "
                f"Available documents (sample): {available}"
            )
        query_lower = query.lower()
        query_terms = query_lower.split()
        scored: List[tuple] = []
        for ev in entries:
            full = ev.get("evidence_text_full_page", "")
            snippet = ev.get("evidence_text", "")
            combined = (full + " " + snippet).lower()
            score = sum(1 for t in query_terms if t in combined)
            scored.append((score, ev.get("evidence_page_num", 0), snippet, full))

        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:3]
        parts = []
        for score, page, snippet, full in top:
            parts.append(
                f"[Page {page}]\n{full[:3000] if full else snippet[:1000]}"
            )
        return "\n\n---\n\n".join(parts) if parts else "No relevant passages found."

    @staticmethod
    def _calculate(expression: str) -> str:
        safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_names["abs"] = abs
        safe_names["round"] = round
        try:
            result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
            return str(result)
        except Exception as exc:
            return f"Error evaluating '{expression}': {exc}"

    async def teardown(self) -> None:
        pass
