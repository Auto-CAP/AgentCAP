import abc
from typing import Any, Dict, List, Optional, Sequence

import aiohttp


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


class MedAgentBenchToolBackend(ToolBackend):
    """Tool backend for MedAgentBench — FHIR API on a local Docker container."""

    FHIR_TOOLS: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "patient_search",
                "description": (
                    "Patient search across demographics and identifiers in the FHIR server."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "string",
                            "description": "The patient's street address.",
                        },
                        "address-city": {
                            "type": "string",
                            "description": "The city for patient's home address.",
                        },
                        "address-postalcode": {
                            "type": "string",
                            "description": "The postal code for patient's home address.",
                        },
                        "address-state": {
                            "type": "string",
                            "description": "The state for the patient's home address.",
                        },
                        "birthdate": {
                            "type": "string",
                            "description": "The patient's date of birth in the format YYYY-MM-DD.",
                        },
                        "family": {
                            "type": "string",
                            "description": "The patient's family (last) name.",
                        },
                        "gender": {
                            "type": "string",
                            "description": "The patient's legal sex. Starting in the August 2021 version of Epic, the legal-sex parameter is preferred.",
                        },
                        "given": {
                            "type": "string",
                            "description": "The patient's given name. May include first and middle names.",
                        },
                        "identifier": {
                            "type": "string",
                            "description": "The patient's identifier.",
                        },
                        "legal-sex": {
                            "type": "string",
                            "description": "The patient’s legal sex. Takes precedence over the gender search parameter. Available starting in the August 2021 version of Epic.",
                        },
                        "name": {
                            "type": "string",
                            "description": "Any part of the patient's name. When discrete name parameters are used, such as family or given, this parameter is ignored.",
                        },
                        "telecom": {
                            "type": "string",
                            "description": "The patient's phone number or email.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "condition_search",
                "description": "Search patient conditions/problems from the FHIR server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": 'Always "problem-list-item" for this API.',
                        },
                        "patient": {
                            "type": "string",
                            "description": "Reference to a patient resource the condition is for.",
                        },
                    },
                    "required": ["patient"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lab_search",
                "description": "Search lab results from the FHIR Observation resource (uses category=laboratory).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The observation identifier (base name).",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date when the specimen was obtained.",
                        },
                        "patient": {
                            "type": "string",
                            "description": "Reference to a patient resource the condition is for.",
                        },
                        "category": {
                            "type": "string",
                            "description": 'Always "laboratory" for lab searches.',
                        },
                    },
                    "required": ["code", "patient"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vital_search",
                "description": "Search vital signs from the FHIR Observation resource (uses category=vital-signs).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": 'Use "vital-signs" to search for vitals observations.',
                        },
                        "date": {
                            "type": "string",
                            "description": "The date range for when the observation was taken.",
                        },
                        "patient": {
                            "type": "string",
                            "description": "Reference to a patient resource the condition is for.",
                        },
                    },
                    "required": ["category", "patient"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vital_create",
                "description": "Create a vital-sign observation in the FHIR server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "description": 'Use "Observation" for vitals observations.',
                        },
                        "category": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "coding": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "system": {
                                                    "type": "string",
                                                    "description": 'Use "http://hl7.org/fhir/observation-category" ',
                                                },
                                                "code": {
                                                    "type": "string",
                                                    "description": 'Use "vital-signs" ',
                                                },
                                                "display": {
                                                    "type": "string",
                                                    "description": 'Use "Vital Signs" ',
                                                },
                                            },
                                        },
                                    }
                                },
                            },
                        },
                        "code": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The flowsheet ID, encoded flowsheet ID, or LOINC codes to flowsheet mapping. What is being measured.",
                                }
                            },
                        },
                        "effectiveDateTime": {
                            "type": "string",
                            "description": "The date and time the observation was taken, in ISO format.",
                        },
                        "status": {
                            "type": "string",
                            "description": 'The status of the observation. Only a value of "final" is supported. We do not support filing data that isn\'t finalized.',
                        },
                        "valueString": {
                            "type": "string",
                            "description": "Measurement value",
                        },
                        "subject": {
                            "type": "object",
                            "properties": {
                                "reference": {
                                    "type": "string",
                                    "description": "The patient FHIR ID for whom the observation is about.",
                                }
                            },
                        },
                    },
                    "required": [
                        "resourceType",
                        "category",
                        "code",
                        "effectiveDateTime",
                        "status",
                        "valueString",
                        "subject",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "medication_request_search",
                "description": "Search medication requests/orders for a patient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "The category of medication orders to search for. By default all categories are searched.\n\nSupported categories:\nInpatient\nOutpatient (those administered in the clinic - CAMS)\nCommunity (prescriptions)\nDischarge",
                        },
                        "date": {
                            "type": "string",
                            "description": "The medication administration date. This parameter corresponds to the dosageInstruction.timing.repeat.boundsPeriod element. Medication orders that do not have start and end dates within the search parameter dates are filtered. If the environment supports multiple time zones, the search dates are adjusted one day in both directions, so more medications might be returned than expected. Use caution when filtering a medication list by date as it is possible to filter out important active medications. Starting in the November 2022 version of Epic, this parameter is respected. In May 2022 and earlier versions of Epic, this parameter is allowed but is ignored and no date filtering is applied.",
                        },
                        "patient": {
                            "type": "string",
                            "description": "The FHIR patient ID.",
                        },
                    },
                    "required": ["patient"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "medication_request_create",
                "description": "Create a medication request/order in the FHIR server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "description": 'Use "MedicationRequest" for medication requests.',
                        },
                        "medicationCodeableConcept": {
                            "type": "object",
                            "properties": {
                                "coding": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "system": {
                                                "type": "string",
                                                "description": 'Coding system such as "http://hl7.org/fhir/sid/ndc" ',
                                            },
                                            "code": {
                                                "type": "string",
                                                "description": "The actual code",
                                            },
                                            "display": {
                                                "type": "string",
                                                "description": "Display name",
                                            },
                                        },
                                    },
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The order display name of the medication, otherwise the record name.",
                                },
                            },
                        },
                        "authoredOn": {
                            "type": "string",
                            "description": "The date the prescription was written.",
                        },
                        "dosageInstruction": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "route": {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "The medication route.",
                                            }
                                        },
                                    },
                                    "doseAndRate": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "doseQuantity": {
                                                    "type": "object",
                                                    "properties": {
                                                        "value": {"type": "number"},
                                                        "unit": {
                                                            "type": "string",
                                                            "description": 'unit for the dose such as "g" ',
                                                        },
                                                    },
                                                },
                                                "rateQuantity": {
                                                    "type": "object",
                                                    "properties": {
                                                        "value": {"type": "number"},
                                                        "unit": {
                                                            "type": "string",
                                                            "description": 'unit for the rate such as "h" ',
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "status": {
                            "type": "string",
                            "description": 'The status of the medication request. Use "active" ',
                        },
                        "intent": {
                            "type": "string",
                            "description": 'Use "order" ',
                        },
                        "subject": {
                            "type": "object",
                            "properties": {
                                "reference": {
                                    "type": "string",
                                    "description": "The patient FHIR ID for who the medication request is for.",
                                }
                            },
                        },
                    },
                    "required": [
                        "resourceType",
                        "medicationCodeableConcept",
                        "authoredOn",
                        "dosageInstruction",
                        "status",
                        "intent",
                        "subject",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "procedure_search",
                "description": "Search completed procedures for a patient.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "External CPT codes associated with the procedure.",
                        },
                        "date": {
                            "type": "string",
                            "description": "Date or period that the procedure was performed, using the FHIR date parameter format.",
                        },
                        "patient": {
                            "type": "string",
                            "description": "Reference to a patient resource the condition is for.",
                        },
                    },
                    "required": ["date", "patient"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "service_request_create",
                "description": "Create a service request/order in the FHIR server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "description": 'Use "ServiceRequest" for service requests.',
                        },
                        "code": {
                            "type": "object",
                            "description": "The standard terminology codes mapped to the procedure, which can include LOINC, SNOMED, CPT, CBV, THL, or Kuntalitto codes.",
                            "properties": {
                                "coding": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "system": {
                                                "type": "string",
                                                "description": 'Coding system such as "http://loinc.org" ',
                                            },
                                            "code": {
                                                "type": "string",
                                                "description": "The actual code",
                                            },
                                            "display": {
                                                "type": "string",
                                                "description": "Display name",
                                            },
                                        },
                                    },
                                }
                            },
                        },
                        "authoredOn": {
                            "type": "string",
                            "description": "The order instant. This is the date and time of when an order is signed or signed and held.",
                        },
                        "status": {
                            "type": "string",
                            "description": 'The status of the service request. Use "active" ',
                        },
                        "intent": {
                            "type": "string",
                            "description": 'Use "order" ',
                        },
                        "priority": {
                            "type": "string",
                            "description": 'Use "stat" ',
                        },
                        "subject": {
                            "type": "object",
                            "properties": {
                                "reference": {
                                    "type": "string",
                                    "description": "The patient FHIR ID for who the service request is for.",
                                }
                            },
                        },
                        "note": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Free text comment here",
                                }
                            },
                        },
                        "occurrenceDateTime": {
                            "type": "string",
                            "description": "The date and time for the service request to be conducted, in ISO format.",
                        },
                    },
                    "required": [
                        "resourceType",
                        "code",
                        "authoredOn",
                        "status",
                        "intent",
                        "priority",
                        "subject",
                    ],
                },
            },
        },
    ]

    def __init__(
        self,
        session: aiohttp.ClientSession,
        fhir_base_url: str = "http://localhost:8080/fhir",
    ):
        self._session = session
        self._fhir_base = fhir_base_url.rstrip("/")

    async def setup(self, task_config: Dict[str, Any]) -> bool:
        del task_config
        try:
            async with self._session.get(f"{self._fhir_base}/metadata") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        return self.FHIR_TOOLS

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        endpoint, method = self._resolve_endpoint(name)
        request_args = dict(arguments or {})
        if name == "lab_search":
            request_args["category"] = "laboratory"
        elif name == "vital_search":
            request_args["category"] = "vital-signs"

        if method == "GET":
            async with self._session.get(endpoint, params=request_args) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    return {
                        "status": "error",
                        "status_code": resp.status,
                        "detail": body[:500],
                    }
                data = await resp.json(content_type=None)
                return self._format_fhir_response(data)
        if method == "POST":
            payload = request_args.get("payload", request_args)
            async with self._session.post(endpoint, json=payload) as resp:
                if resp.status in (200, 201):
                    return {"status": "success", "status_code": resp.status}
                body = await resp.text()
                return {
                    "status": "error",
                    "status_code": resp.status,
                    "detail": body[:500],
                }
        raise RuntimeError(f"Unsupported HTTP method for {name}: {method}")

    async def teardown(self) -> None:
        pass

    def _resolve_endpoint(self, name: str) -> tuple[str, str]:
        """Map tool name to FHIR endpoint + HTTP method."""
        mapping = {
            "patient_search": (f"{self._fhir_base}/Patient", "GET"),
            "condition_search": (f"{self._fhir_base}/Condition", "GET"),
            "lab_search": (f"{self._fhir_base}/Observation", "GET"),
            "vital_search": (f"{self._fhir_base}/Observation", "GET"),
            "vital_create": (f"{self._fhir_base}/Observation", "POST"),
            "medication_request_search": (
                f"{self._fhir_base}/MedicationRequest",
                "GET",
            ),
            "medication_request_create": (
                f"{self._fhir_base}/MedicationRequest",
                "POST",
            ),
            "procedure_search": (f"{self._fhir_base}/Procedure", "GET"),
            "service_request_create": (f"{self._fhir_base}/ServiceRequest", "POST"),
        }
        if name not in mapping:
            raise ValueError(f"Unknown FHIR tool: {name}")
        return mapping[name]

    @staticmethod
    def _format_fhir_response(data: Any) -> str:
        """Format FHIR Bundle response for LLM consumption."""
        import json

        if isinstance(data, dict) and data.get("resourceType") == "Bundle":
            entries = data.get("entry", [])
            total = data.get("total", len(entries))
            results = []
            for entry in entries[:10]:
                if isinstance(entry, dict):
                    resource = entry.get("resource", {})
                    results.append(resource)
            return json.dumps({"total": total, "results": results}, indent=2, default=str)
        return json.dumps(data, indent=2, default=str)
