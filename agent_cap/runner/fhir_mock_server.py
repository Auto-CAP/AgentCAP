"""Lightweight FHIR mock server for MedAgentBench.

Loads pre-dumped FHIR data into memory and serves the resource types
needed by MedAgentBench tools. No Docker or Java dependency required.
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aiohttp import web


class FHIRMockServer:
    """In-memory FHIR server backed by MedAgentBench JSON dumps."""

    DEFAULT_DATA_DIR = (
        Path(__file__).resolve().parent.parent.parent
        / "third_party"
        / "MedAgentBench"
        / "data"
        / "fhir_dump"
    )

    RESOURCE_TYPES = [
        "Patient",
        "Condition",
        "Observation",
        "MedicationRequest",
        "Procedure",
        "ServiceRequest",
    ]

    def __init__(
        self,
        data_dir: Optional[Union[Path, str]] = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        """Initialize a local FHIR mock server.

        Args:
            data_dir: Directory containing {ResourceType}.json dumps.
            host: Host to bind the server to.
            port: Port to bind to (0 means auto-assign a free port).
        """
        self._data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        self._host = host
        self._port = port
        self._db: Dict[str, List[Dict[str, Any]]] = {}
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self.base_url: str = ""

    def _load_data(self) -> None:
        """Load all supported FHIR dump files into memory."""
        db: Dict[str, List[Dict[str, Any]]] = {}
        for resource_type in self.RESOURCE_TYPES:
            path = self._data_dir / f"{resource_type}.json"
            if not path.exists():
                db[resource_type] = []
                continue
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            db[resource_type] = payload if isinstance(payload, list) else []
        self._db = db

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/fhir/metadata", self._handle_metadata)
        app.router.add_get("/fhir/{rtype}", self._handle_search)
        app.router.add_post("/fhir/{rtype}", self._handle_create)
        return app

    async def _handle_metadata(self, request: web.Request) -> web.Response:
        del request
        return web.json_response(
            {
                "resourceType": "CapabilityStatement",
                "fhirVersion": "4.0.1",
                "status": "active",
            }
        )

    async def _handle_search(self, request: web.Request) -> web.Response:
        """Handle GET /fhir/{ResourceType}?param=value."""
        resource_type = str(request.match_info.get("rtype", ""))
        if resource_type not in self.RESOURCE_TYPES:
            return web.json_response({"error": "Unknown resource type"}, status=400)

        resources = self._db.get(resource_type, [])
        params = dict(request.query)
        filtered = self._filter_resources(resources, params, resource_type)

        bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": len(filtered),
            "entry": [{"resource": r} for r in filtered[:500]],
        }
        return web.json_response(bundle)

    @staticmethod
    def _extract_date_value(resource: Dict[str, Any]) -> str:
        """Pick the best-effort searchable date/time string from a resource."""
        candidate_keys = [
            "effectiveDateTime",
            "authoredOn",
            "performedDateTime",
            "occurrenceDateTime",
            "recordedDate",
            "onsetDateTime",
            "issued",
        ]
        for key in candidate_keys:
            value = resource.get(key)
            if isinstance(value, str) and value:
                return value
        performed = resource.get("performedPeriod")
        if isinstance(performed, dict):
            start = performed.get("start")
            if isinstance(start, str) and start:
                return start
        return ""

    @staticmethod
    def _normalize_patient_id(value: str) -> str:
        if "/" in value:
            return value.split("/")[-1]
        return value

    def _filter_resources(
        self,
        resources: List[Dict[str, Any]],
        params: Dict[str, str],
        resource_type: str,
    ) -> List[Dict[str, Any]]:
        """Filter resources by common FHIR search parameters used in MedAgentBench."""
        results = resources

        patient_id = params.get("patient") or params.get("subject")
        if patient_id:
            normalized_patient_id = self._normalize_patient_id(patient_id)
            if resource_type == "Patient":
                results = [
                    r for r in results if str(r.get("id", "")) == normalized_patient_id
                ]
            else:
                results = [
                    r
                    for r in results
                    if normalized_patient_id
                    in str((r.get("subject") or {}).get("reference", ""))
                    or normalized_patient_id
                    in str((r.get("patient") or {}).get("reference", ""))
                ]

        category = params.get("category")
        if category:
            category_lower = category.lower()
            results = [
                r
                for r in results
                if category_lower in json.dumps(r.get("category", [])).lower()
            ]

        code = params.get("code")
        if code:
            code_lower = code.lower()
            results = [
                r
                for r in results
                if code_lower in json.dumps(r.get("code", {})).lower()
                or code_lower
                in json.dumps(r.get("medicationCodeableConcept", {})).lower()
            ]

        if resource_type == "Patient":
            name = params.get("name")
            if name:
                name_lower = name.lower()
                results = [
                    r
                    for r in results
                    if name_lower in json.dumps(r.get("name", [])).lower()
                ]

            given = params.get("given")
            if given:
                given_lower = given.lower()
                results = [
                    r
                    for r in results
                    if given_lower in json.dumps(r.get("name", [])).lower()
                ]

            family = params.get("family")
            if family:
                family_lower = family.lower()
                results = [
                    r
                    for r in results
                    if family_lower in json.dumps(r.get("name", [])).lower()
                ]

            birthdate = params.get("birthdate")
            if birthdate:
                results = [r for r in results if r.get("birthDate", "") == birthdate]

            identifier = params.get("identifier")
            if identifier:
                results = [
                    r
                    for r in results
                    if identifier == str(r.get("id", ""))
                    or identifier in json.dumps(r.get("identifier", []))
                ]

        date_param = params.get("date")
        if date_param:
            clauses = [c.strip() for c in date_param.split(",") if c.strip()]
            for clause in clauses:
                op = "eq"
                target = clause
                if len(clause) >= 3 and clause[:2] in {"ge", "gt", "le", "lt", "eq"}:
                    op = clause[:2]
                    target = clause[2:]

                if target:
                    if op == "ge":
                        results = [
                            r
                            for r in results
                            if self._extract_date_value(r)
                            and self._extract_date_value(r) >= target
                        ]
                    elif op == "gt":
                        results = [
                            r
                            for r in results
                            if self._extract_date_value(r)
                            and self._extract_date_value(r) > target
                        ]
                    elif op == "le":
                        results = [
                            r
                            for r in results
                            if self._extract_date_value(r)
                            and self._extract_date_value(r) <= target
                        ]
                    elif op == "lt":
                        results = [
                            r
                            for r in results
                            if self._extract_date_value(r)
                            and self._extract_date_value(r) < target
                        ]
                    else:
                        results = [
                            r
                            for r in results
                            if target in self._extract_date_value(r)
                        ]

        sort = params.get("_sort")
        if sort == "-date":
            results = sorted(results, key=self._extract_date_value, reverse=True)

        count = params.get("_count")
        if count:
            try:
                count_value = int(count)
                if count_value >= 0:
                    results = results[:count_value]
            except ValueError:
                pass

        return results

    async def _handle_create(self, request: web.Request) -> web.Response:
        """Handle POST /fhir/{ResourceType}."""
        resource_type = str(request.match_info.get("rtype", ""))
        if resource_type not in self.RESOURCE_TYPES:
            return web.json_response({"error": "Unknown resource type"}, status=400)

        try:
            payload = await request.json()
        except Exception:
            body = await request.text()
            try:
                payload = json.loads(body)
            except Exception:
                return web.json_response({"error": "Invalid JSON"}, status=400)

        if not isinstance(payload, dict):
            return web.json_response({"error": "Payload must be an object"}, status=400)

        payload["id"] = payload.get("id") or str(uuid.uuid4())
        payload["resourceType"] = resource_type
        self._db.setdefault(resource_type, []).append(payload)

        return web.json_response(payload, status=201)

    async def start(self) -> str:
        """Start the server and return the base FHIR URL."""
        if self._runner is not None and self.base_url:
            return self.base_url

        self._load_data()
        self._app = self._build_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        sockets = getattr(getattr(self._site, "_server", None), "sockets", None)
        if not sockets:
            raise RuntimeError("FHIR mock server failed to bind a TCP socket")
        actual_port = sockets[0].getsockname()[1]
        self.base_url = f"http://{self._host}:{actual_port}/fhir"
        return self.base_url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None
        self.base_url = ""
