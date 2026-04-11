import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import aiohttp

from agent_cap.core.evaluator import Evaluator, EvalResult


logger = logging.getLogger("agent_cap.gtfa_eval")


def get_single_claim_evaluation_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "claim_text": {"type": "string"},
            "coverage_outcome": {
                "type": "string",
                "enum": ["fulfilled", "partially_fulfilled", "not_fulfilled"],
            },
            "justification": {"type": "string"},
            "confidence_level": {"type": "number"},
        },
        "required": [
            "claim_text",
            "coverage_outcome",
            "justification",
            "confidence_level",
        ],
    }


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def _runner() -> Any:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()


class GTFAEvaluator(Evaluator):
    def __init__(
        self,
        judge_model: str = "google/gemini-3.1-flash-lite-preview",
        judge_base_url: str = "https://openrouter.ai/api/v1",
        judge_api_key: Optional[str] = None,
        pass_threshold: float = 0.75,
    ):
        self.judge_model = judge_model
        self.judge_base_url = judge_base_url.rstrip("/")
        self.judge_api_key = judge_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.pass_threshold = pass_threshold

    @staticmethod
    def _get_single_claim_evaluation_prompt(claim: str, response: str) -> str:
        return f"""You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning."""

    @staticmethod
    def _parse_message_content(content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            content = "\n".join(text_parts)
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    async def _judge_claim(
        self, session: aiohttp.ClientSession, response: str, claim: str
    ) -> Dict[str, Any]:
        prompt = self._get_single_claim_evaluation_prompt(claim, response)

        headers: Dict[str, str] = {}
        if self.judge_api_key:
            headers["Authorization"] = f"Bearer {self.judge_api_key}"

        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_object",
                "response_schema": get_single_claim_evaluation_schema(),
            },
            "temperature": 0.0,
            "stream": False,
        }
        try:
            async with session.post(
                f"{self.judge_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"judge failed ({resp.status}): {body}")
                result = await resp.json()
                parsed = self._parse_message_content(
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
        except Exception as exc:
            logger.error("GTFA judge error for claim %r: %s", claim, exc)
            return {
                "claim_text": claim,
                "coverage_outcome": "not_fulfilled",
                "justification": f"Evaluation failed: {exc}",
                "confidence_level": 0.1,
            }

        coverage_outcome = str(parsed.get("coverage_outcome", "not_fulfilled")).strip()
        if coverage_outcome not in {"fulfilled", "partially_fulfilled", "not_fulfilled"}:
            coverage_outcome = "not_fulfilled"
        return {
            "claim_text": str(parsed.get("claim_text", claim)),
            "coverage_outcome": coverage_outcome,
            "justification": str(parsed.get("justification", "")),
            "confidence_level": float(parsed.get("confidence_level", 0.5) or 0.5),
        }

    async def _evaluate_async(self, claims: List[str], response: str) -> Dict[str, Any]:
        coverage_to_score = {
            "fulfilled": 1.0,
            "partially_fulfilled": 0.5,
            "not_fulfilled": 0.0,
        }

        async with aiohttp.ClientSession() as session:
            tasks = [self._judge_claim(session, response, claim) for claim in claims]
            claim_results = await asyncio.gather(*tasks)

        per_claim: List[Dict[str, Any]] = []
        total_score = 0.0
        fulfilled_count = 0
        partially_fulfilled_count = 0
        total_confidence = 0.0

        for result in claim_results:
            coverage_outcome = result.get("coverage_outcome", "not_fulfilled")
            score = coverage_to_score.get(coverage_outcome, 0.0)
            total_score += score
            total_confidence += float(result.get("confidence_level", 0.5) or 0.5)

            if score >= 1.0:
                fulfilled_count += 1
                covered: Any = True
            elif score >= 0.5:
                partially_fulfilled_count += 1
                covered = "partial"
            else:
                covered = False

            per_claim.append(
                {
                    "claim": result.get("claim_text", ""),
                    "coverage_outcome": coverage_outcome,
                    "score": score,
                    "covered": covered,
                    "reasoning": result.get("justification", ""),
                    "confidence_level": float(result.get("confidence_level", 0.5) or 0.5),
                }
            )

        coverage_score = round(total_score / len(claims), 3) if claims else 0.0
        avg_confidence = total_confidence / len(claims) if claims else 0.5
        return {
            "per_claim": per_claim,
            "coverage_score": coverage_score,
            "total_claims": len(claims),
            "fully_covered_claims": fulfilled_count,
            "partially_covered_claims": partially_fulfilled_count,
            "explanation": "Evaluation complete",
            "confidence": avg_confidence,
        }

    def evaluate(self, task_config: Dict[str, Any], backend: Any) -> EvalResult:
        claims = task_config.get("gtfa_claims", []) or []
        if not isinstance(claims, list):
            claims = [str(claims)]
        claims = [str(c).strip() for c in claims if str(c).strip()]
        response = str(task_config.get("response", "") or "").strip()

        if not claims:
            return EvalResult(
                passed=True,
                score=1.0,
                details={
                    "per_claim": [],
                    "coverage_score": None,
                    "fully_covered_claims": 0,
                    "partially_covered_claims": 0,
                    "total_claims": 0,
                    "explanation": "No claims provided",
                    "confidence": 1.0,
                },
            )

        if not response:
            per_claim = [
                {
                    "claim": claim,
                    "coverage_outcome": "not_fulfilled",
                    "score": 0.0,
                    "covered": False,
                    "reasoning": "empty response",
                    "confidence_level": 1.0,
                }
                for claim in claims
            ]
            return EvalResult(
                passed=False,
                score=0.0,
                details={
                    "per_claim": per_claim,
                    "coverage_score": 0.0,
                    "fully_covered_claims": 0,
                    "partially_covered_claims": 0,
                    "total_claims": len(claims),
                    "explanation": "Empty response",
                    "confidence": 1.0,
                },
            )

        details = _run_coro_sync(self._evaluate_async(claims, response))
        score = float(details.get("coverage_score", 0.0) or 0.0)

        return EvalResult(
            passed=score >= self.pass_threshold,
            score=score,
            details=details,
        )
