"""Stable reasoning boundary for the isolated Qwen environment.

This deterministic adapter is deliberately small.  A future Qwen/Ollama/vLLM
implementation can replace this function without changing the orchestrator or
the consistency engine.
"""

from __future__ import annotations

from typing import Any


def run_qwen_reasoning(fusion_result: dict[str, Any]) -> dict[str, Any]:
    scores = fusion_result.get("hypothesis_scores", [])
    top = scores[0].get("hypothesis") if scores and isinstance(scores[0], dict) else None
    claims = []
    if top:
        claims.append(
            {
                "claim_id": "mock-primary-hypothesis",
                "claim_type": "cause",
                "subject": "change",
                "value": top,
            }
        )
    return {
        "adapter": "deterministic_mock_qwen",
        "claims": claims,
        "reasoning_summary": "Structured claims were generated from the fusion ranking for interface validation; this is not a causal conclusion.",
    }

