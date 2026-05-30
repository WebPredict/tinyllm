"""
Query router — decides how much scaffolding each query needs.

Routes queries to different tiers based on complexity, confidence,
and whether the query matches cached results. This saves latency
and compute by only using heavy modules when necessary.

Domain-agnostic — routing decisions are based on general signals.

Usage:
    router = Router()
    tier = router.route("Add types to this function")
    # tier.level = 2 (needs RAG + validation)
    # tier.modules = ["rag", "validator"]
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class RoutingDecision:
    level: int              # 0=cache, 1=direct, 2=rag, 3=full pipeline
    modules: list           # which modules to activate
    reason: str             # why this tier was chosen
    cache_hit: bool = False
    confidence: float = 0.0


class Router:
    def __init__(self, cache: Optional['Cache'] = None,
                 confidence_fn: Optional[Callable] = None):
        """
        Args:
            cache: Optional cache instance for tier 0 (instant responses)
            confidence_fn: Optional function that estimates model confidence
                          on a query. Returns float 0-1.
        """
        self.cache = cache
        self.confidence_fn = confidence_fn

        # Configurable thresholds
        self.confidence_high = 0.8    # above this → direct generation
        self.confidence_medium = 0.5  # above this → RAG-augmented
        # below medium → full pipeline

        # Keyword signals for complexity estimation
        self.simple_signals = [
            "add types", "rename", "format", "import", "export",
            "const ", "let ", "var ", "interface ", "type ",
        ]
        self.complex_signals = [
            "refactor", "redesign", "optimize", "debug", "fix",
            "create", "build", "implement", "component", "hook",
            "test", "explain", "why", "how does",
        ]

    def route(self, query: str) -> RoutingDecision:
        """Decide which tier to use for a query."""

        # Tier 0: Cache hit
        if self.cache:
            cached = self.cache.get(query)
            if cached is not None:
                return RoutingDecision(
                    level=0,
                    modules=[],
                    reason="Cache hit",
                    cache_hit=True,
                )

        # Estimate complexity
        complexity = self._estimate_complexity(query)

        # Check model confidence if available
        confidence = None
        if self.confidence_fn:
            confidence = self.confidence_fn(query)

        # Route based on complexity and confidence
        if complexity == "simple" and (confidence is None or confidence > self.confidence_high):
            return RoutingDecision(
                level=1,
                modules=["generate"],
                reason=f"Simple query, high confidence",
                confidence=confidence or 0,
            )

        if complexity == "medium" or (confidence and confidence > self.confidence_medium):
            return RoutingDecision(
                level=2,
                modules=["rag", "generate", "validator"],
                reason=f"Medium complexity, using RAG + validation",
                confidence=confidence or 0,
            )

        # Complex or low confidence → full pipeline
        return RoutingDecision(
            level=3,
            modules=["rag", "memory", "generate", "validator", "verifier"],
            reason=f"Complex query, using full pipeline",
            confidence=confidence or 0,
        )

    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity from surface signals."""
        query_lower = query.lower()
        query_len = len(query.split())

        # Very short queries are usually simple
        if query_len <= 3:
            return "simple"

        # Check for complex signals
        complex_count = sum(1 for s in self.complex_signals if s in query_lower)
        simple_count = sum(1 for s in self.simple_signals if s in query_lower)

        if complex_count > simple_count:
            return "complex"
        if simple_count > 0:
            return "simple"

        # Default based on length
        if query_len > 15:
            return "complex"
        return "medium"
