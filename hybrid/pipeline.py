"""
Hybrid pipeline — orchestrates all modules into a single generate call.

This is the glue that connects RAG, memory, validators, verifier loop,
router, and cache into one coherent system. Domain-agnostic — domain-specific
behavior comes from the validators and RAG corpus you plug in.

Usage:
    from hybrid.pipeline import HybridPipeline

    pipeline = HybridPipeline(
        generate_fn=my_model_generate,
        rag=rag_instance,
        memory=memory_instance,
        validators=[syntax_check, type_check],
    )

    result = pipeline.run("Create a toggle button component")
    print(result.output)
    print(result.tier)        # which routing tier was used
    print(result.modules)     # which modules fired
    print(result.latency_ms)  # total time
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from hybrid.cache import Cache
from hybrid.memory import StructuredMemory
from hybrid.router import Router
from hybrid.verifier import VerifierLoop


@dataclass
class PipelineResult:
    output: str
    tier: int
    modules_used: list
    latency_ms: float
    cache_hit: bool = False
    rag_context: str = ""
    memory_context: str = ""
    verifier_attempts: int = 1
    passed_validation: bool = True
    errors: list = field(default_factory=list)


class HybridPipeline:
    def __init__(
        self,
        generate_fn: Callable[[str], str],
        rag=None,
        memory: Optional[StructuredMemory] = None,
        validators: list = None,
        cache: Optional[Cache] = None,
        max_verifier_attempts: int = 3,
        use_self_consistency: bool = False,
        n_candidates: int = 5,
        use_bracket_balancing: bool = True,
        use_repetition_filter: bool = True,
    ):
        self.generate_fn = generate_fn
        self.rag = rag
        self.memory = memory or StructuredMemory()
        self.cache = cache or Cache()
        self.validators = validators or []
        self.max_attempts = max_verifier_attempts
        self.use_self_consistency = use_self_consistency
        self.n_candidates = n_candidates
        self.use_bracket_balancing = use_bracket_balancing
        self.use_repetition_filter = use_repetition_filter

        self.router = Router(cache=self.cache)

        # Post-processing
        from hybrid.constrained import BracketBalancer, RepetitionFilter, StopSequence
        self.stop_filter = StopSequence(lambda p, **kw: p)  # just the truncation logic
        self.bracket_balancer = BracketBalancer(lambda p, **kw: p)
        self.repetition_filter = RepetitionFilter(lambda p, **kw: p)

    def run(self, query: str) -> PipelineResult:
        """Run the full hybrid pipeline on a query."""
        t0 = time.time()
        modules_used = []

        # Route
        route = self.router.route(query)

        # Tier 0: Cache hit
        if route.cache_hit:
            cached = self.cache.get(query)
            if cached:
                return PipelineResult(
                    output=cached,
                    tier=0,
                    modules_used=["cache"],
                    latency_ms=(time.time() - t0) * 1000,
                    cache_hit=True,
                )

        # Build augmented prompt
        prompt_parts = []
        rag_context = ""
        memory_context = ""

        # Tier 2+: Add RAG context
        if route.level >= 2 and self.rag:
            rag_context = self.rag.augment_prompt(query, top_k=3)
            prompt_parts.append(rag_context)
            modules_used.append("rag")

        # Tier 3: Add memory context
        if route.level >= 3 and self.memory:
            memory_context = self.memory.get_context_for_prompt()
            if memory_context:
                prompt_parts.append(f"// Context from history:\n{memory_context}")
                modules_used.append("memory")

        # Add the actual query
        prompt_parts.append(query)
        full_prompt = "\n\n".join(prompt_parts)

        # Generate (with or without verifier loop)
        modules_used.append("generate")

        if self.use_self_consistency and route.level >= 2 and self.validators:
            # Self-consistency: generate N, pick best
            from hybrid.self_consistency import SelfConsistency
            sc = SelfConsistency(
                generate_fn=lambda p, temperature=0.7: self.generate_fn(p),
                validators=self.validators,
                n_candidates=self.n_candidates,
            )
            sc_result = sc.run(full_prompt)
            output = sc_result.output
            passed = sc_result.passed
            attempts = sc_result.n_generated
            errors = []
            modules_used.append(f"self_consistency({sc_result.n_passed}/{sc_result.n_generated})")

        elif route.level >= 2 and self.validators:
            # Use verifier loop
            verifier = VerifierLoop(
                generate_fn=self.generate_fn,
                validators=self.validators,
                max_attempts=self.max_attempts,
            )
            vresult = verifier.run(full_prompt)
            output = vresult.output
            passed = vresult.passed
            attempts = vresult.attempts
            errors = vresult.final_errors
            modules_used.append("verifier")
        else:
            # Direct generation
            output = self.generate_fn(full_prompt)
            passed = True
            attempts = 1
            errors = []

        # Post-processing
        if self.use_repetition_filter:
            output = self.repetition_filter.filter_repetition(output)
        output = self.stop_filter.truncate(output)
        if self.use_bracket_balancing:
            output = self.bracket_balancer.balance(output)
            errors = []

        # Record in memory
        self.memory.record_interaction(
            task_type=route.reason,
            input_text=query,
            output_text=output,
            success=passed,
            validator_passed=passed,
            was_repaired=attempts > 1,
            latency_ms=(time.time() - t0) * 1000,
            modules_used=modules_used,
        )

        if not passed and errors:
            self.memory.record_error(
                error_type="validation_failure",
                details="; ".join(errors[:3]),
                task_type=route.reason,
            )

        # Cache successful results
        if passed:
            self.cache.put(query, output)

        return PipelineResult(
            output=output,
            tier=route.level,
            modules_used=modules_used,
            latency_ms=(time.time() - t0) * 1000,
            rag_context=rag_context,
            memory_context=memory_context,
            verifier_attempts=attempts,
            passed_validation=passed,
            errors=errors,
        )
