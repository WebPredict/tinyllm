"""
Self-consistency — generate multiple candidates and pick the best one.

Instead of hoping one generation is correct, generate N candidates,
validate each, and select the best. Works with any generator and validators.

Usage:
    from hybrid.self_consistency import SelfConsistency

    sc = SelfConsistency(
        generate_fn=my_model_generate,
        validators=[syntax_check, type_check],
        n_candidates=5,
    )
    result = sc.run("function Button(")
    print(result.output)         # best candidate
    print(result.candidates)     # all N candidates with scores
    print(result.pass_rate)      # what fraction passed validation
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Candidate:
    output: str
    passed: bool
    errors: list
    score: float  # 0-1, higher is better
    temperature: float = 0.0


@dataclass
class SelfConsistencyResult:
    output: str               # best candidate
    passed: bool              # did the best candidate pass validation?
    candidates: list          # all candidates with scores
    n_generated: int
    n_passed: int
    pass_rate: float
    total_time_ms: float
    selection_method: str     # how the winner was chosen


class SelfConsistency:
    def __init__(
        self,
        generate_fn: Callable,
        validators: list = None,
        n_candidates: int = 5,
        temperature_range: tuple = (0.5, 1.0),
        selection: str = "best_score",  # "best_score", "majority", "first_valid"
    ):
        """
        Args:
            generate_fn: Function(prompt, temperature) → str
            validators: List of validator functions
            n_candidates: Number of candidates to generate
            temperature_range: (min, max) temperature for diversity
            selection: How to pick the winner
                "best_score": highest validation score
                "first_valid": first candidate that passes all validators
                "majority": most common output among valid candidates
        """
        self.generate_fn = generate_fn
        self.validators = validators or []
        self.n_candidates = n_candidates
        self.temp_min, self.temp_max = temperature_range
        self.selection = selection

    def _score_candidate(self, output: str) -> tuple:
        """Run validators and compute a score. Returns (passed, errors, score)."""
        if not self.validators:
            return True, [], 1.0

        all_errors = []
        checks_passed = 0
        total_checks = 0

        for validator in self.validators:
            result = validator(output)

            if hasattr(result, 'passed'):
                passed = result.passed
                errors = getattr(result, 'errors', [])
            elif isinstance(result, dict):
                passed = result.get('passed', True)
                errors = result.get('errors', [])
            else:
                passed = bool(result)
                errors = [] if passed else ["Validation failed"]

            total_checks += 1
            if passed:
                checks_passed += 1
            else:
                all_errors.extend(errors)

        score = checks_passed / max(total_checks, 1)
        all_passed = len(all_errors) == 0

        return all_passed, all_errors, score

    def _compute_diversity_bonus(self, output: str, previous_outputs: list) -> float:
        """Small bonus for outputs that are different from others.
        Helps avoid picking from a cluster of identical bad outputs."""
        if not previous_outputs:
            return 0

        # Simple character-level similarity check
        similarities = []
        for prev in previous_outputs:
            common = sum(1 for a, b in zip(output, prev) if a == b)
            max_len = max(len(output), len(prev), 1)
            similarities.append(common / max_len)

        avg_similarity = sum(similarities) / len(similarities)
        # Bonus for being different (0 if identical, small bonus if diverse)
        return (1 - avg_similarity) * 0.1

    def run(self, prompt: str) -> SelfConsistencyResult:
        """Generate N candidates, validate, and pick the best."""
        t0 = time.time()
        candidates = []
        previous_outputs = []

        for i in range(self.n_candidates):
            # Vary temperature across candidates for diversity
            if self.n_candidates > 1:
                t = self.temp_min + (self.temp_max - self.temp_min) * (i / (self.n_candidates - 1))
            else:
                t = (self.temp_min + self.temp_max) / 2

            # Generate
            try:
                output = self.generate_fn(prompt, temperature=t)
            except Exception as e:
                output = f"Error: {e}"

            # Score
            passed, errors, score = self._score_candidate(output)

            # Small diversity bonus
            diversity = self._compute_diversity_bonus(output, previous_outputs)
            final_score = score + diversity

            candidates.append(Candidate(
                output=output,
                passed=passed,
                errors=errors,
                score=final_score,
                temperature=t,
            ))
            previous_outputs.append(output)

            # Early exit for "first_valid" mode
            if self.selection == "first_valid" and passed:
                break

        # Select winner
        n_passed = sum(1 for c in candidates if c.passed)

        if self.selection == "first_valid":
            valid = [c for c in candidates if c.passed]
            winner = valid[0] if valid else max(candidates, key=lambda c: c.score)
            method = "first_valid" if valid else "best_score_fallback"

        elif self.selection == "majority":
            # Group by output, pick most common among valid
            valid = [c for c in candidates if c.passed]
            if valid:
                # Simple: pick the output that appears most often
                from collections import Counter
                counts = Counter(c.output for c in valid)
                most_common_output = counts.most_common(1)[0][0]
                winner = next(c for c in valid if c.output == most_common_output)
                method = "majority"
            else:
                winner = max(candidates, key=lambda c: c.score)
                method = "best_score_fallback"

        else:  # best_score
            winner = max(candidates, key=lambda c: c.score)
            method = "best_score"

        total_time = (time.time() - t0) * 1000

        return SelfConsistencyResult(
            output=winner.output,
            passed=winner.passed,
            candidates=candidates,
            n_generated=len(candidates),
            n_passed=n_passed,
            pass_rate=n_passed / len(candidates),
            total_time_ms=total_time,
            selection_method=method,
        )
