"""
Verifier loop — generic generate → validate → revise cycle.

Takes any generator function and any list of validators, runs the generation,
checks the output, and if it fails, feeds errors back for revision.
Domain-agnostic — works with any validator plugins.

Usage:
    from hybrid.verifier import VerifierLoop

    loop = VerifierLoop(
        generate_fn=my_model_generate,
        validators=[syntax_check, type_check, lint_check],
        max_attempts=3,
    )
    result = loop.run(prompt="Create a button component")
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class VerifierResult:
    output: str
    passed: bool
    attempts: int
    total_time_ms: float
    history: list = field(default_factory=list)
    final_errors: list = field(default_factory=list)


class VerifierLoop:
    def __init__(
        self,
        generate_fn: Callable[[str], str],
        validators: list = None,
        max_attempts: int = 3,
        revision_prompt_fn: Optional[Callable] = None,
    ):
        """
        Args:
            generate_fn: Function that takes a prompt string and returns generated text.
            validators: List of validator functions. Each takes a string and returns
                       an object with .passed (bool) and .errors (list of strings).
            max_attempts: Maximum number of generate-validate-revise cycles.
            revision_prompt_fn: Optional function that builds a revision prompt from
                               the original prompt, the failed output, and error list.
                               If None, uses a default format.
        """
        self.generate_fn = generate_fn
        self.validators = validators or []
        self.max_attempts = max_attempts
        self.revision_prompt_fn = revision_prompt_fn or self._default_revision_prompt

    def _default_revision_prompt(self, original_prompt: str, failed_output: str,
                                  errors: list) -> str:
        """Default revision prompt format."""
        error_text = "\n".join(f"- {e}" for e in errors[:5])
        return (
            f"{original_prompt}\n\n"
            f"// Previous attempt had errors:\n"
            f"{error_text}\n\n"
            f"// Previous output:\n"
            f"{failed_output[:500]}\n\n"
            f"// Please fix these errors:\n"
        )

    def _run_validators(self, output: str) -> tuple:
        """Run all validators on the output. Returns (passed, errors, details)."""
        all_errors = []
        all_details = []

        for validator in self.validators:
            result = validator(output)

            # Support both object-style (.passed, .errors) and dict-style
            if hasattr(result, 'passed'):
                passed = result.passed
                errors = getattr(result, 'errors', [])
            elif isinstance(result, dict):
                passed = result.get('passed', True)
                errors = result.get('errors', [])
            else:
                # Assume boolean
                passed = bool(result)
                errors = [] if passed else ["Validation failed"]

            validator_name = getattr(validator, '__name__',
                                    getattr(validator, 'validator', 'unknown'))
            all_details.append({
                "validator": validator_name,
                "passed": passed,
                "errors": errors,
            })

            if not passed:
                all_errors.extend(errors)

        all_passed = len(all_errors) == 0
        return all_passed, all_errors, all_details

    def run(self, prompt: str) -> VerifierResult:
        """Run the generate-validate-revise loop."""
        t0 = time.time()
        history = []
        current_prompt = prompt

        for attempt in range(1, self.max_attempts + 1):
            # Generate
            t_gen = time.time()
            output = self.generate_fn(current_prompt)
            gen_time = (time.time() - t_gen) * 1000

            # Validate
            if self.validators:
                t_val = time.time()
                passed, errors, details = self._run_validators(output)
                val_time = (time.time() - t_val) * 1000
            else:
                passed, errors, details = True, [], []
                val_time = 0

            history.append({
                "attempt": attempt,
                "prompt": current_prompt[:200],
                "output": output[:500],
                "passed": passed,
                "errors": errors,
                "gen_time_ms": gen_time,
                "val_time_ms": val_time,
                "validator_details": details,
            })

            if passed:
                return VerifierResult(
                    output=output,
                    passed=True,
                    attempts=attempt,
                    total_time_ms=(time.time() - t0) * 1000,
                    history=history,
                )

            # Build revision prompt for next attempt
            if attempt < self.max_attempts:
                current_prompt = self.revision_prompt_fn(prompt, output, errors)

        # Failed after all attempts
        total_time = (time.time() - t0) * 1000
        return VerifierResult(
            output=output,  # return last attempt
            passed=False,
            attempts=self.max_attempts,
            total_time_ms=total_time,
            history=history,
            final_errors=errors,
        )
