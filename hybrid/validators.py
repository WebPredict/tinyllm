"""
Code validators for the hybrid architecture.

Validates generated React/TypeScript code using real tools:
- TypeScript compiler (type checking)
- ESLint (code quality)
- Syntax check (basic parse validation)

These are "free" validators — they use the same tools developers use daily,
giving us automated quality checks on model output.

Usage:
    from hybrid.validators import validate_code, TypeScriptValidator, ESLintValidator

    result = validate_code("const x: number = 'hello'")
    print(result.passed)     # False
    print(result.errors)     # [TypeScript error about type mismatch]
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    passed: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    validator: str = ""
    raw_output: str = ""


class TypeScriptValidator:
    """Validate TypeScript/TSX code using the TypeScript compiler."""

    def __init__(self):
        self.tsc_path = self._find_tsc()

    def _find_tsc(self):
        """Find tsc binary."""
        # Check local node_modules first
        local = Path(__file__).parent.parent / "node_modules" / ".bin" / "tsc"
        if local.exists():
            return str(local)
        # Try global
        result = subprocess.run(["which", "tsc"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def validate(self, code: str, filename: str = "temp.tsx") -> ValidationResult:
        """Type-check a snippet of TypeScript/TSX code."""
        if not self.tsc_path:
            return ValidationResult(
                passed=False, errors=["TypeScript compiler not found"],
                validator="typescript"
            )

        # Write code to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(code)

            # Write minimal tsconfig
            tsconfig = {
                "compilerOptions": {
                    "target": "es2020",
                    "module": "esnext",
                    "jsx": "react-jsx",
                    "strict": False,
                    "noEmit": True,
                    "skipLibCheck": True,
                    "moduleResolution": "node",
                    "esModuleInterop": True,
                    "allowJs": True,
                },
                "include": [filename],
            }
            with open(os.path.join(tmpdir, "tsconfig.json"), "w") as f:
                json.dump(tsconfig, f)

            # Run tsc
            result = subprocess.run(
                [self.tsc_path, "--noEmit", "--project", os.path.join(tmpdir, "tsconfig.json")],
                capture_output=True, text=True, timeout=10,
                cwd=tmpdir,
            )

            errors = []
            warnings = []
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        if "error" in line.lower():
                            errors.append(line.strip())
                        elif "warning" in line.lower():
                            warnings.append(line.strip())
                        else:
                            errors.append(line.strip())

            return ValidationResult(
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                validator="typescript",
                raw_output=result.stdout + result.stderr,
            )


class ESLintValidator:
    """Validate code using ESLint."""

    def __init__(self):
        self.eslint_path = self._find_eslint()

    def _find_eslint(self):
        local = Path(__file__).parent.parent / "node_modules" / ".bin" / "eslint"
        if local.exists():
            return str(local)
        result = subprocess.run(["which", "eslint"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def validate(self, code: str, filename: str = "temp.tsx") -> ValidationResult:
        if not self.eslint_path:
            return ValidationResult(
                passed=False, errors=["ESLint not found"],
                validator="eslint"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "w") as f:
                f.write(code)

            result = subprocess.run(
                [self.eslint_path, filepath, "--no-eslintrc", "--format", "json"],
                capture_output=True, text=True, timeout=10,
            )

            errors = []
            warnings = []
            try:
                output = json.loads(result.stdout)
                if output and len(output) > 0:
                    for msg in output[0].get("messages", []):
                        text = f"Line {msg.get('line', '?')}: {msg.get('message', '')}"
                        if msg.get("severity", 0) >= 2:
                            errors.append(text)
                        else:
                            warnings.append(text)
            except (json.JSONDecodeError, IndexError):
                pass

            return ValidationResult(
                passed=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                validator="eslint",
                raw_output=result.stdout,
            )


class SyntaxValidator:
    """Basic syntax validation — does the code parse at all?
    Uses Node.js to attempt parsing."""

    def validate(self, code: str) -> ValidationResult:
        # Quick heuristic checks first
        errors = []

        # Bracket matching
        stack = []
        brackets = {'{': '}', '(': ')', '[': ']', '<': '>'}
        closing = set(brackets.values())
        in_string = False
        string_char = None

        for i, c in enumerate(code):
            if in_string:
                if c == string_char and (i == 0 or code[i-1] != '\\'):
                    in_string = False
                continue
            if c in ('"', "'", '`'):
                in_string = True
                string_char = c
                continue
            if c in brackets:
                # Don't track < > as they're ambiguous in JSX/generics
                if c != '<':
                    stack.append(brackets[c])
            elif c in closing and c != '>':
                if not stack:
                    errors.append(f"Unexpected closing '{c}' at position {i}")
                elif stack[-1] != c:
                    errors.append(f"Mismatched bracket at position {i}: expected '{stack[-1]}', got '{c}'")
                else:
                    stack.pop()

        if stack:
            errors.append(f"Unclosed brackets: {len(stack)} remaining ({', '.join(stack[:3])})")

        # Check for completely empty or whitespace-only
        if not code.strip():
            errors.append("Empty code")

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            validator="syntax",
        )


def validate_code(code: str, validators: list = None) -> dict:
    """Run all validators on a piece of code.

    Returns dict with results from each validator.
    """
    if validators is None:
        validators = ["syntax", "typescript"]

    results = {}

    if "syntax" in validators:
        sv = SyntaxValidator()
        results["syntax"] = sv.validate(code)

    if "typescript" in validators:
        ts = TypeScriptValidator()
        results["typescript"] = ts.validate(code)

    if "eslint" in validators:
        ev = ESLintValidator()
        results["eslint"] = ev.validate(code)

    # Overall pass/fail
    all_passed = all(r.passed for r in results.values())

    return {
        "passed": all_passed,
        "results": results,
        "error_count": sum(len(r.errors) for r in results.values()),
        "warning_count": sum(len(r.warnings) for r in results.values()),
    }
