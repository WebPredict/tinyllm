"""
Test the code validators with sample inputs.
Verifies that validators correctly catch errors and pass valid code.

Usage: python scripts/test_validators.py
"""

import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hybrid.validators import validate_code, TypeScriptValidator, SyntaxValidator


def test(name, code, expect_pass):
    """Run validators and check if result matches expectation."""
    result = validate_code(code)
    status = "PASS" if result["passed"] == expect_pass else "FAIL"
    icon = "✓" if status == "PASS" else "✗"
    print(f"  {icon} {name}")
    if status == "FAIL":
        print(f"    Expected {'pass' if expect_pass else 'fail'}, got {'pass' if result['passed'] else 'fail'}")
    for vname, vresult in result["results"].items():
        if vresult.errors:
            print(f"    {vname}: {', '.join(vresult.errors[:3])}")
    return status == "PASS"


print("=" * 60)
print("  Validator Tests")
print("=" * 60)
print()

passed = 0
total = 0

# --- Valid code (should pass) ---
print("Valid code (should pass):")

total += 1; passed += test("Simple variable",
    "const x: number = 42",
    expect_pass=True)

total += 1; passed += test("Arrow function",
    "const add = (a: number, b: number): number => a + b",
    expect_pass=True)

total += 1; passed += test("React component",
    """
function Button({ children }: { children: React.ReactNode }) {
  return <button>{children}</button>
}
""",
    expect_pass=True)

total += 1; passed += test("Interface",
    """
interface User {
  id: string
  name: string
  email: string
}
""",
    expect_pass=True)

total += 1; passed += test("useState hook",
    """
const [count, setCount] = React.useState(0)
""",
    expect_pass=True)

print()

# --- Invalid code (should fail) ---
print("Invalid code (should fail):")

total += 1; passed += test("Unclosed brace",
    "function broken() {",
    expect_pass=False)

total += 1; passed += test("Unclosed paren",
    "const x = foo(bar(",
    expect_pass=False)

total += 1; passed += test("Empty code",
    "",
    expect_pass=False)

total += 1; passed += test("Mismatched brackets",
    "const x = [1, 2, 3}",
    expect_pass=False)

print()

# --- TypeScript type errors (should fail if TS validator works) ---
print("TypeScript type errors (depends on tsc availability):")

ts = TypeScriptValidator()
if ts.tsc_path:
    total += 1; passed += test("Type mismatch",
        "const x: number = 'hello'",
        expect_pass=False)

    total += 1; passed += test("Undefined variable",
        "console.log(undefinedVar)",
        expect_pass=True)  # Not an error in non-strict mode

    total += 1; passed += test("Valid typed code",
        """
interface Props {
  name: string
  age: number
}

function greet(props: Props): string {
  return 'Hello ' + props.name
}
""",
        expect_pass=True)
else:
    print("  (skipped — tsc not found)")

print()
print("=" * 60)
print(f"  Results: {passed}/{total} tests passed")
print("=" * 60)
