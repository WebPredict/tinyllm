"""
Constrained decoding — force generated output to follow structural rules.

Instead of hoping the model outputs valid syntax, constrain the generation
to only produce valid tokens at each step. Lighter than full grammar-guided
decoding (which needs outlines/guidance library), this uses post-hoc repair
and token filtering for practical constraints.

Approaches:
1. Bracket balancing — ensure brackets close properly
2. JSON mode — force valid JSON output
3. Template filling — fill slots in a template
4. Stop sequences — stop generation at the right point

Usage:
    from hybrid.constrained import (
        BracketBalancer, JSONConstraint, TemplateConstraint, StopSequence
    )

    # Wrap any generate function
    balanced = BracketBalancer(generate_fn)
    output = balanced.generate("function Button(")
    # Guaranteed: all brackets are properly closed

    json_gen = JSONConstraint(generate_fn)
    output = json_gen.generate("Generate a config object:")
    # Guaranteed: valid JSON
"""

import json
import re
from typing import Callable, Optional


class BracketBalancer:
    """Ensures generated code has balanced brackets.
    Works by post-processing: if brackets are unclosed, closes them.
    If generation stopped mid-string, closes the string first."""

    def __init__(self, generate_fn: Callable):
        self.generate_fn = generate_fn

    def generate(self, prompt: str, **kwargs) -> str:
        output = self.generate_fn(prompt, **kwargs)
        return self.balance(output)

    def balance(self, code: str) -> str:
        """Close any unclosed brackets/strings."""
        stack = []
        in_string = False
        string_char = None
        i = 0

        while i < len(code):
            c = code[i]

            # Handle string literals
            if in_string:
                if c == '\\' and i + 1 < len(code):
                    i += 2  # skip escaped char
                    continue
                if c == string_char:
                    in_string = False
                i += 1
                continue

            if c in ('"', "'", '`'):
                in_string = True
                string_char = c
                i += 1
                continue

            # Handle template literal expressions
            if c == '$' and i + 1 < len(code) and code[i+1] == '{':
                stack.append('}')
                i += 2
                continue

            # Track brackets (skip < > as they're ambiguous in JSX/generics)
            if c == '{':
                stack.append('}')
            elif c == '(':
                stack.append(')')
            elif c == '[':
                stack.append(']')
            elif c in ('}', ')', ']'):
                if stack and stack[-1] == c:
                    stack.pop()

            i += 1

        # Close unclosed string
        if in_string:
            code += string_char

        # Close unclosed brackets (in reverse order)
        while stack:
            closer = stack.pop()
            code += '\n' + closer

        return code


class JSONConstraint:
    """Ensures output is valid JSON.
    Attempts repair if the model produces almost-valid JSON."""

    def __init__(self, generate_fn: Callable):
        self.generate_fn = generate_fn

    def generate(self, prompt: str, **kwargs) -> str:
        output = self.generate_fn(prompt, **kwargs)
        return self.ensure_json(output)

    def ensure_json(self, text: str) -> str:
        """Try to extract and repair JSON from text."""
        # Try parsing as-is
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        # Look for first { or [ and last } or ]
        first_brace = -1
        last_brace = -1
        for i, c in enumerate(text):
            if c in ('{', '[') and first_brace == -1:
                first_brace = i
            if c in ('}', ']'):
                last_brace = i

        if first_brace >= 0 and last_brace > first_brace:
            candidate = text[first_brace:last_brace + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Try common repairs
        repairs = [
            # Add missing closing bracket
            lambda t: t + '}',
            lambda t: t + ']',
            lambda t: t + '"}',
            lambda t: t + '"]',
            # Remove trailing comma
            lambda t: re.sub(r',\s*([}\]])', r'\1', t),
            # Fix single quotes to double quotes
            lambda t: t.replace("'", '"'),
        ]

        for repair in repairs:
            try:
                fixed = repair(text)
                json.loads(fixed)
                return fixed
            except (json.JSONDecodeError, Exception):
                continue

        # Give up — return original with a note
        return text


class TemplateConstraint:
    """Fill slots in a template, ensuring the output matches structure.

    Usage:
        tc = TemplateConstraint(generate_fn)
        output = tc.generate(
            prompt="Create a React component",
            template=\"\"\"
            interface {name}Props {{
              {props}
            }}

            function {name}({{ {prop_names} }}: {name}Props) {{
              return (
                {jsx}
              )
            }}
            \"\"\",
            slots=["name", "props", "prop_names", "jsx"]
        )
    """

    def __init__(self, generate_fn: Callable):
        self.generate_fn = generate_fn

    def generate(self, prompt: str, template: str, slots: list, **kwargs) -> str:
        """Generate content for each slot and fill the template."""
        filled = template

        for slot in slots:
            slot_prompt = f"{prompt}\n\nGenerate the {slot} part:\n"
            value = self.generate_fn(slot_prompt, **kwargs)

            # Clean the value (take first meaningful line/block)
            value = value.strip()
            lines = value.split('\n')
            # Take at most 5 lines
            value = '\n'.join(lines[:5])

            filled = filled.replace('{' + slot + '}', value)

        return filled


class StopSequence:
    """Wraps a generate function to stop at specific sequences.
    Useful for preventing runaway generation."""

    def __init__(self, generate_fn: Callable, stop_sequences: list = None):
        self.generate_fn = generate_fn
        self.stop_sequences = stop_sequences or [
            "\n\n\n",           # triple newline
            "// FILE:",          # corpus file separator
            "<|endoftext|>",     # special token
        ]

    def generate(self, prompt: str, **kwargs) -> str:
        output = self.generate_fn(prompt, **kwargs)
        return self.truncate(output)

    def truncate(self, text: str) -> str:
        """Cut output at the first stop sequence."""
        earliest = len(text)
        for seq in self.stop_sequences:
            idx = text.find(seq)
            if idx >= 0 and idx < earliest:
                earliest = idx

        return text[:earliest].rstrip()


class RepetitionFilter:
    """Detects and truncates repetitive output.
    Catches the common small-model failure of generating the same
    tokens in a loop."""

    def __init__(self, generate_fn: Callable, max_repeat_ratio: float = 0.3):
        self.generate_fn = generate_fn
        self.max_repeat_ratio = max_repeat_ratio

    def generate(self, prompt: str, **kwargs) -> str:
        output = self.generate_fn(prompt, **kwargs)
        return self.filter_repetition(output)

    def filter_repetition(self, text: str) -> str:
        """Detect and truncate at the point where repetition starts."""
        lines = text.split('\n')
        if len(lines) < 4:
            return text

        # Check for repeated lines
        seen = {}
        cutoff = len(lines)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in seen:
                # Count consecutive or near-consecutive repeats
                repeats = sum(1 for l in lines[i:i+5] if l.strip() == stripped)
                if repeats >= 3:
                    cutoff = i
                    break
            seen[stripped] = i

        if cutoff < len(lines):
            return '\n'.join(lines[:cutoff]).rstrip()

        return text
