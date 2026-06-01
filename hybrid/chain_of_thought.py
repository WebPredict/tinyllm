"""
Chain-of-thought / scratchpad module.

Makes the model reason explicitly before producing a final answer.
Instead of going directly from instruction to code, the model first
writes out its thinking, then produces the answer.

This helps with:
- Multi-step tasks ("add types, then write a test")
- Debugging ("what's wrong with this code?")
- Decision-making ("should I use useState or useReducer?")

Domain-agnostic — works with any generate function.

Usage:
    from hybrid.chain_of_thought import ChainOfThought

    cot = ChainOfThought(generate_fn)
    result = cot.run("Add error handling to this fetch call")
    print(result.thinking)    # the reasoning trace
    print(result.answer)      # the final code
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CoTResult:
    thinking: str           # the reasoning trace
    answer: str             # the final output
    full_output: str        # thinking + answer combined
    thinking_tokens: int    # how many tokens spent on reasoning
    answer_tokens: int      # how many tokens in the answer
    total_time_ms: float
    used_cot: bool          # whether CoT was actually triggered


# Prompts that wrap the user's query with thinking instructions
COT_PROMPT_TEMPLATE = """<|thinking|>
Let me think about this step by step:
- What is being asked?
- What patterns or approaches apply?
- What are the key decisions?

Task: {instruction}
{input_section}
My reasoning:
"""

COT_ANSWER_TEMPLATE = """
<|answer|>
Based on my reasoning above, here is the solution:
"""


class ChainOfThought:
    def __init__(
        self,
        generate_fn: Callable,
        thinking_tokens: int = 100,
        answer_tokens: int = 150,
        min_complexity: int = 5,    # minimum word count to trigger CoT
    ):
        """
        Args:
            generate_fn: Function(prompt, max_tokens, temperature) → str
            thinking_tokens: Max tokens for the reasoning trace
            answer_tokens: Max tokens for the final answer
            min_complexity: Only use CoT for queries with this many+ words
        """
        self.generate_fn = generate_fn
        self.thinking_tokens = thinking_tokens
        self.answer_tokens = answer_tokens
        self.min_complexity = min_complexity

    def should_use_cot(self, query: str) -> bool:
        """Decide whether this query benefits from chain-of-thought."""
        words = query.lower().split()

        # Too short — just generate directly
        if len(words) < self.min_complexity:
            return False

        # Keywords that suggest reasoning would help
        reasoning_signals = [
            "why", "how", "explain", "debug", "fix", "what's wrong",
            "compare", "difference", "choose", "decide", "should",
            "refactor", "optimize", "improve", "best way",
            "step by step", "complex", "multiple",
        ]

        return any(signal in query.lower() for signal in reasoning_signals)

    def run(self, instruction: str, input_text: str = "",
            force_cot: bool = False) -> CoTResult:
        """Run generation with optional chain-of-thought reasoning."""
        t0 = time.time()

        use_cot = force_cot or self.should_use_cot(instruction)

        if not use_cot:
            # Direct generation without CoT
            if input_text:
                prompt = f"<|instruction|>\n{instruction}\n<|input|>\n{input_text}\n<|output|>\n"
            else:
                prompt = f"<|instruction|>\n{instruction}\n<|input|>\n\n<|output|>\n"

            answer = self.generate_fn(prompt, max_tokens=self.answer_tokens)

            return CoTResult(
                thinking="",
                answer=answer,
                full_output=answer,
                thinking_tokens=0,
                answer_tokens=len(answer.split()),
                total_time_ms=(time.time() - t0) * 1000,
                used_cot=False,
            )

        # CoT: first generate thinking, then answer
        input_section = f"Input code:\n{input_text}\n" if input_text else ""
        thinking_prompt = COT_PROMPT_TEMPLATE.format(
            instruction=instruction,
            input_section=input_section,
        )

        # Generate reasoning
        thinking = self.generate_fn(
            thinking_prompt,
            max_tokens=self.thinking_tokens,
            temperature=0.7,
        )

        # Clean up thinking — stop at answer marker if model generated one
        if "<|answer|>" in thinking:
            thinking = thinking.split("<|answer|>")[0]

        # Now generate the answer, with thinking as context
        answer_prompt = (
            f"{thinking_prompt}{thinking}\n"
            f"{COT_ANSWER_TEMPLATE}"
        )

        if input_text:
            answer_prompt += f"\n<|instruction|>\n{instruction}\n<|input|>\n{input_text}\n<|output|>\n"
        else:
            answer_prompt += f"\n<|instruction|>\n{instruction}\n<|input|>\n\n<|output|>\n"

        answer = self.generate_fn(
            answer_prompt,
            max_tokens=self.answer_tokens,
            temperature=0.5,  # lower temp for the answer
        )

        # Clean up
        if "<|output|>" in answer:
            answer = answer.split("<|output|>")[-1]
        if "<|endoftext|>" in answer:
            answer = answer.split("<|endoftext|>")[0]

        total_time = (time.time() - t0) * 1000

        return CoTResult(
            thinking=thinking.strip(),
            answer=answer.strip(),
            full_output=f"Thinking:\n{thinking.strip()}\n\nAnswer:\n{answer.strip()}",
            thinking_tokens=len(thinking.split()),
            answer_tokens=len(answer.split()),
            total_time_ms=total_time,
            used_cot=True,
        )


class MultiStepCoT:
    """Break a complex task into steps, execute each one.

    Usage:
        ms = MultiStepCoT(generate_fn)
        result = ms.run("Add TypeScript types, then write a test for it",
                        input_text="function add(a, b) { return a + b }")
    """

    def __init__(self, generate_fn: Callable, max_steps: int = 5):
        self.generate_fn = generate_fn
        self.max_steps = max_steps

    def decompose(self, instruction: str) -> list:
        """Break an instruction into sub-steps."""
        # Use the model to decompose, or use simple heuristics
        steps = []

        # Split on common conjunctions
        parts = instruction.replace(" and then ", " then ").split(" then ")
        if len(parts) > 1:
            steps = [p.strip() for p in parts if p.strip()]
        else:
            # Try splitting on "and"
            parts = instruction.split(" and ")
            if len(parts) > 1 and len(parts) <= 3:
                steps = [p.strip() for p in parts if p.strip()]
            else:
                steps = [instruction]

        return steps[:self.max_steps]

    def run(self, instruction: str, input_text: str = "") -> dict:
        """Execute a multi-step task."""
        steps = self.decompose(instruction)
        results = []
        current_input = input_text

        for i, step in enumerate(steps):
            prompt = f"<|instruction|>\n{step}\n<|input|>\n{current_input}\n<|output|>\n"
            output = self.generate_fn(prompt, max_tokens=200, temperature=0.5)

            if "<|output|>" in output:
                output = output.split("<|output|>")[-1]
            if "<|endoftext|>" in output:
                output = output.split("<|endoftext|>")[0]

            results.append({
                "step": i + 1,
                "instruction": step,
                "input": current_input[:200],
                "output": output.strip(),
            })

            # Feed output as input to next step
            current_input = output.strip()

        return {
            "steps": results,
            "final_output": results[-1]["output"] if results else "",
            "n_steps": len(results),
        }
