# Instruction Tuning Data Format

How we'll structure training data to teach the model to follow instructions, not just complete code.

**Status:** Design phase — this is used after we have a larger model (100M+ from cloud training).

---

## The Problem

Our current model can do this:
```
Input:  "function Button("
Output: "{ className, ...props }: React.ComponentProps<typeof Button>) {"
```

We want it to do this:
```
Input:  "Create a toggle button component that switches between light and dark mode"
Output: "function ThemeToggle() {
          const [dark, setDark] = useState(false)
          return <button onClick={() => setDark(!dark)}>{dark ? '🌙' : '☀️'}</button>
        }"
```

---

## Data Format

Each training example is a JSON object with instruction, optional input context, and expected output:

```json
{
  "instruction": "Add TypeScript types to this React component's props",
  "input": "function UserCard({ user, onEdit }) {\n  return <div>{user.name}</div>\n}",
  "output": "interface User {\n  id: string\n  name: string\n  email: string\n}\n\ninterface UserCardProps {\n  user: User\n  onEdit: (id: string) => void\n}\n\nfunction UserCard({ user, onEdit }: UserCardProps) {\n  return <div>{user.name}</div>\n}"
}
```

### Chat format for training

The model learns a specific format:

```
<|instruction|>
Add TypeScript types to this React component's props
<|input|>
function UserCard({ user, onEdit }) {
  return <div>{user.name}</div>
}
<|output|>
interface User {
  id: string
  name: string
  email: string
}
...
<|endoftext|>
```

---

## Task Categories

### 1. Code transformation (easiest — checkable by compiler)

```json
{
  "instruction": "Convert this JavaScript function to TypeScript",
  "input": "function add(a, b) { return a + b }",
  "output": "function add(a: number, b: number): number { return a + b }"
}
```

```json
{
  "instruction": "Convert this class component to a functional component with hooks",
  "input": "class Counter extends React.Component { state = { count: 0 }; render() { return <button onClick={() => this.setState({ count: this.state.count + 1 })}>{this.state.count}</button> } }",
  "output": "function Counter() {\n  const [count, setCount] = useState(0)\n  return <button onClick={() => setCount(count + 1)}>{count}</button>\n}"
}
```

### 2. Code generation from description (medium — partially checkable)

```json
{
  "instruction": "Create a React hook that debounces a value",
  "input": "",
  "output": "function useDebounce<T>(value: T, delay: number): T {\n  const [debouncedValue, setDebouncedValue] = useState(value)\n\n  useEffect(() => {\n    const timer = setTimeout(() => setDebouncedValue(value), delay)\n    return () => clearTimeout(timer)\n  }, [value, delay])\n\n  return debouncedValue\n}"
}
```

```json
{
  "instruction": "Create a loading spinner component using Tailwind CSS",
  "input": "",
  "output": "function Spinner() {\n  return (\n    <div className=\"flex items-center justify-center\">\n      <div className=\"h-8 w-8 animate-spin rounded-full border-4 border-gray-200 border-t-blue-500\" />\n    </div>\n  )\n}"
}
```

### 3. Test generation (checkable — test runner validates)

```json
{
  "instruction": "Write a test for this function",
  "input": "function formatCurrency(amount: number): string {\n  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount)\n}",
  "output": "describe('formatCurrency', () => {\n  it('formats positive amounts', () => {\n    expect(formatCurrency(1234.56)).toBe('$1,234.56')\n  })\n\n  it('formats zero', () => {\n    expect(formatCurrency(0)).toBe('$0.00')\n  })\n\n  it('formats negative amounts', () => {\n    expect(formatCurrency(-50)).toBe('-$50.00')\n  })\n})"
}
```

### 4. Explanation (hardest — not easily checkable)

```json
{
  "instruction": "Explain what this code does",
  "input": "const memoized = useMemo(() => items.filter(i => i.active).sort((a, b) => a.name.localeCompare(b.name)), [items])",
  "output": "This filters the items array to only include active items, sorts them alphabetically by name, and memoizes the result so it only recalculates when the items array changes. Without useMemo, this filter and sort would run on every render."
}
```

### 5. Bug fix (checkable — before/after compilation)

```json
{
  "instruction": "Fix the bug in this component",
  "input": "function Timer() {\n  const [count, setCount] = useState(0)\n  useEffect(() => {\n    setInterval(() => setCount(count + 1), 1000)\n  }, [])\n  return <div>{count}</div>\n}",
  "output": "function Timer() {\n  const [count, setCount] = useState(0)\n  useEffect(() => {\n    const id = setInterval(() => setCount(c => c + 1), 1000)\n    return () => clearInterval(id)\n  }, [])\n  return <div>{count}</div>\n}"
}
```

---

## How to Generate Training Data

### Method 1: Extract from existing repos (free)

From our cloned repos, create pairs by:
- Taking a function and generating an instruction that describes it
- Taking a PR diff and creating a "make this change" instruction
- Taking a typed file and creating "add types" from the untyped version
- Taking a test file and creating "write tests for" from the source

### Method 2: Synthetic via Claude API (~$200-400)

Prompt Claude to generate instruction/code pairs:

```
Generate 10 React/TypeScript instruction-following examples.
Each should have an instruction (what to do), optional input (existing code),
and output (the resulting code). Focus on practical daily tasks.
Format as JSON array.
```

### Method 3: Self-distillation (free, after initial model works)

Use the hybrid system (model + validators) to generate pairs:
1. Generate code from an instruction using the model
2. Validate with TypeScript compiler + ESLint
3. If valid, add (instruction, code) pair to training set
4. Retrain on accumulated pairs

---

## Validation Pipeline for Instruction Data

Every generated pair runs through:

```
1. TypeScript compiler — does the output code compile?
2. ESLint — does it pass linting?
3. Format check — is the output valid code, not prose?
4. Length check — is the output reasonable length?
5. Instruction match — does the output address the instruction?
   (heuristic: check that key terms from instruction appear in output)
```

Pairs that fail validation are either repaired or discarded.

---

## Volume Targets

| Source | Pairs | Quality |
|---|---|---|
| Repo extraction (transforms) | 5,000-10,000 | High |
| Repo extraction (test generation) | 2,000-5,000 | High |
| Claude API synthetic | 5,000-10,000 | High (validated) |
| Self-distillation (later) | Ongoing | Medium-high |
| **Total for first instruction tune** | **~15,000-25,000** | |

This is enough for a LoRA fine-tune on a 1B-3B model. Full instruction tuning would want 50K-100K+ pairs.

---

## When to Use This

1. After cloud training produces a 100M+ base model
2. After the base model shows good code completion ability
3. Instruction tuning is a fine-tuning step ON TOP of the base model
4. It doesn't replace pretraining — it teaches the model a new interaction pattern
