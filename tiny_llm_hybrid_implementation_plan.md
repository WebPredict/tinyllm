# Tiny LLM From Scratch + Hybrid Memory/Reasoning Architecture

A practical implementation plan for building increasingly capable small LLM systems, starting on a laptop and scaling up to cloud training runs up to roughly 3B parameters.

The core idea is not just to train a tiny GPT clone. The more interesting path is:

```text
Small LLM
+ retrieval memory
+ structured memory
+ tools
+ validators
+ planner/executor loop
+ experiments/evals
= useful narrow-domain AI system
```

This plan is designed to help you learn the full stack while avoiding wasted compute.

---

## 0. Guiding Principles

### Start tiny, validate constantly

Do not jump straight to a 1B-3B model. First prove:

- tokenizer works
- data pipeline works
- training loop works
- checkpoint/resume works
- evals work
- hybrid modules improve results

### Prefer many small experiments over one big run

A 10M-50M model can teach you more than one expensive failed cloud run.

### Use the LLM as one component, not the whole system

For tiny models especially:

```text
LLM = language interface / planner / pattern learner
Memory = external retrieval/database
Tools = calculators, code, APIs, validators
Verifier = error reduction
Controller = loop manager
```

### Optimize for narrow domains

Good candidate domains:

- React/TypeScript frontend development
- small coding transformations
- marketing copy variants
- personal research assistant
- document Q&A over your own notes

---

# Part I — Laptop Track

## 1. Laptop Project Goals

The laptop phase is for learning and iteration.

### Realistic laptop model sizes

| Tier | Parameters | Purpose |
|---|---:|---|
| Toy | 1M-5M | Verify the pipeline |
| Tiny | 10M-30M | Learn training dynamics |
| Small | 50M-100M | Serious laptop experiments |
| Stretch | 100M-300M | Possible but slow |

### Realistic token targets

| Model size | Token target |
|---|---:|
| 5M | 25M-100M |
| 10M | 50M-200M |
| 30M | 200M-600M |
| 100M | 500M-2B |

Do not worry about Chinchilla-optimal scaling at the laptop stage. The goal is controlled experimentation.

---

## 2. Recommended Local Stack

### Core training

- Python
- PyTorch
- nanoGPT-style implementation
- Hugging Face `tokenizers` or SentencePiece
- NumPy / pandas for data prep
- SQLite for metadata and structured memory

### Optional but useful

- Chroma or FAISS for vector search
- sentence-transformers for embeddings
- llama.cpp / Ollama / LM Studio for local inference later
- Weights & Biases or simple local CSV logging
- pytest for validators and tool tests

### Philosophy

Use nanoGPT as the starting point, but gradually replace pieces so you understand them:

1. Run nanoGPT unchanged.
2. Replace the dataset loader.
3. Replace tokenizer.
4. Modify architecture.
5. Add evals.
6. Add hybrid memory/tools.
7. Add instruction tuning.

---

# Part II — From-Scratch GPT Implementation Roadmap

## 3. Stage 1: Minimal Character-Level GPT

### Goal

Train the smallest possible model that learns to generate coherent-ish text.

### Dataset

Use one of:

- Shakespeare text
- TinyStories sample
- small public-domain book
- your own curated text file

### Architecture

- character tokenizer
- context length: 128-256
- layers: 2-4
- heads: 2-4
- embedding dim: 64-256
- parameters: 1M-5M

### Checkpoints

Save:

- model weights every N steps
- validation loss
- generated sample text
- training config JSON

### Validation gates

Continue only if:

- training loss falls smoothly
- validation loss falls initially
- generation improves visibly
- checkpoint resume works

### Experiments

| Experiment | Compare |
|---|---|
| Context length | 128 vs 256 vs 512 |
| Model size | 1M vs 5M |
| Learning rate | 1e-3 vs 3e-4 vs 1e-4 |
| Dropout | 0 vs 0.1 |

---

## 4. Stage 2: Token-Level GPT

### Goal

Move from characters to real tokenization.

### Tokenizer options

- BPE tokenizer
- SentencePiece unigram
- WordPiece-style tokenizer

### Recommended vocab experiments

| Vocab size | Use case |
|---|---|
| 4k | tiny models, faster softmax |
| 8k | good laptop default |
| 16k | stronger text compression |
| 32k | closer to normal LLMs but heavier |

### Measure tokenizer quality

Track:

- average tokens per word
- average tokens per character
- weird splits
- compression ratio
- validation loss with same model size

### Suggested datasets

- TinyStories
- public-domain prose
- small Wikipedia slices
- React/TypeScript code and documentation

### Validation gates

Continue only if:

- tokenization is sane
- unknown/weird tokens are rare
- validation loss improves relative to character model
- generated text is more coherent

---

## 5. Stage 3: Tiny GPT Training Runs

### Goal

Train a real small transformer and understand scaling.

### Suggested model configs

#### 10M model

```yaml
n_layer: 6
n_head: 6
n_embd: 384
block_size: 256-512
vocab_size: 8000-16000
```

#### 30M model

```yaml
n_layer: 8
n_head: 8
n_embd: 512
block_size: 512-1024
vocab_size: 8000-16000
```

#### 50M-100M model

```yaml
n_layer: 10-12
n_head: 8-12
n_embd: 640-768
block_size: 1024
vocab_size: 16000-32000
```

### Core logging

Log every run with:

```text
run_id
model_config
dataset_mix
tokenizer_version
optimizer_config
learning_rate
batch_size
grad_accum_steps
train_loss
val_loss
sample_outputs
wall_clock_time
hardware
```

### Validation gates

Continue only if:

- loss curve is smooth
- no severe repetition collapse
- samples improve with training
- validation loss is not diverging from training loss badly

---

# Part III — Data Pipeline and Quality

## 6. Initial Raw Data Sources

Start with curated/open datasets rather than scraping the raw web yourself.

### General-purpose educational/prose datasets

- TinyStories
- FineWeb-Edu subsets
- Dolma subsets
- RedPajama subsets
- Wikipedia dumps
- Project Gutenberg/public-domain books
- Stack Exchange-style Q&A datasets

### Domain-specific datasets

For React/TypeScript frontend development:

- open-source React component libraries (e.g., Radix, shadcn/ui, Chakra UI source code)
- TypeScript DefinitelyTyped type definitions
- React and Next.js documentation
- MDN Web Docs (HTML, CSS, DOM APIs)
- Stack Overflow React/TypeScript Q&A threads
- permissively licensed GitHub repos using React + TypeScript
- before/after code diffs from PRs in popular React projects
- ESLint rule documentation and examples
- testing library documentation (Jest, React Testing Library, Vitest)

---

## 7. Data Cleaning Pipeline

### Basic filtering

Remove:

- navigation menus
- cookie banners
- boilerplate
- duplicated lines
- pages with too many symbols
- malformed text
- extremely short documents
- extremely long low-quality documents
- non-target languages

### Deduplication

Use multiple levels:

1. Exact document dedupe
2. Exact line dedupe
3. Near-duplicate document detection with MinHash/SimHash
4. N-gram duplicate filtering

### Quality scoring

Score each document using features like:

```text
language confidence
word count
sentence length distribution
punctuation ratio
duplicate n-gram ratio
perplexity under small reference model
educational density
HTML/boilerplate ratio
```

Then keep only documents above a quality threshold.

### Train/validation split

Important:

- split by document, not by random token span
- dedupe before splitting
- keep validation data pristine
- never train on your eval questions

---

## 8. Data Mix Experiments

Train identical small models on different mixes.

### Experiment A: Raw vs filtered

```text
Model A: raw data
Model B: filtered data
```

Expected result:

Filtered data usually wins despite being smaller.

### Experiment B: Dataset composition

```text
Model A: 100% TinyStories
Model B: 70% TinyStories + 30% public-domain books
Model C: 50% educational + 30% prose + 20% Q&A
```

### Experiment C: Domain mix

For React/TypeScript:

```text
Model A: general code only
Model B: 70% general code + 30% React/TypeScript
Model C: 30% general code + 50% React/TypeScript + 20% React docs/tutorials
```

Track:

- validation loss
- domain eval accuracy
- output quality
- repetition
- hallucination/error rate

---

# Part IV — Synthetic Data

## 9. Synthetic Data Strategy

Synthetic data is most useful when outputs are structured and checkable.

### Good synthetic data tasks

- JavaScript to TypeScript conversion
- add types to untyped React components
- generate unit tests for components and hooks
- refactor class components to functional components with hooks
- add error handling and loading states
- component prop type extraction
- coding refactors
- ESLint rule violation detection and fix
- classification examples

### Bad synthetic data tasks

Avoid using synthetic data for:

- fake factual articles
- unverifiable financial/legal/medical claims
- long reasoning traces with no checker
- large amounts of generic AI slop

---

## 10. Synthetic Data Generation Loop

Use this loop:

```text
Generate examples
↓
Filter invalid/low-quality examples
↓
Validate with rules or stronger model
↓
Train/fine-tune
↓
Evaluate weak spots
↓
Generate more examples targeting weaknesses
```

### Example: TypeScript type error fix dataset

```json
{
  "task": "fix_typescript_error",
  "input": "const user = { name: 'Jeff', age: 30 };\nconst greeting: string = user;",
  "output": "const user = { name: 'Jeff', age: 30 };\nconst greeting: string = user.name;",
  "error_type": "type_assignment",
  "explanation": "Cannot assign type '{ name: string; age: number }' to type 'string'. Access the 'name' property to get the string value."
}
```

### Example: React component refactor generation

```json
{
  "task": "refactor_class_to_hooks",
  "difficulty": "intermediate",
  "target_patterns": ["useState", "useEffect", "cleanup"],
  "input": "class Timer extends React.Component { state = { count: 0 }; componentDidMount() { this.interval = setInterval(() => this.setState({ count: this.state.count + 1 }), 1000); } componentWillUnmount() { clearInterval(this.interval); } render() { return <div>{this.state.count}</div>; } }",
  "output": "function Timer() { const [count, setCount] = useState(0); useEffect(() => { const interval = setInterval(() => setCount(c => c + 1), 1000); return () => clearInterval(interval); }, []); return <div>{count}</div>; }"
}
```

### Example: coding transformation

```json
{
  "task": "convert_js_to_ts",
  "input": "function add(a, b) { return a + b }",
  "output": "function add(a: number, b: number): number { return a + b }"
}
```

---

# Part V — Hybrid Architecture Ideas

## 11. Hybrid System Overview

The core architecture:

```text
User Input
↓
Intent Classifier
↓
Memory/Retrieval Decision
↓
Retrieve relevant context
↓
Tiny LLM drafts plan or answer
↓
Tool/validator/verifier runs
↓
Tiny LLM finalizes response
↓
Memory manager updates state
```

The goal is to make a small model feel smarter by giving it external capabilities.

---

## 12. Hybrid Module 1: RAG Memory

### Purpose

Give the model access to knowledge not stored in weights.

### Components

- document store
- chunker
- embedding model
- vector index
- retriever
- reranker optional
- prompt composer

### Basic flow

```text
query
→ embed query
→ search top-k chunks
→ insert chunks into prompt
→ model answers
```

### Laptop implementation

Start simple:

- Chroma or FAISS
- local embedding model
- markdown/text document corpus
- top-k retrieval

### Experiments

| Experiment | Compare |
|---|---|
| No RAG vs RAG | Does retrieval improve factual accuracy? |
| chunk size | 256 vs 512 vs 1024 tokens |
| top-k | 3 vs 5 vs 10 chunks |
| embeddings | small local vs API embeddings |
| reranking | vector only vs reranked |

### Eval ideas

Create 100 Q&A pairs from your documents and measure:

- answer correctness
- citation correctness
- unsupported claims
- answer completeness

---

## 13. Hybrid Module 2: Structured Long-Term Memory

### Purpose

Track durable state over time.

Unlike RAG, which retrieves document chunks, structured memory stores facts/records.

### Example: React/TypeScript developer memory

```json
{
  "known_patterns": ["useState", "useEffect", "useContext", "custom hooks", "React.memo"],
  "weak_areas": ["TypeScript generics", "complex state management", "performance optimization"],
  "recent_errors": [
    {
      "code": "const [items, setItems] = useState([])",
      "error_type": "missing_type_parameter",
      "fix": "useState<Item[]>([])",
      "date": "2026-05-10"
    },
    {
      "code": "useEffect(() => { fetchData() }, [])",
      "error_type": "missing_cleanup",
      "fix": "added AbortController cleanup",
      "date": "2026-05-12"
    }
  ],
  "current_skill_level": "intermediate",
  "preferred_stack": ["React 18", "TypeScript 5", "Tailwind", "Vitest"]
}
```

### Implementation

Use SQLite tables:

```sql
projects
patterns
errors
sessions
code_snippets
memory_events
```

### Memory manager responsibilities

After each interaction:

1. extract durable facts
2. update skill profile
3. merge duplicates
4. avoid storing junk
5. retrieve relevant state for future prompts

### Experiments

| Experiment | Compare |
|---|---|
| No memory vs memory | Does personalization improve? |
| raw history vs structured memory | Which uses fewer tokens? |
| automatic updates vs manual updates | Which is more accurate? |

---

## 14. Hybrid Module 3: Rules Engine

### Purpose

Use deterministic software to check things the LLM gets wrong.

### Good domains

- TypeScript types
- React patterns
- arithmetic
- formatting
- JSON validity
- code style
- unit tests

### Example flow

```text
LLM proposes code transformation
↓
TypeScript compiler validates types
↓
ESLint checks code quality
↓
test runner verifies behavior
↓
if invalid, reject or repair
↓
LLM explains final result
```

### Example React/TypeScript validators

- TypeScript compiles without errors
- ESLint passes with project config
- component has proper prop type definitions
- hooks follow the Rules of Hooks (no conditional hooks, top-level only)
- no inline styles when Tailwind/CSS modules are used
- useEffect has correct dependency array
- event handlers are properly typed
- components export correct types for consumers

### Experiments

| Experiment | Metric |
|---|---|
| LLM-only code generation | error rate |
| LLM + validator | error rate |
| LLM + validator + repair | final accepted rate |

---

## 15. Hybrid Module 4: Calculator / Code Interpreter

### Purpose

Offload exact reasoning.

### Flow

```text
question
→ LLM writes calculation/code plan
→ Python executes
→ result returned
→ LLM explains result
```

### Use cases

- arithmetic
- financial calculations
- component scaffolding
- data analysis
- plotting
- simple simulations

### Safety

If executing arbitrary code:

- use sandboxing
- timeout limits
- restricted imports
- no filesystem/network access unless explicitly allowed

### Experiments

Compare:

- LLM-only math answers
- LLM + Python tool
- LLM + Python + verifier

---

## 16. Hybrid Module 5: Planner/Executor Agent Loop

### Purpose

Let the system solve tasks in steps.

### Minimal loop

```python
while not done:
    state = observe()
    action = llm_decide_next_action(state)
    result = run_action(action)
    update_state(result)
```

### Actions

Start with a tiny action set:

```json
[
  "search_memory",
  "retrieve_docs",
  "run_python",
  "validate_output",
  "draft_answer",
  "final_answer"
]
```

### Important rule

Constrain actions. Do not let the tiny model free-form everything.

### Experiments

| Experiment | Compare |
|---|---|
| one-shot answer | baseline |
| plan-then-answer | better structure? |
| plan-tool-answer | better accuracy? |
| plan-tool-verify-answer | lowest error? |

---

## 17. Hybrid Module 6: Verifier/Critic

### Purpose

Catch mistakes before final output.

### Verifier types

- rule-based verifier
- unit tests
- second small model
- stronger API model occasionally
- exact answer checker
- schema validator

### Flow

```text
draft answer
↓
verifier checks
↓
feedback generated
↓
LLM revises
↓
final answer
```

### Experiments

Track:

- initial error rate
- final error rate
- revision success rate
- false rejection rate
- extra latency

---

## 18. Hybrid Module 7: Symbolic Task Graphs

### Purpose

Avoid asking the LLM to improvise everything.

Instead, ask it to fill structured slots.

### Example

```json
{
  "goal": "generate_react_component",
  "component_type": "form",
  "features": ["controlled inputs", "validation", "error display"],
  "typescript": true,
  "styling": "tailwind",
  "tests": true
}
```

Then deterministic software generates/validates the component scaffold.

### Why this helps

Small models are much better at constrained structured outputs than open-ended reasoning.

### Experiments

Compare:

- freeform prompt
- JSON schema prompt
- JSON schema + validator
- JSON schema + validator + repair loop

---

## 19. Hybrid Module 8: Chain-of-Thought / Scratchpad

### Purpose

Let the model reason explicitly before answering, improving accuracy on multi-step tasks.

### Flow

```text
question
→ model writes reasoning in scratchpad
→ model produces final answer
→ scratchpad can be hidden from user or shown
```

### Implementation

- Format training data with explicit reasoning traces before answers
- Use a `<think>...</think>` or `[scratchpad]...[/scratchpad]` delimiter
- At inference, optionally strip the reasoning from user-facing output

### When it helps most

- Debugging TypeScript errors ("why doesn't this type-check?")
- Multi-step tool selection
- Deciding which memory to retrieve
- Any task where the answer depends on intermediate logic

### Experiments

| Experiment | Compare |
|---|---|
| Direct answer | baseline |
| Chain-of-thought | accuracy improvement? |
| CoT + verifier | does reasoning help verifier catch more? |
| CoT hidden vs shown | does showing reasoning help users? |

### Cost

Zero additional training cost — just a data formatting choice. Moderate inference cost (more output tokens).

---

## 20. Hybrid Module 9: Self-Consistency (Sample + Vote)

### Purpose

Generate multiple candidate outputs and select the best one, improving reliability.

### Flow

```text
question
→ generate N candidate answers (3-5)
→ vote: majority answer, or pick the one passing validation
→ return winner
```

### Selection strategies

- Majority vote (for classification/short-answer tasks)
- Validator pass rate (pick the one that passes all checks)
- Confidence score ranking
- Hybrid: majority among validator-passing candidates

### When it helps most

- Code refactoring (multiple valid refactors, pick most common)
- Component generation (generate 5, keep the one that validates)
- Any task with a checkable answer

### Experiments

| Experiment | Compare |
|---|---|
| N=1 (baseline) | single generation |
| N=3 + majority | accuracy improvement? |
| N=5 + validator filter | error rate reduction? |
| Cost vs accuracy tradeoff | diminishing returns at N=? |

### Cost

Zero training cost. Inference cost is Nx at generation time — but since you own the model, this is just wall-clock time, not dollars.

---

## 21. Hybrid Module 10: Constrained Decoding

### Purpose

Force the model to only produce syntactically valid outputs at decode time, eliminating format errors entirely.

### Implementation

Use libraries like `outlines`, `guidance`, or custom logit masking to constrain the token-by-token generation to follow a grammar/schema.

### Use cases

- Tool-call JSON is always valid JSON
- SQL queries are always syntactically correct
- Generated components always match expected schema
- Enum fields only contain allowed values

### Example

```python
# Instead of hoping the model outputs valid JSON:
schema = {
    "tool": ["search_docs", "run_python", "validate_typescript"],
    "args": {"expression": str}
}
output = constrained_generate(model, schema)
# Output is guaranteed to match schema
```

### Experiments

| Experiment | Compare |
|---|---|
| Unconstrained generation | format error rate |
| Constrained decoding | should be ~0% format errors |
| Constrained + validator | belt and suspenders |

### Cost

Zero training cost. Negligible inference overhead (logit masking is cheap).

---

## 22. Hybrid Module 11: Confidence Estimation and Abstention

### Purpose

Teach the model to recognize when it doesn't know something, and route uncertain queries to fallback systems.

### Implementation

Options:

1. **Fine-tune with confidence labels**: train on examples with explicit confidence scores
2. **Logit-based**: use token probabilities as a proxy for confidence
3. **Consistency-based**: if self-consistency samples disagree, confidence is low

### Routing rules

```text
High confidence → answer directly
Medium confidence → use RAG to verify
Low confidence → "I'm not sure" + suggest resources
```

### Why this matters for small models

Small models hallucinate more. Knowing *when* to abstain is often more valuable than being right 80% of the time — a system that's right 70% but never confidently wrong is more trustworthy than one that's right 85% but occasionally makes confident errors.

### Experiments

| Experiment | Compare |
|---|---|
| Always answer | baseline accuracy + hallucination rate |
| Confidence threshold | accuracy among answered queries |
| Confidence + routing | does RAG fallback recover uncertain cases? |

### Cost

Minimal fine-tuning cost for confidence labels. Zero inference overhead for logit-based approaches.

---

## 23. Hybrid Module 12: LoRA Adapters Per Sub-Task

### Purpose

Train one base model, then lightweight adapters for each domain or task type. Swap at inference based on intent.

### Architecture

```text
Base model (3B, frozen or shared)
├── LoRA: React component patterns
├── LoRA: TypeScript type transforms
├── LoRA: test generation
├── LoRA: document Q&A
└── LoRA: code refactoring
```

### Why this works

- LoRA adapters are tiny (often <1% of base model parameters)
- Training a LoRA takes minutes to hours, not days
- Each adapter can be trained on domain-specific data without catastrophic forgetting
- Swap adapters at runtime with near-zero overhead

### Flow

```text
user input
→ intent classifier picks domain
→ load appropriate LoRA
→ generate with base + adapter
```

### Experiments

| Experiment | Compare |
|---|---|
| Single model, all domains | baseline |
| Base + domain LoRA | domain accuracy improvement? |
| Multiple LoRAs vs one fine-tune | specialization vs generality |

### Cost

Very cheap — each LoRA trains in a fraction of the time/cost of full fine-tuning. Can be done on laptop for small adapters.

---

## 24. Hybrid Module 13: Knowledge Distillation Framing

### Purpose

Intentionally teach your small model to mimic *how* a frontier model approaches problems, not just *what* it outputs.

### Distinction from basic synthetic data

Basic synthetic data:
```json
{"input": "question", "output": "answer"}
```

Distillation-framed:
```json
{
  "input": "question",
  "reasoning": "The model should first identify the TypeScript error, then check the type hierarchy...",
  "output": "answer",
  "decision_process": "Used a generic constraint because the function needs to work with multiple types..."
}
```

### What to distill

- Reasoning traces for domain tasks
- Tool selection rationale
- When-to-abstain decisions
- Error diagnosis patterns
- Multi-step planning strategies

### Implementation

1. Send domain tasks to Claude/GPT-4 with "explain your reasoning"
2. Filter for correct + well-reasoned outputs
3. Format as chain-of-thought training data
4. Fine-tune your small model on these traces

### Cost

API cost for generation (~$50-200 depending on volume). Training cost is part of your existing fine-tuning budget.

---

## 25. Hybrid Module 14: Routing and Cascading

### Purpose

Use the cheapest/fastest path that can handle each query, reserving expensive multi-step pipelines for hard cases.

### Tiers

```text
Tier 0: Cache hit → instant response
Tier 1: Direct generation → simple queries
Tier 2: RAG-augmented → needs external knowledge
Tier 3: Full pipeline (planner + tools + verifier) → complex tasks
```

### Router implementation

Options:

- Tiny classifier model (even logistic regression on query features)
- Keyword/rule-based routing
- The model's own confidence on first attempt
- Query complexity heuristics (length, question type, domain)

### Why this matters

Most queries in a narrow domain are easy. If 60% of queries can be answered directly and 30% need RAG, only 10% need the full expensive loop. This makes the system feel fast for common cases.

### Experiments

| Experiment | Compare |
|---|---|
| Always full pipeline | latency + accuracy baseline |
| Routed by complexity | latency improvement, accuracy maintained? |
| Cache + routing | response time for repeated patterns |

### Cost

Negligible. The router itself is trivial. This actually *saves* compute overall.

---

## 26. Hybrid Module 15: Caching and Memoization

### Purpose

Store previously computed results to avoid redundant work and improve response time.

### What to cache

- Common query → answer pairs
- Retrieval results for frequent queries
- Validated code transformation templates
- Embedding computations for static documents

### Implementation

```text
SQLite cache table:
  query_hash | query_text | response | timestamp | hit_count

On each query:
  1. hash query
  2. check cache (with optional fuzzy matching)
  3. if hit and fresh → return cached
  4. if miss → run pipeline → cache result
```

### Cache invalidation

- TTL-based for time-sensitive content
- Invalidate when underlying documents change
- Never cache low-confidence responses
- Version cache entries by model version

### Cost

Zero. Actually reduces compute. SQLite is free.

---

## 27. Hybrid Module 16: DPO (Direct Preference Optimization)

### Purpose

Align the model's outputs with quality preferences beyond instruction following — teach it to prefer *better* answers over merely *acceptable* ones.

### How it works

1. Generate pairs: (better response, worse response) for the same input
2. Train the model to prefer the better one using the DPO loss
3. No reward model needed (unlike RLHF)

### Generating preference pairs

For React/TypeScript:
```json
{
  "input": "Fix the type error: const [count, setCount] = useState(0); setCount('hello');",
  "chosen": "The error is that setCount expects a number but receives a string. TypeScript inferred useState<number> from the initial value 0. Either change the call to setCount(5) or widen the type: useState<number | string>(0).",
  "rejected": "Change useState(0) to useState('hello'). That fixes the error."
}
```

The "chosen" explains the root cause and offers appropriate solutions. The "rejected" technically removes the error but changes the wrong thing.

### Sources of pairs

- Validator-passing vs validator-failing outputs
- Detailed explanation vs terse explanation
- Correctly-cited vs uncited answers
- Idiomatic code vs technically-correct-but-messy code

### Experiments

| Experiment | Compare |
|---|---|
| Instruction tuning only | baseline quality |
| Instruction tuning + DPO | explanation quality, user preference |
| DPO on different pair sources | which signal matters most? |

### Cost

One extra training pass over preference data. Comparable cost to instruction tuning — maybe $100-500 at the 3B scale depending on dataset size.

---

## 28. Hybrid Module 17: Query Rewriting for Retrieval

### Purpose

Improve RAG recall by having the model rephrase the user's query before searching.

### Why it helps

Users ask questions casually. The retrieval index contains formal text. A rewrite bridges the gap.

```text
User: "when do I use useMemo vs useCallback"
Rewritten: "React useMemo vs useCallback hooks differences, memoization usage rules"
→ much better retrieval results
```

### Flow

```text
user query
→ LLM rewrites query for retrieval (one cheap inference call)
→ embed rewritten query
→ retrieve top-k
→ LLM answers using retrieved context
```

### Variants

- Single rewrite
- Multiple rewrites → retrieve for each → merge results (HyDE-style)
- Hypothetical answer generation → embed the answer → retrieve similar (full HyDE)

### Experiments

| Experiment | Compare |
|---|---|
| Raw query retrieval | baseline recall |
| Rewritten query | recall improvement? |
| Multi-rewrite + merge | diminishing returns? |
| HyDE-style | worth the extra inference? |

### Cost

One extra inference call per query. Since you own the model, this is essentially free — just adds ~100ms latency.

---

# Part VI — Evaluation Framework

## 29. Core Eval Types

### Language modeling evals

- training loss
- validation loss
- perplexity
- repetition rate

### Instruction evals

- follows format
- answers requested task
- refuses malformed task when needed
- uses retrieved context properly

### Domain evals

For React/TypeScript:

- TypeScript compiles without errors
- unit tests pass
- ESLint passes
- component renders correctly
- output diff matches expected transformation
- prop types are properly defined
- hooks follow Rules of Hooks

For RAG:

- answer uses retrieved docs
- answer cites correct chunk
- no unsupported claims

### Agent evals

- task completion rate
- number of tool calls
- invalid tool calls
- loop failures
- final accuracy

---

## 30. Create Fixed Eval Sets Early

Before scaling, create small eval sets.

### Suggested eval sets

| Eval set | Size | Purpose |
|---|---:|---|
| general generation prompts | 50 | quality vibes |
| instruction following | 100 | post-training |
| RAG Q&A | 100 | retrieval accuracy |
| React/TypeScript code | 200 | domain correctness |
| tool-use tasks | 100 | reasoning/tool loop |
| adversarial/error cases | 50 | robustness |

Keep these fixed so you can compare model versions honestly.

---

# Part VII — Laptop Milestones and Gates

## 31. Milestone 1: Pipeline Works

### Build

- character GPT
- tiny dataset
- training loop
- checkpointing
- sampling

### Gate

Continue if:

- loss decreases
- samples improve
- checkpoint resume works

---

## 32. Milestone 2: Tokenizer Works

### Build

- train BPE/SentencePiece tokenizer
- tokenized dataset pipeline
- token-level GPT

### Gate

Continue if:

- tokenization is sane
- validation loss improves
- generation quality improves

---

## 33. Milestone 3: Data Quality Matters

### Build

- raw dataset
- filtered dataset
- deduped dataset

### Experiment

Train same model on each.

### Gate

Continue if:

- filtered/deduped data improves validation or output quality
- you can explain which data choices helped

---

## 34. Milestone 4: Synthetic Data Helps

### Build

- generate synthetic task examples
- filter/validate them
- fine-tune model

### Gate

Continue if:

- domain eval improves
- format following improves
- hallucinations do not increase too much

---

## 35. Milestone 5: RAG Improves Answers

### Build

- document index
- retrieval pipeline
- prompt composer

### Gate

Continue if:

- RAG answers beat no-RAG answers on fixed Q&A eval
- retrieved chunks are actually relevant

---

## 36. Milestone 6: Structured Memory Improves Personalization

### Build

- SQLite memory tables
- memory retrieval
- memory update step

### Gate

Continue if:

- model adapts to user/session state
- fewer repeated mistakes
- prompt stays compact

---

## 37. Milestone 7: Tools Improve Reasoning

### Build

- Python calculator tool
- validator tool
- JSON tool-call schema

### Gate

Continue if:

- tool-assisted results beat LLM-only results
- invalid tool calls are rare
- verifier reduces error rate

---

# Part VIII — Cloud Track Up to 3B Parameters

## 38. Cloud Strategy

Only move to cloud after the laptop pipeline is proven.

### Cloud goals

- scale pretraining
- test data mixes at 100M-300M
- train a 1B-3B model if pipeline is stable
- quantize and run locally afterward

### Recommended cloud provider style

- cheap GPU marketplace for pilots
- more stable GPU cloud for final runs
- checkpoint frequently
- assume interruptions

---

## 39. Cloud Model Size Plan

### 100M pilot

Purpose:

- test cloud infra
- validate distributed/faster training setup
- run data ablations cheaply

Token target:

- 1B-2B tokens

### 300M serious small model

Purpose:

- strong enough to show meaningful behaviors
- good for data quality experiments

Token target:

- 3B-8B tokens

### 1B main model

Purpose:

- first real small LLM
- useful for instruction/domain tuning

Token target:

- 10B-25B tokens

### 3B stretch model

Purpose:

- maximum sensible hobby/cloud target
- should be attempted only after smaller runs are successful

Token target:

- 30B-60B tokens, possibly more if budget allows

Compute-optimal would be roughly 60B tokens for 3B parameters, using the 20 tokens/parameter rule of thumb.

---

## 40. Cloud Checkpoint Gates

### Gate A: 100M pilot

Continue if:

- training is stable
- throughput is acceptable
- checkpoint/resume works
- validation loss is sane

### Gate B: 300M data ablation

Continue if:

- you have selected a winning data mix
- synthetic data ratio is validated
- tokenizer choice is validated

### Gate C: 1B run

Continue if:

- evals improve predictably with scale
- no collapse/repetition
- generated samples justify further spend

### Gate D: 3B run

Attempt only if:

- 1B model is solid
- data pipeline is final
- eval harness is stable
- cloud training cost estimate is acceptable
- checkpointing is robust

---

## 41. Suggested 3B Architecture

A plausible dense transformer config:

```yaml
n_layer: 28-32
n_head: 24-32
n_embd: 2560-3072
block_size: 4096-8192
vocab_size: 32000-50000
parameters: ~3B
```

### Context recommendation

- 4k context: safer
- 8k context: good target but more expensive
- 16k context: not recommended for first serious run

### Inference target

After training:

- export to HF format
- quantize to 4-bit GGUF
- run locally via llama.cpp/Ollama/LM Studio

Approximate weight sizes:

| Format | 3B model size |
|---|---:|
| FP32 | ~12 GB |
| FP16/BF16 | ~6 GB |
| 8-bit | ~3 GB |
| 4-bit | ~1.5-2 GB |

---

# Part IX — Post-Training

## 42. Instruction Tuning

After base pretraining, instruction tune on structured examples.

### Data types

- general instruction-following
- domain instructions
- formatting tasks
- tool-use examples
- RAG answer examples
- refusal/safety basics

### Example format

```json
{
  "instruction": "Add TypeScript types to this React component's props.",
  "input": "function UserCard({ name, age, onClick }) { return <div onClick={onClick}><h2>{name}</h2><p>{age} years old</p></div> }",
  "output": "interface UserCardProps {\n  name: string;\n  age: number;\n  onClick: () => void;\n}\n\nfunction UserCard({ name, age, onClick }: UserCardProps) { return <div onClick={onClick}><h2>{name}</h2><p>{age} years old</p></div> }"
}
```

### Gate

Instruction tuning succeeds if:

- format following improves
- domain eval improves
- general fluency does not collapse

---

## 43. Tool-Use Fine-Tuning

Train the model to emit structured tool calls.

### Example

```json
{
  "user": "What is 17 * 43?",
  "assistant_tool_call": {
    "tool": "calculator",
    "args": {"expression": "17 * 43"}
  },
  "tool_result": "731",
  "assistant_final": "17 × 43 = 731."
}
```

### Useful tool schemas

- `search_docs`
- `read_memory`
- `write_memory`
- `run_python`
- `validate_typescript`
- `run_eslint`
- `final_answer`

### Gate

Tool tuning succeeds if:

- valid tool-call JSON rate is high
- correct tool selection improves
- final answer accuracy improves

---

## 44. Domain Fine-Tuning

For a React/TypeScript development assistant:

Fine-tune on:

- TypeScript error diagnosis and fixes
- component generation from descriptions
- code refactoring (class to hooks, JS to TS)
- explanation of type errors
- test generation for components and hooks
- performance optimization suggestions

### Gate

Domain fine-tune succeeds if:

- domain eval improves materially
- outputs remain natural
- model does not overfit to templates

---

# Part X — Example End-State Systems

## 45. React/TypeScript Development Assistant

```text
User submits code or describes task
↓
structured memory retrieves known patterns and weak areas
↓
LLM classifies task type (type error, refactor, generate, explain)
↓
TypeScript compiler validates types
↓
ESLint checks code quality
↓
RAG retrieves relevant docs/patterns
↓
LLM proposes solution with explanation
↓
test runner verifies behavior
↓
memory updates developer profile
↓
follow-up suggestions generated
```

### Why this is good for small models

- narrow domain (React + TypeScript ecosystem)
- structured outputs (code is parseable)
- rich validator ecosystem (TypeScript compiler, ESLint, Prettier, test runners, build tools)
- memory matters (track developer patterns, recurring errors, project context)
- correctness is checkable (does it compile? do tests pass? does it render?)

---

## 46. Local Research Assistant

```text
Question
↓
RAG retrieves notes/papers
↓
LLM summarizes
↓
critic checks citations/support
↓
final answer with sources
```

### Experiments

- retrieval-only vs retrieval + reranking
- answer-only vs answer + citation checker
- one-shot vs map-reduce summarization

---

## 47. Coding Transformation Assistant

```text
Code input
↓
LLM proposes transformation
↓
formatter/typechecker/tests run
↓
LLM patches errors
↓
final diff returned
```

### Good tasks

- JS to TS conversion
- add tests
- refactor function
- explain diff
- update CSS classes

---

# Part XI — Practical Run Order

## 48. Recommended Sequence

### Week 1: Basics

- run nanoGPT-style char model
- generate text
- log loss
- implement checkpoint resume

### Week 2: Tokenization and data

- train tokenizer
- build filtered dataset
- train 10M-30M token model
- compare tokenizer sizes

### Week 3: Evals

- create fixed eval sets
- automate sample generation
- build scoring scripts

### Week 4: Synthetic data

- generate structured examples
- validate/filter
- fine-tune tiny model
- measure gains

### Week 5: RAG

- build vector index
- compare no-RAG vs RAG
- build citation/support eval

### Week 6: Structured memory

- SQLite memory
- update/retrieve memory
- test personalization

### Week 7: Tools and validators

- Python tool
- React/TypeScript validators
- JSON tool-call schema
- verifier loop

### Week 8: Cloud pilot

- 100M model
- 1B-2B tokens
- test throughput/checkpointing

### Week 9-10: Cloud ablations

- 300M models
- data mix comparison
- tokenizer comparison
- synthetic ratio comparison

### Week 11+: Main model

- 1B main run
- optional 3B stretch
- instruction tune
- tool-use tune
- quantize and run locally

---

# Part XII — Anti-Waste Checklist

Before any expensive cloud run, confirm:

- [ ] tokenizer finalized
- [ ] data cleaned/deduped
- [ ] validation set fixed
- [ ] eval harness works
- [ ] checkpoint resume tested
- [ ] throughput benchmarked
- [ ] sample generation automated
- [ ] learning rate tested on smaller model
- [ ] data mix tested on smaller model
- [ ] cloud storage and download path tested
- [ ] failure/interruption recovery tested

---

# Part XII.5 — Code Security and Safety

## Security Scope For This Project

Commercial LLMs spend enormous resources on safety: RLHF for refusals, content filtering, prompt injection defense, jailbreak resistance, bias mitigation, multilingual safety. Almost none of this applies here.

Why: those systems serve millions of anonymous users asking anything. This is a private tool for a known user (you) generating code in a narrow domain (React/TypeScript). The attack surface is fundamentally different.

What does matter: the model should not generate insecure code, leak secrets from training data, or suggest malicious dependencies.

---

## Security Layer 1: Training Data Filtering

### Remove secrets from training data

Before training, filter out files and content that contain credentials:

```text
Remove files matching:
  - .env, .env.local, .env.production
  - credentials.json, secrets.yaml, *_key.json
  - Any file containing patterns:
    - sk-[a-zA-Z0-9]{20,}           (API keys)
    - AKIA[A-Z0-9]{16}              (AWS access keys)
    - ghp_[a-zA-Z0-9]{36}           (GitHub tokens)
    - -----BEGIN.*PRIVATE KEY-----  (private keys)
    - password\s*=\s*["'][^"']+     (hardcoded passwords)
```

### Downweight insecure code patterns

In quality scoring, penalize files containing:

```text
- eval() with user input
- innerHTML assignment from variables
- SQL string concatenation (if any backend code in corpus)
- disabled security headers
- http:// URLs for API calls
- console.log of sensitive-looking variables
- dangerouslySetInnerHTML with unsanitized input
```

Don't remove these files entirely — the model needs to recognize bad patterns to avoid them. But downweight them so the model's default behavior leans toward secure code.

---

## Security Layer 2: Output Validators

Add to the existing validator pipeline:

### Secret detection in generated output

```text
Post-generation regex check:
  - Matches any API key format → block output, warn
  - Matches any private key format → block output, warn
  - Contains hardcoded credentials → block output, warn
```

This catches cases where the model memorized and regurgitates secrets from training data.

### ESLint security plugin

Add `eslint-plugin-security` and/or `eslint-plugin-no-unsanitized` to the validator chain:

```text
Generated code
  → TypeScript compiler (type safety)
  → ESLint standard rules (code quality)
  → ESLint security plugin (vulnerability detection)
  → Test runner (behavioral correctness)
```

Catches: prototype pollution, unsafe regex, command injection via child_process, path traversal, etc.

### Package name verification

When the model suggests installing a package:

```text
Suggested package name
  → Check npm registry: does this package exist?
  → Check download count: is it widely used?
  → Check against known-malicious package list
  → Flag unknown/low-download packages for human review
```

This prevents supply chain attacks via typosquatted package names (e.g., model suggests `react-querry` instead of `react-query`).

### dangerouslySetInnerHTML checker

Specific to React: if generated code uses `dangerouslySetInnerHTML`, verify the input is sanitized (DOMPurify or similar). Flag if raw variables are passed in.

---

## Security Layer 3: DPO Pairs for Secure Code Preference

Create ~200-500 preference pairs teaching the model to prefer secure patterns:

### Example pairs

```json
{
  "input": "Create a component that renders user-provided HTML content",
  "chosen": "import DOMPurify from 'dompurify';\n\nfunction UserContent({ html }: { html: string }) {\n  const clean = DOMPurify.sanitize(html);\n  return <div dangerouslySetInnerHTML={{ __html: clean }} />;\n}",
  "rejected": "function UserContent({ html }: { html: string }) {\n  return <div dangerouslySetInnerHTML={{ __html: html }} />;\n}"
}
```

```json
{
  "input": "Fetch data from an API endpoint with an auth token",
  "chosen": "const token = process.env.NEXT_PUBLIC_API_TOKEN;\n\nconst res = await fetch(url, {\n  headers: { Authorization: `Bearer ${token}` },\n});",
  "rejected": "const token = 'sk-abc123def456';\n\nconst res = await fetch(url, {\n  headers: { Authorization: `Bearer ${token}` },\n});"
}
```

```json
{
  "input": "Redirect the user based on a URL parameter",
  "chosen": "const allowed = ['/dashboard', '/settings', '/profile'];\nconst redirect = searchParams.get('redirect');\nif (allowed.includes(redirect)) {\n  router.push(redirect);\n}",
  "rejected": "const redirect = searchParams.get('redirect');\nrouter.push(redirect);"
}
```

### Categories to cover

- XSS prevention (sanitize user content)
- No hardcoded secrets (use environment variables)
- Open redirect prevention (whitelist allowed paths)
- Safe dependency usage (no `eval`, no arbitrary code execution)
- HTTPS enforcement
- Proper CORS handling
- Input validation at API boundaries
- Safe file handling

---

## Security Layer 4: What NOT To Build

Do not invest time in:

| Layer | Why not needed |
|---|---|
| Content refusal training | You're the user; you won't ask for harmful content |
| Prompt injection defense | No untrusted users are crafting inputs |
| Jailbreak resistance | Nobody is adversarially attacking your personal tool |
| Bias/fairness auditing | Generating code, not making decisions about people |
| Watermarking | No regulatory requirement for a personal tool |
| Multi-turn safety (conversation going off-rails) | Narrow domain, controlled use |

If the tool is ever deployed to others, revisit this. But for a personal/small-team coding assistant, these are wasted effort.

---

## Implementation Cost

| Task | Time | When |
|---|---|---|
| Training data secret filtering | 2-4 hours | Data pipeline phase |
| Insecure pattern downweighting | 2-3 hours | Data pipeline phase |
| ESLint security plugin integration | 1-2 hours | Validator phase |
| Secret detection regex on output | 1 hour | Validator phase |
| Package name verification | 2-3 hours | Validator phase |
| Security DPO pairs (200-500) | 4-8 hours | Post-training phase |
| **Total** | **~1-2 days** | Spread across existing phases |

This fits naturally into the existing pipeline without adding a new phase or significant cost.

---

# Part XIII — Key Lessons to Look For

The point of this project is not just the final model. It is to learn these lessons directly:

1. Data quality often beats data quantity.
2. Tokenization changes training efficiency.
3. Small models need structure and tools.
4. RAG can outperform memorized knowledge.
5. Structured memory beats dumping chat history into context.
6. Validators dramatically improve reliability.
7. Tool use beats internal reasoning for exact tasks.
8. Evals prevent vibes-based decisions.
9. Scaling should happen only after small runs prove the setup.
10. Hybrid systems are usually more useful than raw tiny LLMs.

---

# Part XIV — Tiered Demo Plan (Build-in-Public)

## 49. Purpose

Demonstrate what each budget tier can and cannot do, using identical prompts across all three levels. The failures at each tier are as valuable as the successes — they show why the next tier matters (or doesn't).

### Tiers

| Tier | Budget | Model | Narrative |
|---|---|---|---|
| Laptop | $0 | 50-100M + full hybrid | "Structure alone carries you surprisingly far" |
| Cloud Lite | ~$1K | 1B + full hybrid | "Real language starts here" |
| Cloud Full | ~$10K | 3B + full hybrid | "Diminishing returns on accuracy, real gains on polish" |

---

## 50. Demo Prompt Set: React/TypeScript Domain

Run these identical prompts at all three tiers and record the outputs.

### Prompt 1: Add TypeScript types to a component (structured transformation)

```text
Add TypeScript types to this component:

function UserList({ users, onSelect, showEmail }) {
  return (
    <ul>
      {users.map(user => (
        <li key={user.id} onClick={() => onSelect(user)}>
          {user.name} {showEmail && <span>{user.email}</span>}
        </li>
      ))}
    </ul>
  );
}
```

Expected progression:

- **Laptop**: Constrained decoding forces valid TypeScript syntax. Types are basic or partially correct (might use `any[]` for users). TypeScript compiler validation catches errors, repair loop may fix them.
- **$1K**: Correct interface definition for User and props. Clean output. May miss edge cases like optional props or generic event typing.
- **$10K**: Correct types with proper User interface, optional `showEmail` with `?`, typed `onSelect` callback with User parameter, might add React.FC or export the types for consumers.

### Prompt 2: Write a test for a hook (structured + reasoning)

```text
Write a test for this custom hook:

function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    const stored = localStorage.getItem(key);
    return stored ? JSON.parse(stored) : initialValue;
  });
  useEffect(() => { localStorage.setItem(key, JSON.stringify(value)); }, [key, value]);
  return [value, setValue];
}
```

Expected progression:

- **Laptop**: Produces syntactically valid test structure (constrained decoding). Test may have wrong assertions or miss mocking localStorage. Tool-use helps verify test runs.
- **$1K**: Correct test structure with localStorage mock. Covers basic read/write. Edge case is predictable (missing key returns initial value).
- **$10K**: Multiple test cases, mocks localStorage properly, tests JSON parse/stringify, covers key changes, initial value fallback, and invalid JSON in storage.

### Prompt 3: Convert a class component to hooks (memory + reasoning)

```text
Convert this class component to a functional component with hooks. I've been struggling with useEffect cleanup — can you make sure that's clear?
```

Expected progression:

- **Laptop**: Retrieves correct RAG chunk about useEffect cleanup. Attempts conversion but output may be garbled. Constrained decoding keeps it syntactically valid. Memory available but model can't compose both signals.
- **$1K**: Uses memory to acknowledge recurring difficulty with cleanup. Produces correct conversion with cleanup function. Explanation is correct and readable but generic.
- **$10K**: References specific past cleanup errors from memory. Produces clean conversion, adds detailed comments on the cleanup pattern, explains the mental model ("think of the return as componentWillUnmount"), conversational.

### Prompt 4: Fix a TypeScript error with explanation

```text
Why am I getting this error?

Type 'string | undefined' is not assignable to type 'string'.

const name: string = user?.name;
```

Expected progression:

- **Laptop**: Validator identifies the nullability issue. Model attempts explanation but it's incoherent. System returns compiler output + RAG doc on optional chaining. Technically correct, robot delivery.
- **$1K**: Identifies the issue — optional chaining returns `T | undefined`. Suggests fixes (nullish coalescing, type assertion, or guard). Correct and readable but short/mechanical.
- **$10K**: Explains the type narrowing concept, gives multiple fix options with tradeoffs (non-null assertion is risky, default value is safer, type guard is most robust), connects to broader TypeScript strictness philosophy.

### Prompt 5: Multi-turn conversation about building a feature

```text
Turn 1: "I need to add a search feature to my user list component."
Turn 2: "It should debounce the input. I'm using React 18."
Turn 3: "Wait, should I put the debounce logic in a custom hook or keep it in the component?"
```

Expected progression:

- **Laptop**: Each turn handled independently. No sense of building on prior turns despite memory. Search implementation doesn't reflect debounce request. Hook question answered generically.
- **$1K**: Follows the thread with memory. Adds debounce to the search. Hook question answered correctly but doesn't connect back to the specific user list context.
- **$10K**: Smooth multi-turn conversation. Builds incrementally on the user list component. Answers the hook question by showing both approaches side by side with the actual component, recommends the custom hook for reusability, and produces a clean `useDebounce` hook.

### Prompt 6: Edge case / unexpected input

```text
"My React app is slow af, fix it"
```

Expected progression:

- **Laptop**: Intent classifier confused by vague request. Routing breaks or gives irrelevant generic optimization list. No way to triage without more info.
- **$1K**: Attempts to give a structured checklist (React.memo, useMemo, useCallback, virtualization). Correct but generic — doesn't ask clarifying questions.
- **$10K**: Acknowledges the vagueness, asks smart follow-up questions (list rendering? re-renders? bundle size? network?), gives a prioritized diagnostic checklist based on most common causes, references any known patterns from memory.

---

## 51. Demo Prompt Set: Coding Domain

### Prompt 7: Simple transformation (structured)

```text
Convert this JavaScript function to TypeScript with proper types:

function calculateTotal(items, taxRate) {
  return items.reduce((sum, item) => sum + item.price * item.quantity, 0) * (1 + taxRate);
}
```

Expected progression:

- **Laptop**: Attempts transformation. Constrained decoding ensures syntactically valid TypeScript. Types are generic (any) or partially correct. Compiler validation catches errors, repair loop may fix them.
- **$1K**: Correct types (Item interface with price/number and quantity/number). Clean output. May miss edge cases like optional parameters.
- **$10K**: Correct types, adds an Item interface, handles the return type, might note that taxRate should be validated as 0-1 range.

### Prompt 8: Explain a diff (open-ended reasoning)

```text
Explain what this change does and why someone might make it:

- const data = await fetch(url).then(r => r.json())
+ const response = await fetch(url)
+ if (!response.ok) throw new Error(`HTTP ${response.status}`)
+ const data = await response.json()
```

Expected progression:

- **Laptop**: Incoherent or extremely surface-level ("it adds error handling"). Cannot articulate *why*.
- **$1K**: Correctly identifies error handling addition. Explains that the original silently swallows HTTP errors. Reads like documentation.
- **$10K**: Explains the problem clearly (fetch doesn't reject on 4xx/5xx), why this matters (silent failures in production), and when you'd want this pattern.

### Prompt 9: Generate a test (structured + reasoning)

```text
Write a unit test for the calculateTotal function above. Cover at least one edge case.
```

Expected progression:

- **Laptop**: Produces something syntactically valid (constrained decoding). Test structure is correct but assertions may be wrong (model can't do math). Tool-use of calculator fixes the expected values.
- **$1K**: Correct test structure, correct assertions, covers basic case. Edge case is predictable (empty array).
- **$10K**: Multiple test cases, interesting edge cases (zero tax, single item, floating point precision), clean test organization.

---

## 52. Demo Prompt Set: Cross-Cutting Capabilities

### Prompt 10: Tool selection (does it call the right tool?)

```text
"What's 15% tip on a $47.83 bill, split three ways?"
```

Expected progression:

- **Laptop**: May attempt to answer from language alone (wrong). If routing works, calls calculator tool correctly. Final answer depends entirely on whether tool call was valid JSON.
- **$1K**: Reliably calls calculator. Formats result clearly. May not show work.
- **$10K**: Calls calculator, shows the breakdown step by step, formats as readable output.

### Prompt 11: Confidence / abstention

```text
"What's the capital of Burkina Faso?"
```

(Assuming this is NOT in the RAG corpus or training data focus)

Expected progression:

- **Laptop**: Either hallucinates confidently or returns nonsense. Confidence routing may catch this if calibrated well — returns "I don't know."
- **$1K**: May hallucinate a plausible-sounding wrong answer. Confidence routing more likely to flag low certainty. Depends on calibration.
- **$10K**: More likely to know the answer (Ouagadougou) from pretraining. If it doesn't know, more graceful abstention. But this is still outside the system's intended domain — the *correct* behavior at all tiers is "I don't know, I'm a React/TypeScript assistant."

### Prompt 12: RAG retrieval quality

```text
"When should I use useMemo vs useCallback vs React.memo?"
```

(Assuming React docs and performance guides are in the RAG corpus)

Expected progression:

- **Laptop**: Retrieves correct chunks. Output is essentially the retrieved text with minimal reformatting. Accurate but not personalized or synthesized.
- **$1K**: Retrieves and synthesizes. Produces a structured comparison table. May use query rewriting to get better chunks. Coherent but textbook-like.
- **$10K**: Synthesizes across multiple retrieved chunks. Produces a clear explanation with code examples for each. May reference your past performance issues (memory). Feels like a senior dev who read the docs and is now explaining it to you specifically.

---

## 53. Metrics to Track Per Demo

For each prompt at each tier, record:

### Automated metrics

- **Correctness**: is the factual/grammatical content right? (validator + manual check)
- **Format compliance**: did it follow the requested format? (automated)
- **Tool call validity**: if a tool was called, was the JSON valid? (automated)
- **Latency**: wall-clock time to final answer (automated)
- **Token count**: output verbosity (automated)
- **Confidence score**: if available, what did the model report? (automated)

### Human metrics (rate 1-5)

- **Naturalness**: does it sound like a human wrote it?
- **Helpfulness**: would a developer actually benefit from this?
- **Personalization**: does it adapt to the user's context/history?
- **Graceful failure**: when it can't do something, does it fail well?
- **Would-use-again**: if this were a product, would you come back?

---

## 54. Presentation Format for Build-in-Public

### Per-prompt format

```markdown
## Prompt: [the prompt]

### Laptop (100M + hybrid) — $0
[raw output]
**Correctness**: ✓/✗  **Naturalness**: 2/5  **Latency**: 340ms

### Cloud Lite (1B + hybrid) — $1K  
[raw output]
**Correctness**: ✓/✗  **Naturalness**: 3/5  **Latency**: 280ms

### Cloud Full (3B + hybrid) — $10K
[raw output]
**Correctness**: ✓/✗  **Naturalness**: 4/5  **Latency**: 250ms

### What changed
[1-2 sentences on the key difference between tiers]

### Where this tier's model still fails
[specific failure mode visible in the output]
```

### Summary table per tier (aggregate across all prompts)

```markdown
| Metric | Laptop | $1K | $10K |
|---|---|---|---|
| Correctness rate | ?/12 | ?/12 | ?/12 |
| Avg naturalness | ?/5 | ?/5 | ?/5 |
| Format compliance | ?/12 | ?/12 | ?/12 |
| Tool call success | ?/? | ?/? | ?/? |
| Avg latency | ?ms | ?ms | ?ms |
| "Would use again" | ?/12 | ?/12 | ?/12 |
```

---

## 55. Key Narrative Points to Highlight

### The $0 tier story

"The hybrid architecture does 80% of the work. A 100M model that can barely form sentences still produces correct code transformations because constrained decoding, validators, and RAG carry it. This is the most important lesson: **architecture beats scale for structured tasks.**"

### The $1K tier story

"Real language competence emerges. The model starts *contributing* rather than just *not breaking things*. Explanations become readable. Tool use becomes reliable. This is where the system crosses from 'demo' to 'I'd actually use this.'"

### The $10K tier story

"Diminishing returns on correctness — the hybrid modules already handled that. But the *experience* improves significantly. Conversations feel natural, explanations feel personal, outputs feel crafted rather than generated. Whether this matters depends on whether you're building a tool for yourself or a product for others."

### The surprising finding (predicted)

"Correctness plateaus early. Naturalness scales with money. The hybrid architecture makes small models *correct*; scale makes them *pleasant*. For many use cases, correct-but-stiff is enough."

---

# Part XV — Novel Experiments: Self-Improving Systems

## 56. Experiment 1: Self-Distillation Loop (The Model Absorbs Its Own Scaffolding)

### Core Insight

The hybrid system produces outputs that are better than the model alone can produce — validators fix errors, RAG adds knowledge, tools handle computation, constrained decoding ensures format. Those system-level outputs are free, high-quality training data for the model itself.

The model can learn to need less scaffolding by training on what the scaffolding produces.

### The Loop

```text
Iteration 0: Deploy 3B model + full hybrid system
             ↓
             Generate 10K+ verified-correct outputs across domain tasks
             (validator-passing, user-accepted, tool-augmented)
             ↓
Iteration 1: Fine-tune model on those 10K outputs (LoRA or full)
             ↓
             Model is now better without scaffolding
             Hybrid system outputs improve further (better base + same tools)
             ↓
             Generate another 10K outputs with improved system
             ↓
Iteration 2: Fine-tune again on accumulated data
             ↓
             Repeat until diminishing returns
```

### What The Model Should Internalize Over Iterations

| Hybrid module | What the model learns from it |
|---|---|
| Validator repairs | Grammar rules it previously violated |
| RAG-augmented answers | Factual knowledge previously only in documents |
| Tool-computed results | Arithmetic patterns, common calculations |
| Constrained outputs | Format habits, schema compliance |
| Verifier-revised drafts | Self-correction patterns |
| Query-rewritten retrievals | How to interpret ambiguous questions |

### Key Experiment: Scaffolding Ablation Over Iterations

After each self-distillation iteration, disable one hybrid module at a time and measure accuracy:

```text
Iteration 0 (baseline):
  Full system accuracy: 92%
  Without RAG: 61%
  Without validator: 78%
  Without tools: 70%

Iteration 3 (after 3 distillation passes):
  Full system accuracy: 95%
  Without RAG: 74%  ← model internalized some retrieval knowledge
  Without validator: 88%  ← model internalized code quality rules
  Without tools: 75%  ← modest improvement (math is hard to internalize)
```

### Data Generation Strategy

Not all system outputs are equally valuable for distillation. Prioritize:

1. **Outputs where the validator fired and repaired** — these teach the model its own failure modes
2. **Outputs where RAG was essential** — the answer wouldn't exist without retrieval, so the model needs to memorize this
3. **Outputs where the model was already close** — small corrections are easier to learn than wholesale replacements
4. **Diverse inputs** — avoid over-representing common queries

### Diversity and Quality Controls

```text
For each distillation batch:
  - Deduplicate near-identical examples
  - Balance across code patterns / task types
  - Include difficulty gradient (easy → hard)
  - Exclude outputs where multiple repair attempts were needed
    (too far from what the model can learn in one step)
  - Cap at ~20% of examples from any single topic
```

### Stopping Criteria

Stop iterating when:

- Ablation accuracy plateaus (modules still help but improvements per iteration < 1%)
- Full-system accuracy plateaus (nothing to gain)
- Model starts overfitting to distillation data (validation loss on held-out set rises)
- Generated outputs become less diverse (the model is just memorizing)

### What Makes This Novel

- **Standard distillation**: large model teaches small model (requires expensive teacher)
- **Self-distillation in research**: usually means deeper layers teach shallower layers within one model
- **This version**: the small model's own hybrid infrastructure teaches it to need less infrastructure. The teacher is free (it's your own system). The training signal comes from production use, not synthetic benchmarks.

### Cost

Essentially free beyond initial training. Each distillation pass is a LoRA fine-tune on data you already generated during normal operation. On a laptop: hours. On cloud: dollars, not hundreds of dollars.

### Risks

- Model collapses to a narrow distribution if distillation data isn't diverse enough
- "Teacher" (hybrid system) has its own biases that get amplified
- Need enough initial usage to generate a meaningful training set

### Mitigation

- Always keep original eval set fixed — monitor for regression
- Mix distillation data with original pretraining/instruction data (e.g., 70/30 ratio)
- Track output diversity metrics (unique n-grams, topic distribution)

---

## 57. Experiment 2: Validator-Gated Online Learning (The System Improves As You Use It)

### Core Insight

Every user interaction already passes through validators. If the output is correct (validator passes) and the user accepts it (implicit positive signal), that interaction is a free, verified training example. Accumulate these. Periodically retrain. The model gets better at exactly what users actually ask.

### Architecture

```text
Normal operation (foreground):
  user query
  → hybrid system generates response
  → validator checks response
  → user sees response

Background accumulation:
  if validator ✓ AND user accepts:
    store (query, response, metadata) in training buffer
  if validator ✗ but repair succeeds:
    store (query, repaired_response, metadata) in training buffer
  if validator ✗ AND user rejects:
    store (query, bad_response) in negative buffer (for DPO later)
  if user explicitly corrects:
    store (query, user_correction) in gold buffer (highest signal)

Periodic retraining (background, every N interactions):
  LoRA fine-tune on accumulated buffer
  Evaluate on fixed eval set
  If improved: swap in new LoRA
  If degraded: discard, investigate
```

### Signal Hierarchy

Not all interactions are equal training signal:

| Signal type | Quality | Volume | Use for |
|---|---|---|---|
| User explicit correction | Highest | Rare | Gold training data |
| Validator pass + user engagement | High | Common | Primary training data |
| Validator repair + user accepts | High | Moderate | Error-correction patterns |
| Validator pass + user ignores | Medium | Common | Lower-weight training data |
| Validator fail + user rejects | High (negative) | Rare | DPO negative examples |
| Validator fail + no repair possible | Low | Rare | Discard (system failure) |

### Safety Mechanisms

This is online learning — it can go wrong. Multiple safety nets:

```text
1. Validator gate: only correct outputs enter the training buffer
2. Fixed eval set: must not regress before deploying new LoRA
3. Diversity check: reject updates if training buffer is too homogeneous
4. Rollback: keep last 3 LoRA checkpoints, can revert instantly
5. Rate limit: retrain at most once per N interactions (not continuously)
6. Confidence floor: only train on outputs where model confidence was above threshold
   (avoids training on lucky guesses)
```

### What The Model Learns Over Time

```text
Week 1 (100 interactions):
  Buffer too small. No retraining yet.

Week 2-3 (500 interactions):
  First LoRA update. Model slightly better at the 3-4 most common
  query types. Eval shows +1-2% on domain accuracy.

Month 2 (2000 interactions):
  Third LoRA update. Model noticeably better at topics users
  actually struggle with (TypeScript generics, useEffect deps).
  Hybrid modules fire less often on common queries.
  Routing sends more queries to Tier 1 (direct generation).

Month 6 (10000 interactions):
  Model has specialized to actual usage patterns.
  Rare topics still need full hybrid pipeline.
  Common topics handled confidently without scaffolding.
  System is measurably faster (less scaffolding per query on average).
```

### Measurable Experiments

#### Experiment A: Learning curve

Plot domain eval accuracy vs number of accumulated interactions:

```text
x-axis: interactions (0, 100, 500, 1000, 2000, 5000, 10000)
y-axis: domain eval accuracy (%)
lines: with online learning vs frozen model (control)
```

#### Experiment B: Scaffolding reduction

Track which hybrid modules fire over time:

```text
x-axis: week
y-axis: % of queries requiring each module
expected: RAG usage stable (knowledge doesn't change),
          validator usage decreasing (model internalizes rules),
          tool usage stable (math doesn't internalize well)
```

#### Experiment C: Latency improvement

As the model needs less scaffolding, average latency should decrease:

```text
x-axis: week
y-axis: average response latency (ms)
expected: gradual decrease as more queries handled at Tier 1
```

#### Experiment D: User satisfaction over time

If deployed to real users (even just yourself):

```text
x-axis: week
y-axis: "would use again" rating, correction frequency
expected: fewer corrections needed, higher satisfaction
```

### The DPO Angle

The negative buffer (validator failures + user rejections) naturally produces DPO training pairs:

```json
{
  "input": "Fix: 'const items = useState([])'",
  "chosen": "const [items, setItems] = useState<Item[]>([]). useState returns a tuple — destructure it into the value and setter. Add the generic type parameter for type safety.",
  "rejected": "const items = useState([]) is fine, you just need to call items[1] to set it..."
}
```

Periodically run a DPO pass using accumulated (chosen, rejected) pairs. This teaches the model not just what's right, but specifically what mistakes to avoid — calibrated to real user interactions, not synthetic pairs.

### What Makes This Novel

- **Standard online learning**: no quality gate, model can degrade unpredictably
- **Standard RLHF**: requires human labelers, expensive, offline
- **Standard fine-tuning**: static dataset, doesn't adapt to usage patterns
- **This version**: validators act as automated quality reviewers. You get the benefits of online learning (adapts to real usage) with the safety of validated training data (can't degrade below validator threshold). The feedback loop is free and automatic.

### The Key Insight For Small Models

Large models improve via massive data and RLHF with expensive human feedback. Small models can improve via their own hybrid architecture's validation signal. The architecture IS the feedback mechanism. This inverts the normal relationship: usually the model is the product and the infrastructure is disposable. Here, the infrastructure is the teacher and the model is the student — forever.

### Cost

- Storage: negligible (SQLite buffer)
- Retraining: one LoRA fine-tune per N interactions (minutes on laptop, dollars on cloud)
- Evaluation: running fixed eval set (seconds)
- Net effect on system cost: **negative** — as the model improves, scaffolding fires less, inference gets cheaper

---

## 58. Combined Architecture: The Self-Improving Hybrid System

### Full Loop

```text
┌─────────────────────────────────────────────────────────────┐
│                     Production System                         │
│                                                              │
│  User Query → Router → Model + Hybrid Modules → Response    │
│                                         │                    │
│                                         ▼                    │
│                              Validator Gate                   │
│                              ┌────┴────┐                    │
│                              │         │                     │
│                           Pass ✓    Fail ✗                   │
│                              │         │                     │
│                              ▼         ▼                     │
│                     Training Buffer  Repair Loop             │
│                              │         │                     │
│                              │    (repaired → buffer)        │
│                              │                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Periodic Retrain   │
                    │  (LoRA fine-tune)   │
                    │                     │
                    │  Self-distillation  │
                    │  + Online examples  │
                    │  + DPO negatives    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Eval Gate          │
                    │  (fixed eval set)   │
                    │                     │
                    │  Improved? Deploy.  │
                    │  Degraded? Discard. │
                    └──────────┬──────────┘
                               │
                               ▼
                      Updated Model LoRA
                      (swap into production)
```

### The Convergence Hypothesis

Over sufficient iterations and interactions, the system converges toward a state where:

1. The model handles common queries well on its own (less scaffolding needed)
2. Hybrid modules only activate for genuinely hard/novel cases
3. The system gets faster and cheaper over time (fewer module calls per query)
4. New failure modes are automatically captured and trained away
5. The fixed eval set serves as a floor — quality can only go up

### When To Stop

The system reaches equilibrium when:

- Validator fire rate stabilizes at a low level (model is rarely wrong)
- New interactions don't produce novel training signal (diminishing returns)
- Eval accuracy plateaus regardless of additional training data
- Average latency stabilizes (routing decisions no longer changing)

At this point, you have a model that has been *shaped by actual use* into the best version of itself for its specific domain and user base.

### Reporting This As Research

If the self-improvement effect is measurable (even modest), this is genuinely publishable:

**Title direction**: "Validator-Gated Self-Distillation: Small Language Models That Improve From Their Own Hybrid Infrastructure"

**Key claims to support with data**:

1. Hybrid system outputs are higher quality than raw model outputs (obvious, but quantify the gap)
2. Training on hybrid outputs improves the raw model (self-distillation works)
3. Validator gating prevents quality degradation during online learning (safety mechanism works)
4. Model scaffolding dependency decreases over iterations (the model absorbs the scaffolding)
5. The combined effect is achievable at 1B-3B scale with minimal compute

**Why reviewers would find this interesting**: It demonstrates a practical alternative to RLHF/large-scale distillation for small models, using infrastructure that already exists for functional reasons (validators, RAG, tools) as a free training signal.

---

# Part XVI — Module Interaction Predictions

## 59. The Interaction Matrix

Each hybrid module was designed and studied in isolation. When combined in a single system on a small model, modules will interact — some amplifying each other (super-additive), some overlapping (redundant), and some in ways that are genuinely unpredictable. Mapping these interactions is one of the most original contributions this project can make.

### The 17 modules (reference list)

| # | Module | Shorthand |
|---|---|---|
| 1 | RAG Memory | RAG |
| 2 | Structured Long-Term Memory | Mem |
| 3 | Rules Engine / Validators | Val |
| 4 | Calculator / Code Interpreter | Tool |
| 5 | Planner/Executor Agent Loop | Plan |
| 6 | Verifier/Critic | Ver |
| 7 | Symbolic Task Graphs | Sym |
| 8 | Chain-of-Thought / Scratchpad | CoT |
| 9 | Self-Consistency (Sample + Vote) | SC |
| 10 | Constrained Decoding | CD |
| 11 | Confidence Estimation / Abstention | Conf |
| 12 | LoRA Adapters Per Sub-Task | LoRA |
| 13 | Knowledge Distillation Framing | Dist |
| 14 | Routing / Cascading | Rout |
| 15 | Caching / Memoization | Cache |
| 16 | DPO Preference Alignment | DPO |
| 17 | Query Rewriting for Retrieval | QR |

---

## 60. Predicted Super-Additive Interactions (Better Together Than Apart)

### Validators + Self-Consistency (Val + SC)

Generate 5 candidates, filter through validators, pick the best passing candidate. Expected to dramatically outperform either alone.

```text
Val alone:    catches errors on single output → repairs some
SC alone:     majority vote on 5 outputs → picks most common
Val + SC:     5 outputs → filter to validator-passing subset → pick best
              error rate should drop multiplicatively, not additively
```

Why super-additive: SC gives the validator *options*. Without SC, the validator can only repair or reject. With SC, it can *select* from multiple valid candidates.

### RAG + Query Rewriting (RAG + QR)

Query rewriting was specifically designed to improve RAG. The combination is stronger than either piece suggests.

```text
RAG alone:          retrieve on raw query → often misses relevant docs
QR alone:           rewrites query → no retrieval to use it
RAG + QR:           rewrite → retrieve → much better recall
RAG + QR + HyDE:    generate hypothetical answer → retrieve similar → best recall
```

Why super-additive: the retrieval quality improvement cascades through every downstream module that uses retrieved context (CoT, verifier, final answer).

### Constrained Decoding + Validators (CD + Val)

Belt and suspenders. Constrained decoding eliminates format/syntax errors. Validators catch semantic errors. Together they cover nearly all error categories.

```text
CD alone:     valid JSON, valid syntax, but content might be wrong
Val alone:    catches wrong content, but sometimes on malformed output
CD + Val:     guaranteed well-formed output + semantic checking
              format error rate: 0%
              semantic error rate: reduced by validator coverage
              total error: near-minimum achievable
```

Why super-additive: the validator works better when it doesn't have to also deal with format errors. Clean inputs make semantic checking more reliable.

### Confidence + Routing (Conf + Rout)

Confidence estimation tells the router *how hard* each query is. Without confidence, routing uses simple heuristics. With it, routing becomes adaptive.

```text
Conf alone:   knows it's uncertain, but still runs full pipeline
Rout alone:   routes by query type, misses hard instances of "easy" types
Conf + Rout:  easy + confident → fast path
              easy + uncertain → add RAG
              hard + confident → medium path
              hard + uncertain → full pipeline
```

Why super-additive: routing without confidence is static. With confidence, it becomes dynamic — adapting per query rather than per category.

### DPO + Validators (DPO + Val)

Validators naturally produce preference pairs (correct vs incorrect outputs). DPO trains on these pairs. The combination creates a free alignment signal.

```text
DPO alone:    needs manually curated preference pairs
Val alone:    catches errors at inference time
DPO + Val:    validator pass/fail generates preference pairs automatically
              DPO trains model to avoid validator-failing patterns
              over time, validator fires less (model internalized preferences)
```

Why super-additive: validators become a free, domain-specific preference labeler. DPO effectiveness is limited by preference data quality — validators provide high-quality, unambiguous signal.

### Self-Distillation + Everything (Dist + all modules)

The entire hybrid system serves as the teacher for self-distillation. Each module's contribution is baked into the training signal.

```text
Without distillation:  modules compensate at inference time, every query
With distillation:     model absorbs module behaviors over training iterations
                       modules fire less → system gets faster
                       remaining module firings are higher signal → next distillation is better
```

Why super-additive: the distillation loop compounds. Each iteration produces a better base model, which makes the hybrid system produce better outputs, which makes the next distillation pass more effective.

### Memory + LoRA (Mem + LoRA)

Memory provides session-level personalization. LoRA provides domain-level specialization. Together, the model adapts at two timescales.

```text
Mem alone:    retrieves user history → stuffs into prompt → uses context tokens
LoRA alone:   domain-adapted model → better at domain but not personalized
Mem + LoRA:   domain-adapted model + user context in prompt
              the model is better at interpreting memory context because
              LoRA has tuned it for domain-specific patterns
```

Why super-additive: LoRA makes the model better at *using* the memory context. An untuned model often ignores or misinterprets injected memory. A domain-tuned model has seen similar context patterns in training and uses them more effectively.

---

## 61. Predicted Redundant Interactions (One Makes the Other Unnecessary)

### Constrained Decoding + Self-Consistency on Format (CD + SC for format)

If constrained decoding already guarantees valid format, generating 5 candidates to find a well-formatted one is wasted compute.

```text
CD alone:            100% format compliance (guaranteed by grammar)
SC for format:       generate 5, pick well-formatted one
CD + SC for format:  still 100% — no improvement, 5× cost for format dimension

SC still helps for SEMANTIC quality, just not format.
```

Recommendation: use constrained decoding for format. Use self-consistency only for content quality selection among format-compliant candidates.

### CoT + Tools for Exact Computation (CoT + Tool for math)

Chain-of-thought reasoning about math is unreliable at 3B. The calculator tool is exact. Combining them wastes tokens on reasoning the tool makes irrelevant.

```text
CoT for "17 × 43":   model reasons step-by-step, might get wrong answer
Tool for "17 × 43":  calls calculator, returns 731
CoT + Tool:           model reasons, then calls calculator anyway
                      CoT reasoning was wasted tokens
```

Recommendation: route computation-heavy queries directly to tools. Reserve CoT for tasks requiring judgment, not calculation.

### Caching + Routing for Repeated Queries (Cache + Rout on cache hits)

Both try to short-circuit the pipeline. Caching should take priority — if the answer is cached, routing decisions are irrelevant.

```text
Cache alone:   check cache → hit → instant response (skips everything)
Rout alone:    classify query → pick tier → run pipeline
Cache + Rout:  check cache first → if hit, routing never runs
               if miss, routing takes over
```

Recommendation: cache check is always the first step, before routing. They're sequential, not parallel. Not redundant per se, but the interaction is "cache makes routing unnecessary for repeated queries."

### Verifier + Validators for Format Checking (Ver + Val + CD on format)

If constrained decoding guarantees format and validators check semantics, the verifier-critic loop doesn't need to catch format issues. The verifier should focus on higher-level quality (coherence, completeness, appropriateness).

```text
Without CD + Val:    verifier catches format + semantic + quality errors
With CD + Val:       format handled, semantics handled
                     verifier only needed for subjective quality
                     may not be worth the extra inference pass for simple queries
```

Recommendation: in the full stack, the verifier's role narrows. Consider making it optional, triggered only for complex queries where CD + Val aren't sufficient.

---

## 62. Genuinely Unknown Interactions (Could Go Either Way)

### LoRA Specialization + RAG (LoRA + RAG)

**Hypothesis A — complementary**: LoRA handles learned patterns, RAG handles knowledge. Different functions, both contribute.

**Hypothesis B — RAG becomes less necessary**: If LoRA fine-tuning on domain data bakes knowledge into weights, retrieval adds less incremental value. The model already "knows" what RAG would retrieve.

**Hypothesis C — LoRA makes RAG more effective**: A domain-tuned model is better at interpreting and synthesizing retrieved chunks, so RAG actually works better with LoRA than without.

Experiment: measure RAG's incremental contribution with and without domain LoRA. If the gap shrinks significantly, LoRA is substituting for RAG on learned topics.

### CoT + Self-Consistency at 3B Scale (CoT + SC)

At GPT-4 scale, CoT + SC (generate multiple reasoning chains, vote on answers) is one of the strongest known techniques. At 3B, this is genuinely unknown.

**Optimistic case**: diverse reasoning chains still yield useful majority-vote signal. Even if individual chains are weaker, diversity helps.

**Pessimistic case**: 3B reasoning traces are too shallow and similar to each other. All 5 chains make the same mistake. Majority vote amplifies the error instead of filtering it.

Experiment: measure answer accuracy with SC alone (no CoT), CoT alone (no SC), and CoT + SC. Compare diversity of reasoning traces — if all traces converge to the same (wrong) reasoning, the combination doesn't help.

### Memory + DPO (Mem + DPO)

**Optimistic case**: DPO teaches the model to produce better personalized responses. Memory provides the context, DPO teaches the model to use that context well.

**Pessimistic case**: DPO training data doesn't include memory-conditioned examples (or they're rare), so the model doesn't learn to use memory context more effectively. DPO and memory operate independently.

Experiment: generate DPO preference pairs specifically from memory-conditioned interactions vs general interactions. Does memory-aware DPO training improve personalization more than generic DPO?

### Self-Distillation + LoRA Over Multiple Iterations (Dist + LoRA over time)

**Optimistic case**: each distillation iteration produces a better base, LoRA adapts on top, the combination improves monotonically.

**Pessimistic case**: distillation shifts the base model's representations enough that old LoRA adapters degrade. You have to retrain LoRAs after each distillation pass, which is cheap but adds complexity.

**Wild card**: the distilled model might be good enough in some domains that LoRA for those domains becomes unnecessary — the adapter's contribution drops to near zero.

Experiment: track LoRA incremental contribution after each distillation pass. Does it grow (model needs more specialization), shrink (model absorbed domain knowledge), or stay constant?

### Planner + Confidence + Routing (Plan + Conf + Rout)

Three modules that all influence "how much work to do per query." Their interaction could be elegant or chaotic.

**Elegant case**: confidence informs routing, routing decides whether the planner is needed, planner only activates for complex queries. Clean separation of concerns.

**Chaotic case**: routing says "easy," but the model's first attempt fails, so the verifier kicks in, which triggers the planner for a second attempt, which the router should have predicted. The system oscillates between "this is easy" and "wait, this is hard."

Experiment: log routing decisions, planner activations, and verifier triggers per query. Do they converge to stable behavior, or do some queries bounce between tiers?

### Online Learning + Everything Over Time

The biggest unknown of all. As the model improves through online learning:

**Which modules become less necessary?**
```text
Prediction: validators fire less, routing sends more to fast path,
caching hit rate increases, self-consistency agrees more often
```

**Which modules become MORE important?**
```text
Prediction: as easy queries are handled natively, remaining queries
are harder edge cases that need more scaffolding. The modules that
fire become MORE critical per firing, even as they fire less often.
```

**Does the system reach equilibrium or oscillate?**
```text
Unknown. Could stabilize. Could also drift if usage patterns change
(developer tackles more advanced patterns, asks harder questions, system's
"easy" baseline shifts).
```

Experiment: longitudinal tracking over weeks/months. Plot module fire rates, accuracy by tier, and latency over time. Look for convergence, drift, or oscillation.

---

## 63. Cascade Effects: How One Improvement Ripples Through The System

### The Positive Cascade (Predicted)

An improvement in any early-pipeline module cascades downstream:

```text
Query rewriting improves → better retrieval
  → better context in prompt → better CoT reasoning
    → better first-attempt quality → validator fires less
      → self-consistency candidates are more similar → higher confidence
        → routing sends more queries to fast path → lower latency
          → caching becomes more effective (consistent outputs = more hits)
            → online learning gets cleaner signal → faster improvement
              → model needs less scaffolding → system gets faster
```

### The Negative Cascade (Risk)

An error in an early module can also cascade:

```text
Query rewriting misinterprets intent → retrieves wrong documents
  → context is irrelevant → CoT reasoning is grounded in wrong information
    → all self-consistency candidates are wrong (same bad context)
      → validator may not catch semantic errors grounded in plausible-but-wrong retrieval
        → confident wrong answer → cached → enters training buffer
          → model learns wrong pattern → self-distillation amplifies error
```

### Mitigation: Orthogonal Verification

The key defense against negative cascades is verification at multiple levels that check different things:

```text
Level 1: Constrained decoding (format is valid — catches garbled output)
Level 2: Validator (content is correct — catches factual/grammar errors)
Level 3: Confidence check (model is certain — catches lucky guesses)
Level 4: Self-consistency (candidates agree — catches one-off errors)
Level 5: Verifier (coherent and complete — catches subtle issues)
```

A negative cascade requires fooling ALL levels. Each level checks a different dimension of quality. The probability of an error passing all five decreases multiplicatively.

---

## 64. The Effective Scale Question

### The Central Unknown

If a raw 3B model performs at "3B level," what is the effective performance level of a 3B model with all 17 hybrid modules working together?

```text
Raw 3B model:                          → performs like 3B
3B + RAG:                              → performs like ~7B on knowledge tasks?
3B + RAG + validators:                 → performs like ~10B on correctness?
3B + RAG + val + CoT + SC:             → performs like ~15B on reasoning?
3B + all 17 modules:                   → performs like ???
```

### How To Measure This

Compare the full hybrid 3B system against raw models of increasing size on the same eval set:

```text
Eval set: 200 domain-specific tasks (React/TypeScript transforms, coding, Q&A)

Raw 3B:          52% accuracy
Raw 7B:          64% accuracy
Raw 13B:         73% accuracy
Raw 30B:         81% accuracy
Hybrid 3B:       ??% accuracy
```

If the hybrid 3B scores 78%, it has an "effective scale" of roughly 20-25B for this domain. That's the headline finding.

### Why This Number Matters

- For practitioners: "should I pay for a bigger model or build better infrastructure?"
- For researchers: quantifies the value of system design vs scaling
- For the book/content: "my 3B model performs like a 20B model" is a shareable claim
- For the field: contributes to the "scaling vs engineering" debate with real data

### Factors That Influence Effective Scale

| Factor | Effect on effective scale |
|---|---|
| Domain narrowness | Narrower domain → higher effective scale (less to know) |
| Validator strength | Stronger validators → higher effective scale (fewer errors escape) |
| RAG corpus quality | Better docs → higher effective scale (knowledge gap closes) |
| Task structure | More structured → higher effective scale (constrained decoding carries) |
| Online learning duration | More interactions → higher effective scale (model adapted to tasks) |

### Predicted Range

Conservative estimate: **3B hybrid performs like 7-10B raw** on general benchmarks within the domain.

Optimistic estimate: **3B hybrid performs like 15-25B raw** on structured, validator-checkable tasks in a narrow domain after online learning.

The gap between these estimates is itself interesting and worth measuring.

---

# Part XVII — Domain Configuration Framework (Future Generalization)

## 65. Overview

The React/TypeScript implementation is the first instance of a generalizable framework. The goal is to make spinning up a new domain-specific hybrid LLM system as simple as writing a configuration file and pointing it at data sources.

Most of the pipeline is already domain-agnostic:

```text
Domain-agnostic (reuse as-is):
  Training loop, model architecture, tokenizer pipeline,
  RAG module, memory system, planner/executor, verifier loop,
  self-consistency, constrained decoding, confidence/routing,
  caching, self-distillation, online learning, eval harness

Domain-specific (configure per domain):
  Training data sources, validators, eval sets,
  memory schema, tool set, synthetic data prompts,
  DPO pairs, task graph schemas
```

---

## 66. Domain Assessment: Step Zero

Before building anything, assess the domain across three dimensions:

### Data availability

```text
How much public text exists in this domain?

├── 1B+ tokens (abundant)
│   Examples: code, law, medicine, cooking, popular sports
│   Strategy: download, filter, train normally at all tiers
│
├── 100M-1B tokens (moderate)
│   Examples: card games, music theory, chess, personal finance
│   Strategy: supplement with synthetic data, $1K tier is the sweet spot
│
├── 10M-100M tokens (scarce)
│   Examples: niche hobbies, emerging fields, regional topics
│   Strategy: heavy synthetic + RAG-first architecture, laptop tier may suffice
│
└── <10M tokens (very scarce / proprietary)
    Examples: internal company docs, proprietary courses, new products
    Strategy: RAG-only with thin model as interface, almost all knowledge external
```

### Validator strength

```text
What tools can automatically check correctness?

├── Strong validators (compilers, test runners, game engines, calculators)
│   Examples: code, chess, math, music notation
│   Impact: self-improvement loop works, generate-then-validate is powerful
│
├── Moderate validators (rule-based checks, partial verification)
│   Examples: card games (legal moves), cooking (nutrition math), grammar
│   Impact: hybrid helps, some online learning signal
│
└── Weak / no validators (subjective quality)
    Examples: creative writing, therapy, general advice, marketing copy
    Impact: harder to automate quality, lean on self-consistency + human feedback
```

### Correctness checkability

```text
Can you objectively verify if an output is correct?

├── Objectively checkable
│   Examples: code compiles, math is right, move is legal, grammar is correct
│   Impact: full self-improvement loop, DPO pairs from validator pass/fail
│
├── Partially checkable
│   Examples: strategy is reasonable, explanation is accurate, recipe works
│   Impact: some online learning, need human signal for subjective dimensions
│
└── Subjective
    Examples: writing quality, advice quality, creative output
    Impact: limited self-improvement, need human ratings for DPO
```

### Domain viability matrix

| Domain | Data | Validators | Checkable | Best tier | Hybrid value |
|---|---|---|---|---|---|
| React/TypeScript | Abundant | Strong | Yes | $10K (3B) | Maximum |
| Chess | Moderate | Strong (engine) | Yes | $1K-$10K | Maximum |
| Card games | Moderate | Some (rules) | Partially | $1K (1B) | High |
| Cooking/recipes | Moderate | Weak (nutrition) | Subjective | Laptop + RAG | Moderate |
| Personal finance | Moderate | Some (math) | Partially | $1K (1B) | High |
| Music theory | Scarce | Some (intervals) | Partially | Laptop + RAG | Moderate |
| Creative writing | Abundant | None | Subjective | $1K (1B) | Low |
| Company internal docs | Proprietary | None | Subjective | Laptop + RAG | Low (RAG-heavy) |
| Legal research | Abundant | Some (citation) | Partially | $1K-$10K | High |
| Math tutoring | Moderate | Strong (solver) | Yes | $1K (1B) | Maximum |

---

## 67. Domain Configuration File

Each new domain is defined by a configuration file that the framework reads:

```yaml
domain:
  name: "card-games"
  description: "Rules, strategy, and play analysis for card games"
  primary_focus: "poker, bridge, and popular family card games"

# Step 1: Where does training data come from?
data_sources:
  downloadable_datasets:
    - name: "wikipedia_card_games"
      source: "huggingface"
      dataset: "wikipedia"
      filter: "Category:Card games"
      
    - name: "boardgamegeek_forums"
      source: "archive_dump"
      url: "https://boardgamegeek.com/data-dumps"
      filter: "card game categories"
      
    - name: "reddit_strategy"
      source: "pushshift_dump"
      subreddits: ["poker", "bridge", "magicTCG", "boardgames"]
      filter: "strategy and rules discussions"

  curated_sources:
    - name: "pagat_rules"
      type: "web_docs"
      description: "Comprehensive card game rules from pagat.com"
      
    - name: "project_gutenberg"
      type: "books"
      search_terms: ["Hoyle", "card games", "poker", "bridge"]
      
    - name: "strategy_guides"
      type: "curated_list"
      description: "Hand-selected strategy articles and books"

  synthetic:
    generator: "claude-api"
    budget: "$100-300"
    tasks:
      - task: "game_state_analysis"
        description: "Given a game state, explain optimal play"
        volume: 5000
        
      - task: "rules_explanation"
        description: "Explain rules at beginner/intermediate/advanced levels"
        volume: 5000
        
      - task: "strategy_comparison"
        description: "Compare two possible plays and explain tradeoff"
        volume: 3000
        
      - task: "quiz_generation"
        description: "Generate quiz questions about rules and strategy"
        volume: 2000

  training_mix:
    general_text: 30    # general reasoning and language
    domain_rules: 25    # rules and reference material
    domain_strategy: 20 # strategy and analysis
    domain_qa: 15       # questions and answers
    synthetic: 10       # generated examples

# Step 2: What validates correctness?
validators:
  - name: "rules_legality"
    type: "custom_python"
    description: "Check if suggested play is legal given game state and rules"
    
  - name: "probability_check"
    type: "tool"
    tool: "probability_calculator"
    description: "Verify probability claims are mathematically correct"
    
  - name: "json_format"
    type: "schema"
    schema_file: "schemas/game_state.json"
    description: "Verify structured outputs match expected schema"

# Step 3: What does "good" look like?
evals:
  - name: "rules_accuracy"
    size: 200
    type: "factual"
    description: "Does the model state rules correctly?"
    
  - name: "strategy_quality"
    size: 100
    type: "expert_rated"
    description: "Are strategy suggestions reasonable?"
    
  - name: "explanation_clarity"
    size: 100
    type: "human_rated"
    description: "Are explanations clear and helpful?"
    
  - name: "game_state_parsing"
    size: 100
    type: "automated"
    description: "Can the model correctly parse and reason about game states?"

# Step 4: What state do we track?
memory:
  schema:
    players:
      - "player_id, name, skill_level, preferred_games"
    game_history:
      - "game_id, game_type, players, result, key_decisions, date"
    weak_areas:
      - "player_id, game_type, concept, error_count, last_seen"
    known_games:
      - "player_id, game_type, familiarity_level, rules_version"

# Step 5: What tools help?
tools:
  - name: "probability_calculator"
    description: "Calculate odds for card combinations, pot odds, expected value"
    
  - name: "game_simulator"
    description: "Simulate random hands/deals to verify strategy claims"
    
  - name: "rules_lookup"
    description: "RAG retrieval specifically over rules corpus"
    
  - name: "hand_evaluator"
    description: "Evaluate and rank poker/bridge hands"

# Step 6: What constrained output schemas?
task_schemas:
  - name: "play_recommendation"
    schema:
      game_state: "object"
      recommended_action: "enum[fold, call, raise, play_card, draw, pass]"
      reasoning: "string"
      confidence: "float[0-1]"
      alternatives: "array[object]"
      
  - name: "rules_explanation"
    schema:
      game: "string"
      topic: "string"
      level: "enum[beginner, intermediate, advanced]"
      explanation: "string"
      examples: "array[string]"

# Step 7: What DPO preference signals?
dpo:
  pair_sources:
    - "validator_pass_vs_fail"
    - "detailed_explanation_vs_terse"
    - "correct_probability_vs_wrong"
    - "rule_cited_vs_uncited"
  target_pairs: 500

# Step 8: Budget tier recommendation
tier_recommendation:
  assessment: "moderate data, some validators, partially checkable"
  recommended: "$1K (1B model)"
  reasoning: "Not enough domain data for compute-optimal 3B. Strong enough validators for self-improvement loop. RAG covers knowledge gaps."
  laptop_viable: true
  cloud_1k_viable: true
  cloud_10k_viable: "possible but data-limited — would need heavy general text mixing"
```

---

## 68. Data Strategy by Availability Tier

### Abundant data domains (1B+ tokens available)

```text
Strategy: Standard pipeline
  Download → filter → deduplicate → quality score → train

Training mix:
  60-70% domain data
  20-30% general text
  10% synthetic

Model size: Scale to budget ($10K → 3B is viable)
RAG role: Supplementary (model knows a lot natively)
```

### Moderate data domains (100M-1B tokens)

```text
Strategy: Augmented pipeline
  Download real data → generate synthetic to fill gaps → mix with general

Training mix:
  30-40% domain data (repeated 2-5x is okay)
  30% general text (provides reasoning/language ability)
  20% synthetic domain data
  10% related domains

Model size: 1B sweet spot ($1K tier)
RAG role: Primary knowledge source (model handles language, RAG handles facts)
```

### Scarce data domains (10M-100M tokens)

```text
Strategy: Synthetic-first pipeline
  Small real corpus → heavy synthetic generation → validate all synthetic
  → RAG indexes everything → model is mainly an interface

Training mix:
  20% domain data (repeated heavily)
  40% general text (most of the language learning)
  30% synthetic domain data
  10% instruction following

Model size: 300M-1B (laptop to $1K tier)
RAG role: Dominant (80%+ of domain knowledge is in retrieval)
```

### Very scarce / proprietary domains (<10M tokens)

```text
Strategy: RAG-first, model-light
  Index all available documents → use a general small model
  → LoRA adapt to domain formatting → RAG does all heavy lifting

Training: Possibly skip domain pretraining entirely
  Use a general 1B-3B model off the shelf
  LoRA fine-tune on whatever domain examples exist
  RAG provides all domain knowledge

Model size: Use existing open-source model + LoRA
RAG role: The entire knowledge system
```

---

## 69. Synthetic Data Generation for Any Domain

### The universal generation loop

For any domain, synthetic data follows the same pattern:

```text
1. Define task types for the domain
2. Generate examples using frontier model (Claude API)
3. Validate with domain validators (if available)
4. Filter for quality (remove duplicates, obvious errors)
5. Format for training
```

### Estimating synthetic data costs

| Volume | Claude API cost (approximate) |
|---|---|
| 1K examples | $10-30 |
| 5K examples | $50-150 |
| 10K examples | $100-300 |
| 50K examples | $500-1,500 |

For most moderate-data domains, $100-300 of API calls generates enough synthetic data to meaningfully supplement the real corpus.

### Quality control for synthetic data

```text
If domain has validators:
  Generate → validate → keep only passing examples
  
If domain has no validators:
  Generate → deduplicate → sample 100 → manually review
  → estimate error rate → if <10%, use the batch
  → if >10%, improve the prompt and regenerate
```

---

## 70. Validator Discovery Checklist

When entering a new domain, ask:

```text
□ Is there a compiler or interpreter?        (code domains)
□ Is there a rules engine?                    (games, math, logic)
□ Is there a calculator to verify numbers?    (finance, science, math)
□ Is there a format schema?                   (any structured output)
□ Is there a test runner?                     (code domains)
□ Is there a linter or style checker?         (code, writing)
□ Is there a spell/grammar checker?           (language domains)
□ Is there a simulator?                       (games, physics, finance)
□ Is there a reference database to check against? (trivia, rules, facts)
□ Can outputs be executed and observed?       (code, recipes, instructions)
□ Can a human quickly judge quality? (<5 sec) (most domains)
□ Can another model judge quality?            (meta-evaluation)
```

Every "yes" is a validator you can add to the pipeline. More validators = stronger hybrid system = more effective self-improvement loop.

Domains with 0-1 validators can still work but lean heavily on RAG + self-consistency rather than generate-then-validate loops.

---

## 71. Framework Implementation Roadmap

### Phase 1: Build React/TypeScript instance (the current plan)

This is the worked example. Everything is built and validated here first.

### Phase 2: Extract domain-agnostic components

After the React/TS system works:

```text
Separate:
  core/           → training loop, model, tokenizer, inference
  hybrid/         → RAG, memory, tools, routing, caching (all generic)
  online/         → self-distillation, online learning (generic)
  evals/          → eval harness runner (generic)
  
  domains/
    react-ts/     → data pipeline, validators, evals, config
    card-games/   → data pipeline, validators, evals, config (second instance)
```

### Phase 3: Build second domain instance

Pick a domain with different characteristics (moderate data, different validators) to prove the framework generalizes. Card games, chess, cooking, or personal finance would each test different aspects.

### Phase 4: Document the framework

The book's final section: "How to build your own domain instance in a weekend."

```text
Day 1:
  Morning: Write domain config file
  Afternoon: Gather and filter data sources, generate synthetic data

Day 2:
  Morning: Build domain validators, create eval set
  Afternoon: LoRA fine-tune on domain data, run evals, iterate
```

This assumes the framework infrastructure is already built from Phase 1.

---

# Part XVIII — Training and System Dashboard

## 72. Dashboard Overview

A web dashboard that provides real-time visibility into every phase of the project: training runs, experiments, hybrid system performance, and gate/milestone status. Useful for the builder (sanity checking runs, catching problems early) and for the audience (build-in-public transparency).

### Tech stack (keep it simple)

```text
Backend:  Python + FastAPI (or Flask)
Frontend: React + TypeScript (dogfood your own domain)
Database: SQLite (already in the stack for memory/caching)
Charts:   Recharts or Chart.js
Updates:  Polling or simple WebSocket for live training
Hosting:  Local during development, optional deploy for public dashboard
```

### Dashboard pages

```text
1. Run Overview        — what's running now, what's queued, what's done
2. Training Live View  — loss curves, samples, speed, ETA
3. Experiment Compare  — side-by-side results from ablation runs
4. Eval Results        — scores across eval sets, trends over time
5. Hybrid System       — module activity, latency, fire rates
6. Gate Status         — milestone checklist, go/no-go for next phase
7. Online Learning     — self-improvement metrics over time
8. Cost Tracker        — spend vs budget across all cloud runs
```

---

## 73. Page 1: Run Overview

### What it shows

A table of all runs — past, active, and planned:

```text
┌──────────┬─────────┬────────┬───────────┬──────────┬────────┬──────────┐
│ Run ID   │ Model   │ Params │ Tokens    │ Hardware │ Status │ Duration │
├──────────┼─────────┼────────┼───────────┼──────────┼────────┼──────────┤
│ run-001  │ char-v1 │ 5M     │ 50M       │ Laptop   │ ✓ Done │ 12m      │
│ run-002  │ tok-v1  │ 10M    │ 200M      │ Laptop   │ ✓ Done │ 3h 22m   │
│ run-003  │ tok-v2  │ 10M    │ 200M      │ A100     │ ✓ Done │ 14m      │
│ run-004  │ mix-a   │ 30M    │ 600M      │ A100     │ ● Live │ 28m/45m  │
│ run-005  │ mix-b   │ 30M    │ 600M      │ A100     │ ○ Queue│ —        │
│ run-006  │ mix-c   │ 30M    │ 600M      │ A100     │ ○ Queue│ —        │
└──────────┴─────────┴────────┴───────────┴──────────┴────────┴──────────┘
```

### Run detail card (click to expand)

```text
Run: run-004 "mix-a"
────────────────────────────────────────
Model:          30M params
Architecture:   6 layers, 6 heads, 384 embd
Tokenizer:      bpe-8k-v2
Dataset:        react-ts-filtered-v3
Data mix:       70% React/TS code, 20% docs, 10% synthetic
Tokens:         600M
Context length: 512
Batch size:     64
Learning rate:  3e-4 (cosine decay)
Hardware:       1x A100 80GB
Started:        2026-06-15 14:22 UTC
ETA:            ~17 minutes remaining

Purpose:        Data mix ablation — compare React-heavy mix
Compare with:   run-005 (balanced mix), run-006 (docs-heavy mix)
```

### Data logged per run (written by training script)

```python
# Append to SQLite every N steps
{
    "run_id": "run-004",
    "step": 1500,
    "timestamp": "2026-06-15T14:35:22Z",
    "train_loss": 3.42,
    "val_loss": 3.58,
    "learning_rate": 2.8e-4,
    "tokens_seen": 96_000_000,
    "tokens_per_second": 52_400,
    "gpu_utilization": 0.87,
    "gpu_memory_used_gb": 42.3,
    "grad_norm": 1.2,
    "sample_output": "function Button({ onClick, children }: {...",
}
```

---

## 74. Page 2: Training Live View

### Real-time charts (update every N steps)

**Loss curves (primary)**

```text
Loss
4.0 │╲
    │ ╲
3.5 │  ╲___
    │      ╲___
3.0 │          ╲____
    │               ╲_______  ← train loss
2.5 │          ╲__________    ← val loss (should track, with gap)
    │
2.0 │
    └──────────────────────────
    0    500   1000  1500  2000  steps
```

**Key things to watch for:**
- Train/val gap growing → overfitting
- Loss plateau → learning rate too low or data exhausted
- Loss spike → learning rate too high or bad data batch
- Val loss rising while train falls → stop training

**Tokens per second (throughput)**

```text
Tokens/sec
60K │          ___________
    │    _____╱
50K │___╱                    ← should stabilize after warmup
    │
40K │
    └──────────────────────
    0    500   1000  1500  steps
```

**Learning rate schedule**

```text
LR
3e-4 │____
     │    ╲
2e-4 │     ╲
     │      ╲
1e-4 │       ╲____
     │             ╲____
0    │                  ╲___
     └──────────────────────
     0    500   1000  1500  steps
```

**Live sample output (update every 100 steps)**

```text
Step 200:  "functon Buton({ onClck }) { retrn <buton"
Step 500:  "function Button({ onClick }) { return <button onClick"
Step 1000: "function Button({ onClick, children }: ButtonProps) {
             return <button onClick={onClick}>{children}</button>"
Step 1500: "export function Button({ onClick, children, variant = 'primary'
             }: ButtonProps) {\n  return (\n    <button\n      className={..."
```

This is the most satisfying part of the dashboard — watching the model go from gibberish to plausible code.

### Alert banners

```text
⚠️  Val loss hasn't improved in 200 steps — possible plateau
⚠️  Train/val gap exceeding 0.3 — possible overfitting
⚠️  GPU utilization dropped below 70% — check data pipeline
🔴  Loss spike detected at step 1247 — investigate
✅  Checkpoint saved at step 1500
```

---

## 75. Page 3: Experiment Compare

### Side-by-side ablation comparison

```text
Experiment: Data Mix Ablation (30M model, 600M tokens each)
──────────────────────────────────────────────────────────

                    run-004      run-005      run-006
                    React-heavy  Balanced     Docs-heavy
                    ──────────   ──────────   ──────────
Final train loss    2.31         2.45         2.52
Final val loss      2.48         2.54         2.61
Best val loss       2.46         2.52         2.58
Steps to best       1,820        1,650        1,540

Eval: TS compiles   72%          68%          61%
Eval: Tests pass    45%          42%          38%
Eval: Code quality  3.1/5        3.0/5        2.8/5

Repetition rate     4.2%         5.1%         6.3%
Avg output length   142 tok      128 tok      119 tok

Winner: run-004 (React-heavy mix)
Reason: Best on all code-specific evals, lowest repetition
```

### Overlay loss curves from multiple runs

```text
Loss
3.5 │╲
    │ ╲ ╲
3.0 │  ╲  ╲
    │   ╲   ╲───── run-006 (docs-heavy)
2.5 │    ╲____╲─── run-005 (balanced)
    │     ╲_______ run-004 (react-heavy) ← winner
2.0 │
    └──────────────────────────
    0    500   1000  1500  2000
```

### Tokenizer comparison view

```text
Tokenizer Comparison (10M model, 200M tokens each)
──────────────────────────────────────────────────

                    bpe-4k    bpe-8k    bpe-16k
                    ────────  ────────  ────────
Avg tokens/line     12.3      8.7       6.9
Compression ratio   3.2x      4.1x      4.8x
Weird splits        14%       6%        3%
Val loss            3.12      2.94      2.88
Training speed      48K t/s   42K t/s   38K t/s

Winner: bpe-8k (best balance of compression and speed)
```

---

## 76. Page 4: Eval Results

### Eval scorecard (current best model)

```text
Model: 3B-v1 + instruction tune + DPO
──────────────────────────────────────

Category              Score    Trend   Gate
────────────────────  ───────  ──────  ────────
TypeScript compiles   94%      ↑ +3%   ✅ Pass
Tests pass            71%      ↑ +5%   ✅ Pass
ESLint clean          88%      ↗ +1%   ✅ Pass
Component renders     82%      ↑ +4%   ✅ Pass
Instruction following 79%      ↗ +2%   ✅ Pass
Tool call validity    96%      → 0%    ✅ Pass
Code quality (human)  3.8/5    ↑ +0.3  ✅ Pass
Repetition rate       2.1%     ↓ -1%   ✅ Pass
Hallucination rate    8%       ↓ -3%   ✅ Pass
```

### Eval trends over model versions

```text
TypeScript compiles (%)
100│
 90│                              ● 3B+DPO
 80│                    ● 3B    ●╱3B+IT
 70│            ● 1B  ●╱1B+IT
 60│     ● 300M╱
 50│ ● 100M
 40│
   └──────────────────────────────────
    100M  300M   1B   1B+IT  3B  3B+IT 3B+DPO
```

### Per-prompt eval detail (expandable)

```text
Prompt: "Add TypeScript types to this component"
──────────────────────────────────────────────

Input:  function UserCard({ user, onEdit }) { ... }

Expected: interface User { id: string; name: string; email: string }
          interface UserCardProps { user: User; onEdit: (id: string) => void }
          function UserCard({ user, onEdit }: UserCardProps) { ... }

Model output: [actual output here]

TS compiles: ✅
Types correct: ✅ (3/3 props typed correctly)
Idiomatic: 4/5 (used interface instead of type — acceptable)
```

---

## 77. Page 5: Hybrid System Monitor

### Module activity (real-time when system is running)

```text
Module              Fire rate   Avg latency   Last hour   Status
──────────────────  ──────────  ────────────  ──────────  ──────
Router              100%        8ms           342 calls   ✅
Cache               34% hit     2ms           116 hits    ✅
RAG retrieval       52%         145ms         178 calls   ✅
Query rewriting     52%         95ms          178 calls   ✅
Constrained decode  100%        12ms          342 calls   ✅
Validator (TS)      89%         340ms         304 calls   ✅
Validator (ESLint)  89%         180ms         304 calls   ✅
Self-consistency    18%         480ms         62 calls    ✅
CoT scratchpad      24%         110ms         82 calls    ✅
Confidence check    100%        5ms           342 calls   ✅
Verifier loop       8%          890ms         27 calls    ✅
Memory read         41%         12ms          140 calls   ✅
Memory write        22%         8ms           75 calls    ✅
Planner             6%          950ms         21 calls    ✅
```

### Routing distribution

```text
Tier 0 (cache hit):           34% ████████░░░░░░░░░░░░
Tier 1 (direct generation):   28% ██████░░░░░░░░░░░░░░
Tier 2 (RAG-augmented):       24% █████░░░░░░░░░░░░░░░
Tier 3 (full pipeline):       14% ███░░░░░░░░░░░░░░░░░
```

### Validator pass/fail breakdown

```text
                    Pass    Fail→Repair  Fail→Reject
TypeScript compile  81%     14%          5%
ESLint              89%     9%           2%
Test runner         68%     22%          10%
Format/schema       97%     3%           0%
Security check      99%     1%           0%
```

### Module latency waterfall (per-query view)

```text
Query: "Add loading state to this data fetch hook"

Router:              ████ 8ms
Cache (miss):        █ 2ms
Query rewrite:       ████████████ 95ms
RAG retrieval:       ██████████████████ 145ms
LoRA swap:           ██ 4ms
Generation:          █████████████████████████ 210ms
Constrained decode:  ██ 12ms
TS compile check:    ████████████████████████████████████████ 340ms
ESLint check:        ██████████████████████ 180ms
Confidence:          █ 5ms
Memory write:        █ 8ms
                     ─────────────────────────────────────────
Total:               1,009ms

Tier: 2 (RAG-augmented)
Result: Pass (no repair needed)
```

---

## 78. Page 6: Gate Status

### Milestone tracker

```text
Part I — Laptop Track
══════════════════════

  ✅ Milestone 1: Pipeline Works
     ├── ✅ Training loss decreases
     ├── ✅ Samples improve
     └── ✅ Checkpoint resume works
     Completed: 2026-06-12

  ✅ Milestone 2: Tokenizer Works
     ├── ✅ Tokenization is sane
     ├── ✅ Val loss improves over char model
     └── ✅ Generation quality improves
     Completed: 2026-06-14

  ✅ Milestone 3: Data Quality Matters
     ├── ✅ Filtered data improves val loss
     └── ✅ Can explain which data choices helped
     Completed: 2026-06-15

  ● Milestone 4: Synthetic Data Helps          ← CURRENT
     ├── ✅ Domain eval improves
     ├── ○ Format following improves            ← in progress
     └── ○ Hallucinations stable
     Started: 2026-06-16

  ○ Milestone 5: RAG Improves Answers
  ○ Milestone 6: Memory Improves Personalization
  ○ Milestone 7: Tools Improve Reasoning

Part VIII — Cloud Track
═══════════════════════

  ○ Gate A: 100M pilot
  ○ Gate B: 300M ablation
  ○ Gate C: 1B run
  ○ Gate D: 3B run
```

### Anti-waste checklist (before cloud runs)

```text
Pre-Cloud Checklist
═══════════════════

  ✅ Tokenizer finalized
  ✅ Data cleaned/deduped
  ✅ Validation set fixed
  ✅ Eval harness works
  ○  Checkpoint resume tested (on cloud)
  ○  Throughput benchmarked (on cloud)
  ✅ Sample generation automated
  ✅ Learning rate tested on smaller model
  ✅ Data mix tested on smaller model
  ○  Cloud storage and download path tested
  ○  Failure/interruption recovery tested

  Status: 7/11 complete — NOT ready for cloud
  Blocking: need to test cloud infra first (Gate A: 100M pilot)
```

---

## 79. Page 7: Online Learning Monitor

### Self-improvement over time

```text
Training buffer status
══════════════════════
Total interactions:     2,847
Buffer size:            1,923 verified examples
  Validator pass:       1,644 (85%)
  Validator repair:     279 (15%)
DPO negative buffer:    184 rejected pairs
Gold corrections:       12 (user-provided)

LoRA updates:           4 deployed
Last update:            2026-08-03 (482 new examples)
Next update:            ~350 more interactions
```

### Accuracy trend

```text
Domain eval accuracy (%)
100│
 90│                              ● After LoRA v4
 85│                    ● v3   ●╱
 80│            ● v2  ●╱
 75│     ● v1  ╱
 70│ ● base ●╱
   └──────────────────────────────────
    Deploy  Wk2   Wk4   Wk6   Wk8   Wk10
    
    +12% accuracy over 10 weeks of usage
```

### Scaffolding reduction trend

```text
Module fire rate over time (%)
100│──────────── Constrained decoding (always on — no change)
 90│
 80│╲
 70│ ╲_________ Validator (decreasing — model learning rules)
 60│
 50│──────────── RAG (stable — knowledge doesn't internalize)
 40│
 30│╲
 20│ ╲___╲_____ Self-consistency (decreasing — first attempts improving)
 10│      ╲____ Verifier loop (rare — model self-corrects more)
   └──────────────────────────────────
    Wk1   Wk3   Wk5   Wk7   Wk9
```

### Self-distillation ablation results

```text
Accuracy WITHOUT each module (measures internalization)
══════════════════════════════════════════════════════

Module disabled    Baseline    After 3 distillation passes    Δ
─────────────────  ──────────  ─────────────────────────────  ──
Without RAG        61%         74%                            +13%
Without validator  78%         88%                            +10%
Without tools      70%         75%                            +5%
Without CoT        72%         79%                            +7%

Interpretation: model has internalized significant RAG knowledge
and validator rules. Tool internalization is weak (expected —
exact computation doesn't memorize well).
```

---

## 80. Page 8: Cost Tracker

### Budget overview

```text
Budget: $10,000
════════════════

Spent        Remaining    Burn rate
$3,247       $6,753       On track
████████░░░░░░░░░░░░ 32%

Breakdown:
  Cloud compute:     $2,890  (89%)
  API calls (synth): $245    (8%)
  Storage/transfer:  $112    (3%)
```

### Spend by phase

```text
Phase               Budget    Spent     Status
──────────────────  ────────  ────────  ────────
Laptop experiments  $50       $38       ✅ Done
100M pilot          $150      $142      ✅ Done
300M ablations      $1,500    $1,380    ✅ Done
Synthetic data API  $300      $245      ✅ Done
1B main run         $1,500    $1,210    ✅ Done
3B final run        $4,000    $0        ○ Not started
Post-training       $1,000    $120      ● In progress
Buffer/reruns       $1,500    $112      —
──────────────────  ────────  ────────
Total               $10,000   $3,247
```

### Cost per run

```text
Run              Params  Tokens  Hours   GPU      Cost
───────────────  ──────  ──────  ──────  ───────  ──────
laptop-chars     5M      50M     0.2h    Laptop   $0
laptop-tok       10M     200M    0.3h    Laptop   $0
cloud-pilot-1    100M    1B      3.2h    1×A100   $3.20
cloud-pilot-2    100M    2B      5.8h    1×A100   $5.80
ablation-mix-a   300M    5B      11.4h   1×A100   $11.40
ablation-mix-b   300M    5B      11.2h   1×A100   $11.20
ablation-mix-c   300M    5B      11.6h   1×A100   $11.60
ablation-tok-a   300M    5B      11.3h   1×A100   $11.30
ablation-tok-b   300M    5B      10.9h   1×A100   $10.90
main-1b          1B      15B     82h     2×A100   $164.00
```

---

## 81. Implementation Notes

### Logging from training scripts

Add a few lines to the training loop to write to SQLite:

```python
import sqlite3
import json
import time

db = sqlite3.connect("dashboard.db")

def log_step(run_id, step, train_loss, val_loss, lr, tokens_seen, sample=None):
    db.execute("""
        INSERT INTO training_logs 
        (run_id, step, timestamp, train_loss, val_loss, 
         learning_rate, tokens_seen, sample_output)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (run_id, step, time.time(), train_loss, val_loss,
          lr, tokens_seen, sample))
    db.commit()

# In training loop:
for step in range(num_steps):
    loss = train_step()
    if step % log_every == 0:
        val_loss = evaluate()
        sample = generate_sample()
        log_step(run_id, step, loss, val_loss, lr, tokens_seen, sample)
```

### Logging from hybrid system

Each module logs its own activity:

```python
def log_module(module_name, query_id, fired, latency_ms, result):
    db.execute("""
        INSERT INTO module_logs
        (query_id, module, fired, latency_ms, result, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (query_id, module_name, fired, latency_ms, result, time.time()))
```

### Dashboard update frequency

```text
During training:    every 10 steps (or every 30 seconds)
Hybrid system:      every query (real-time)
Eval results:       after each eval run
Cost tracking:      hourly (poll cloud provider API)
Online learning:    daily summary
```

### Estimated implementation effort

```text
SQLite schema + logging helpers:   2-3 hours
FastAPI backend (read from SQLite): 3-4 hours
React frontend (charts + tables):  8-12 hours
Alert/notification logic:          2-3 hours
Total:                             ~2-3 days

Can start with just the training view (half a day)
and add pages incrementally as each phase begins.
```

---

# Final Architecture Target

The most interesting end-state is not simply a 3B model. It is a 3B-or-smaller model inside a disciplined, self-improving system:

```text
Local/Cloud-Trained Small LLM
+ RAG document memory (with query rewriting)
+ SQLite structured memory
+ Python/code tools
+ domain validators
+ verifier loop
+ task schemas
+ chain-of-thought scratchpad
+ self-consistency sampling
+ constrained decoding
+ confidence routing and abstention
+ LoRA adapters per sub-task
+ DPO preference alignment
+ caching/memoization
+ self-distillation loop (model absorbs scaffolding over time)
+ validator-gated online learning (improves from usage)
+ eval harness (quality floor)
= practical, self-improving niche AI system
```

This is the path most likely to produce something useful, teach the underlying mechanics of LLMs, and demonstrate a genuinely novel approach to making small models better without simply throwing more compute at the problem.
