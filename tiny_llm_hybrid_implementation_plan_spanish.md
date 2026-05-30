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

- Spanish learning / grammar drills
- piano/music theory exercises
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
- Spanish/English educational content if doing SentenceLab-style work

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

For Spanish learning:

- public Spanish grammar explanations
- bilingual sentence pairs
- public-domain Spanish texts
- learner-level reading passages
- generated grammar drills

For coding:

- small permissively licensed GitHub repos
- documentation pages
- coding Q&A
- before/after code diffs

For music/piano:

- public-domain theory texts
- MusicXML examples
- ABC notation
- graded sight-reading exercises
- public-domain folk melodies

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

For Spanish learning:

```text
Model A: general text only
Model B: 70% general + 30% Spanish learning
Model C: 30% general + 70% Spanish learning
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

- grammar correction
- translation pairs
- sentence simplification
- fill-in-the-blank drills
- quiz generation
- explanations of mistakes
- coding refactors
- unit-test generation
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

### Example: Spanish correction dataset

```json
{
  "task": "correct_spanish_sentence",
  "input": "El casa es grande.",
  "output": "La casa es grande.",
  "error_type": "gender_agreement",
  "explanation": "Casa is feminine, so the article should be la."
}
```

### Example: CEFR sentence generation

```json
{
  "task": "generate_sentence",
  "level": "A2",
  "target_grammar": "preterite",
  "target_words": ["ayer", "comprar", "libro"],
  "output": "Ayer compré un libro."
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

### Example: Spanish tutor memory

```json
{
  "known_words": ["casa", "comer", "ayer"],
  "weak_grammar": ["preterite_vs_imperfect", "gender_agreement"],
  "recent_errors": [
    {
      "sentence": "El casa es grande",
      "error_type": "gender_agreement",
      "date": "2026-05-05"
    }
  ],
  "current_level": "A2"
}
```

### Implementation

Use SQLite tables:

```sql
users
skills
mistakes
sessions
exercises
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

- Spanish grammar
- music theory
- arithmetic
- formatting
- JSON validity
- code style
- unit tests

### Example flow

```text
LLM proposes exercise
↓
rules engine validates grammar/level
↓
if invalid, reject or repair
↓
LLM explains final answer
```

### Example Spanish validators

- article/noun gender agreement
- verb tense target present
- target vocabulary included
- sentence length appropriate for level
- no forbidden advanced grammar for A1/A2

### Experiments

| Experiment | Metric |
|---|---|
| LLM-only exercises | error rate |
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
- grammar table generation
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
  "goal": "generate_spanish_drill",
  "level": "A2",
  "tense": "preterite",
  "target_words": ["ayer", "comprar", "libro"],
  "exercise_type": "fill_in_blank"
}
```

Then deterministic software generates/validates the exercise.

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

- Grammar explanations ("why is this wrong?")
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

- Grammar correction (multiple valid corrections, pick most common)
- Exercise generation (generate 5, keep the one that validates)
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
- Generated exercises always match expected schema
- Enum fields only contain allowed values

### Example

```python
# Instead of hoping the model outputs valid JSON:
schema = {
    "tool": ["search_docs", "run_python", "validate_spanish"],
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
├── LoRA: Spanish grammar
├── LoRA: Spanish vocabulary
├── LoRA: coding transforms
├── LoRA: document Q&A
└── LoRA: exercise generation
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
  "reasoning": "The model should first identify the verb tense, then check agreement...",
  "output": "answer",
  "decision_process": "Selected preterite because the sentence contains 'ayer'..."
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
- Validated exercise templates
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

For Spanish tutoring:
```json
{
  "input": "Correct: 'El casa es grande'",
  "chosen": "La casa es grande. 'Casa' is feminine, so use 'la' instead of 'el'.",
  "rejected": "La casa es grande. The article was wrong."
}
```

The "chosen" has a better explanation. The "rejected" is technically correct but less helpful.

### Sources of pairs

- Validator-passing vs validator-failing outputs
- Detailed explanation vs terse explanation
- Correctly-cited vs uncited answers
- Appropriate level vs too-advanced responses

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
User: "what's the deal with ser vs estar"
Rewritten: "differences between ser and estar in Spanish, usage rules"
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

For Spanish:

- grammar correctness
- CEFR level match
- target vocabulary inclusion
- explanation clarity

For coding:

- unit tests pass
- TypeScript compiles
- output diff matches expected transformation

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
| Spanish grammar | 200 | domain correctness |
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
  "instruction": "Generate an A2 Spanish fill-in-the-blank exercise using the preterite.",
  "input": "target words: ayer, comprar, libro",
  "output": "Ayer ___ un libro.\nAnswer: compré"
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
- `validate_spanish`
- `generate_exercise`
- `final_answer`

### Gate

Tool tuning succeeds if:

- valid tool-call JSON rate is high
- correct tool selection improves
- final answer accuracy improves

---

## 44. Domain Fine-Tuning

For a SentenceLab-style model:

Fine-tune on:

- grammar correction
- sentence generation
- CEFR grading
- explanation of mistakes
- targeted drills
- translation/simplification

### Gate

Domain fine-tune succeeds if:

- domain eval improves materially
- outputs remain natural
- model does not overfit to templates

---

# Part X — Example End-State Systems

## 45. Spanish Tutor Hybrid System

```text
User answer
↓
structured memory retrieves weak areas
↓
LLM classifies likely mistake
↓
rules engine validates grammar
↓
RAG retrieves grammar explanation
↓
LLM gives correction/explanation
↓
memory updates skill profile
↓
next exercise generated
```

### Why this is good for small models

- narrow domain
- structured outputs
- validators available
- memory matters
- correctness is checkable

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
- Spanish/code validators
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

## 50. Demo Prompt Set: Spanish Tutor Domain

Run these identical prompts at all three tiers and record the outputs.

### Prompt 1: Grammar explanation (open-ended reasoning)

```text
Explain why "Yo estaba comiendo cuando él llegó" uses imperfect + preterite.
```

Expected progression:

- **Laptop**: RAG retrieves correct grammar rule. Model output is garbled or near-verbatim copy of retrieved chunk. Answer is technically correct but reads like a broken search result.
- **$1K**: Coherent explanation. Hits the key points (ongoing action vs interrupting event). Formulaic, reads like a textbook excerpt.
- **$10K**: Natural explanation with analogy or example. Feels like a patient teacher adapting to your level.

### Prompt 2: Structured exercise generation (constrained output)

```text
Generate an A2 fill-in-the-blank exercise about food vocabulary using preterite tense. Target words: comprar, fruta, mercado.
```

Expected progression:

- **Laptop**: Constrained decoding forces valid JSON schema. Validator confirms grammar. But sentence is boring/repetitive: "Ayer compré una fruta."
- **$1K**: Valid exercise, slightly more creative. Occasionally produces something natural: "Ayer compramos frutas en el mercado."
- **$10K**: Consistently varied and contextual: "Ayer mi hermana compró frutas frescas en el mercado del pueblo."

### Prompt 3: Personalized correction (memory + reasoning)

```text
I keep confusing ser and estar with adjectives. Help me.
```

Expected progression:

- **Laptop**: Retrieves correct RAG chunk about ser/estar. Pastes it back near-verbatim. No personalization from memory despite memory being available — model can't compose the two signals.
- **$1K**: Uses memory to acknowledge recurring issue. Gives structured explanation with examples. Can't improvise natural examples or adapt tone.
- **$10K**: References specific past mistakes from memory. Generates targeted examples using vocabulary you already know. Conversational tone. Adapts explanation complexity to your level.

### Prompt 4: Error correction with explanation

```text
Is this correct? "Ella es cansada porque trabajó mucho."
```

Expected progression:

- **Laptop**: Validator flags ser→estar error. Model attempts explanation but it's incoherent. System returns validator output + RAG rule. Technically correct, robot delivery.
- **$1K**: Identifies error, explains that "cansada" is a temporary state requiring estar. Explanation is correct and readable but short/mechanical.
- **$10K**: Identifies error, explains the ser/estar distinction for states vs traits, gives a contrasting example ("Ella es cansada" would mean she's a tiring person), conversational.

### Prompt 5: Multi-turn conversation (coherence test)

```text
Turn 1: "I want to practice ordering food at a restaurant."
Turn 2: "Can you make it harder? I already know básico vocabulary."
Turn 3: "Wait, what tense should I use for polite requests?"
```

Expected progression:

- **Laptop**: Each turn handled independently. No sense of building on prior turns despite memory. Exercises don't reflect difficulty request. Tense question answered generically.
- **$1K**: Follows the thread with memory. Adjusts difficulty. Tense question answered correctly but doesn't connect it back to the restaurant scenario.
- **$10K**: Smooth multi-turn conversation. Adjusts difficulty naturally. Answers tense question and immediately generates a restaurant example using conditional: "¿Podría traerme la cuenta?"

### Prompt 6: Edge case / unexpected input

```text
"How do you say 'I be vibing' in Spanish lol"
```

Expected progression:

- **Laptop**: Intent classifier confused. Routing breaks or gives irrelevant grammar explanation. No humor handling.
- **$1K**: Attempts translation, probably overly literal. Might produce something like "Estoy vibrando" without acknowledging the slang.
- **$10K**: Handles the register shift. Explains there's no direct equivalent, offers informal options ("Estoy en mi onda," "Estoy relajado"), acknowledges the humor.

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
- **$10K**: More likely to know the answer (Ouagadougou) from pretraining. If it doesn't know, more graceful abstention. But this is still outside the system's intended domain — the *correct* behavior at all tiers is "I don't know, I'm a Spanish tutor."

### Prompt 12: RAG retrieval quality

```text
"What's the difference between por and para?"
```

(Assuming a good grammar reference is in the RAG corpus)

Expected progression:

- **Laptop**: Retrieves correct chunks. Output is essentially the retrieved text with minimal reformatting. Accurate but not personalized or synthesized.
- **$1K**: Retrieves and synthesizes. Produces a structured comparison. May use query rewriting to get better chunks. Coherent but textbook-like.
- **$10K**: Synthesizes across multiple retrieved chunks. Produces a clear explanation with examples. May reference your past confusion (memory). Feels like a tutor who read the textbook and is now explaining it to you specifically.

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
- **Helpfulness**: would a learner/user actually benefit from this?
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

"The hybrid architecture does 80% of the work. A 100M model that can barely form sentences still produces correct exercises because constrained decoding, validators, and RAG carry it. This is the most important lesson: **architecture beats scale for structured tasks.**"

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
  Without validator: 88%  ← model internalized grammar rules
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
  - Balance across grammar topics / task types
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
  actually struggle with (ser/estar, preterite/imperfect).
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
  "input": "Correct: 'Yo soy cansado'",
  "chosen": "Yo estoy cansado. Use 'estar' for temporary states like tiredness.",
  "rejected": "Yo soy cansado is correct because..."
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
(user learns more Spanish, asks harder questions, system's
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
Eval set: 200 domain-specific tasks (Spanish grammar, coding, Q&A)

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
