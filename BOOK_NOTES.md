# Book Notes

Interesting observations, learnings, and moments from building the project that could be useful for the book.

---

## Phase 1: Pipeline Setup

### First training run (834K char model on Shakespeare)
- 834K params, 13 minutes on MacBook (MPS), 103K tokens/sec
- Loss went from ~4.5 → 1.13 (train), best val loss 1.54
- Sample progression is the most satisfying thing to watch:
  - Step 250: `"heshes ie ndare hfof asphe"` — gibberish
  - Step 500: `"RIAS:\nAn lee the lit vereer"` — learning structure
  - Step 1000: `"CORIOLANUS:\nPray me this fear"` — real character names
  - Step 2000: `"SICINIUS:\nThere is for claim like of their life?"` — grammatical
  - Step 5000: Full Shakespeare-ish dialogue with stage directions
- Model started overfitting around step 3000 (val loss plateaued while train kept dropping) — expected with only 1MB of data
- The whole thing felt surprisingly simple to get working — nanoGPT-style code is <1000 lines

### Python output buffering gotcha
- First run: couldn't see any output during training because Python buffers stdout when not connected to a terminal
- Fix: force unbuffered output with `os.environ["PYTHONUNBUFFERED"] = "1"` and overriding print with `flush=True`
- Book lesson: always set up unbuffered output in training scripts. Waiting 30+ minutes with no output wondering if it's working is a terrible experience.

### 5M model results — the overfitting lesson
- 4.8M params, same Shakespeare data, 43 minutes on laptop
- Throughput: 30K tok/s (vs 103K for 834K — 3.4x slower for 5.8x more params)
- Best val loss: 1.54 — SAME as the 834K model!
- But then val loss ballooned to 3.09 while train loss dropped to 0.34
- The model memorized Shakespeare rather than learning patterns
- This perfectly demonstrates the plan's principle: data quantity matters more than model size
- The 834K model actually generalizes better because it can't memorize as much
- This is the clearest possible motivation for getting more data before scaling up
- Great book content: "I made the model 6x bigger and it got WORSE"

### React/TypeScript data collection
- Cloned 15 open-source React/TS repos (shallow clone, no history)
- 19,595 TypeScript/TSX files extracted
- 52.3MB corpus (vs 1.1MB Shakespeare — ~50x more data)
- Top sources: Cal.com (4,024 files), Mantine (3,946), shadcn (3,376), Next.js (2,705)
- Key decision: clone curated repos rather than downloading The Stack — faster, no auth needed, guaranteed high quality
- Raw repos were 830MB — deleted after extraction to save disk space. Corpus file has everything needed.
- Filtering mattered: skipped node_modules, dist, test files, .d.ts files, minified files, files <50 chars or >50K chars

### Laptop sleep issue
- Training runs will fail if laptop sleeps mid-run
- macOS setting: "Prevent automatic sleeping on power adapter when the display is off"
- Alternative: `caffeinate -s &` command
- Book lesson: mention this early — readers will lose runs to this

### Model file sizes and formats
- Checkpoints include optimizer state (~3x model weights) — needed for resuming training only
- For inference you only need the weights
- Final deployment path: PyTorch → HuggingFace format → GGUF (quantized) → Ollama
- 3B model: 12GB full → ~2GB at 4-bit quantization → runs on any laptop

### "How long does training take?" is the first question everyone asks
- People want concrete numbers, not formulas
- Built a table early: model size × hardware → wall clock time
- This should be prominently placed in the book

### React/TS training — the model writes code (in progress)
- 5M char model on 52MB React/TS corpus
- By step 750: generating JSX closing tags, className attributes, component names
- By step 1000: `return data`, `<Arrow e="Contectors">` — code-shaped but nonsensical
- By step 2500: `<Button variant="outline">Outline</Button>` — valid JSX!
- Val loss still dropping at step 2500 (0.75), no overfitting — 50x more data is working
- The progression from gibberish to valid JSX in ~20 minutes is very satisfying to watch
- Great visual for the book: side-by-side samples at each step showing code emerging

### Data quality analysis
- Analyzed 19,595 files: average quality 77.5/100
- Only 0.7% very low quality — curating repos upfront paid off
- 990 exact duplicates found, mostly barrel/index files
- Key insight for book: "when you pick good sources, you need less filtering"
- The filtering only removed 0.1% of data by size — the real quality lever was repo selection

### BPE tokenizer trained
- 8,000 vocab BPE tokenizer trained on React/TS corpus in seconds
- Learned domain-specific tokens: [useState], [onClick], [className], [export], [interface]
- Compression: 3.2 chars/token → 3.2x more context at same block size
- `import React from "react"` → 6 tokens instead of 25 characters
- `export default function Home() {` → 6 tokens instead of 32 characters
- This is the single biggest upgrade before scaling model size

### Dashboard is proving its value
- Seeing loss curves in real time during training is much better than reading log files
- The analysis panel catches overfitting and explains it in plain English
- Training planner helps predict time/risk before starting a run
- Building it in React/TS = dogfooding the domain

### Cloud GPU: 15x faster than laptop for $1.49/hr
- RunPod A100 SXM: 308K tok/s vs laptop's 21K tok/s
- 500-step test run cost ~$0.10 and took 26 seconds
- Estimated cost for full 3B training: ~$81 (way under the $10K budget)
- Our cost estimates in the plan were very conservative — good news
- Main gotcha: forgetting to shut down the pod (set a timer!)
- GitHub Releases for hosting training data (52MB corpus) works great — one `wget` on cloud machine

### Hybrid modules are mostly domain-agnostic
- Built all core hybrid modules in ~1 hour: memory, verifier, router, cache, pipeline
- Key insight: only the validators are domain-specific. Everything else is generic.
- The pipeline orchestrator (hybrid/pipeline.py) connects everything into one `pipeline.run(query)` call
- This validates the domain framework idea — swapping domains is mostly swapping validators and RAG corpus
- Book lesson: show the generic modules first, then plug in domain-specific validators

### The verifier loop is deceptively simple
- Just generate → validate → feed errors back → regenerate
- But it's powerful because it turns any validator into a free training signal
- The loop is ~100 lines of code, domain-agnostic
- Multiple validators compose naturally — each catches different error categories

### Routing saves latency on easy queries
- Not every query needs RAG + validation + memory + verifier
- Router estimates complexity from surface signals and routes to appropriate tier
- Cache hit = instant. Simple query = direct generation. Complex = full pipeline.
- This matters for UX — most queries should feel fast

### 30M cloud model — first real code generation
- 30M BPE model on React/TS, 5000 steps on A100, 42 minutes, val loss 1.48
- Generates real import paths, TypeScript decorators, tRPC patterns, proper class syntax
- Best sample: complete NestJS-style API input class with decorators and validators
- SVG path data and translation files appeared in samples — added to filter list for next corpus
- Repetition still a problem (same token loops) — needs larger model or repetition penalty

### LoRA instruction tuning on 30M — surprisingly works
- 115 instruction pairs (15 hand-written + 100 Claude-generated, ~$1)
- LoRA fine-tune: 27 seconds on laptop, loss dropped 1.19 → 0.21
- "Write a custom hook for debouncing" → generated a hook with correct structure (useState, useEffect, return)
- Code has bugs but intent and structure are right — this is exactly what validators fix
- "Add TypeScript types" → understands the task but ignores the specific input code, generates generic interfaces
- Key insight: 30M can learn the instruction→code pattern, but lacks capacity to precisely follow input
- This motivates both scaling up AND the hybrid approach: model gets intent, validators fix execution
- Great book content: show the progression from "can't follow instructions at all" to "gets the idea but sloppy"

### Memory enables personalization without retraining
- SQLite stores interaction history, error patterns, success rates
- Compact context summary injected into prompts (~500 chars)
- The system "learns" your patterns without any model updates
- Over time, the memory + online learning create a personalized tool

---

## Planning Phase Insights

### The $0 / $1K / $10K framing
- Came from a conversation about what each budget tier gets you
- The key insight: correctness plateaus early (hybrid modules handle it), scale buys naturalness/fluency
- "Correct but stiff" vs "correct and natural" is the real difference between tiers
- This framing makes the book accessible — everyone understands budget tiers

### 17 hybrid modules — interaction effects are the novel contribution
- Individual modules are well-studied
- Nobody has measured combined effects on small models
- The interaction matrix (super-additive, redundant, unknown) is genuinely unexplored
- "Effective scale" question: what size raw model does a 3B + 17 modules perform like?

### Self-improvement loop as the book's climax
- Self-distillation (model trains on its own hybrid system's outputs)
- Validator-gated online learning (system improves from usage)
- These are genuinely novel in combination at this scale
- Narrative: "the scaffolding teaches the model to need less scaffolding"

### Domain generalization framework
- Started as React/TS specific, evolved into a configurable framework
- Key insight: data availability determines which tier makes sense
- Domains with strong validators (code, math, games) benefit most from hybrid approach
- YAML config file for spinning up new domain instances

### Book title candidates
- "The Hybrid LLM: How to Make a Tiny Model Punch Above Its Weight" — current favorite
- "Own Your AI" — appeals to independence/privacy motivation
- "Small Models, Smart Systems" — clear but slightly dry
