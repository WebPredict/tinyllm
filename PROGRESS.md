# TinyLLM Build Progress

## Phase 1: Pipeline Setup
- [x] Project structure and venv
- [x] Install PyTorch
- [x] GPT model implementation (model/gpt.py)
- [x] Training loop with logging (training/train.py)
- [x] Download Shakespeare data
- [x] Train 834K char model — pipeline validated (13 min, val loss 1.54)
- [x] Train 5M char model — overfitted on Shakespeare (43 min, val loss 3.09)
- [x] Compare 834K vs 5M results — key learning: data size matters more than model size
- [x] Run evals on both models — 834K wins (perplexity 4.91 vs 22.19)
- [x] Download/prepare React/TS data — 52MB corpus from 15 repos, 19K files
- [x] Build eval harness (evals/eval.py) — perplexity, repetition, structure, prompt completion
- [x] Build basic dashboard — React/TS frontend, loss curves, run comparison, analysis panel
- [x] Train 5M char model on React/TS data — val loss 0.66, no overfitting, generates real JSX
- [x] Run evals on React/TS model — perplexity 1.95, 24/24 prompt pass rate
- [x] Compare React/TS vs Shakespeare — React model vastly better (0.66 vs 1.54 val loss)
- [x] Data quality analysis — 77.5 avg quality, curated repos = minimal filtering needed
- [x] Train BPE tokenizer — 8K vocab, 3.2x compression, domain-specific tokens learned
- [ ] Train 5M BPE model on React/TS data (next)
- [ ] Compare char vs BPE results

## Phase 2: Tokenizer
- [x] Train BPE tokenizer on React/TS data — 8K vocab, 3.2x compression
- [x] Train token-level model — 5M BPE on React/TS (val loss 1.70)
- [ ] Compare char vs BPE results (eval data collected, need comparison)
- [ ] Tokenizer size experiments (4K vs 8K vs 16K)

## Phase 3: Data Quality
- [x] Data filtering pipeline — scripts/filter_data.py
- [x] Deduplication — 990 exact dupes found and removed
- [x] Quality scoring — 77.5 avg quality, scripts/analyze_data.py
- [ ] Raw vs filtered comparison (filtered corpus ready, not yet trained on)

## Phase 4: Synthetic Data
- [x] Generate instruction pairs via Claude API — 115 pairs (15 seed + 100 generated, ~$1)
- [ ] Generate more pairs (500+) for better generalization
- [ ] Validate/filter synthetic data
- [ ] Measure impact

## Phase 4.5: Instruction Tuning (early experiment)
- [x] LoRA fine-tuning implementation — training/finetune_lora.py
- [x] LoRA fine-tune on 30M model — loss 1.19 → 0.21, 27 seconds on laptop
- [x] Test instruction following — model understands intent, output is rough but shows signal
- [ ] Generate more instruction pairs and retrain
- [x] LoRA fine-tune on 100M model — 0.5% trainable, proper freezing
- [ ] Compare instruction-tuned vs base model on eval set
- [ ] Generate more instruction pairs (500+)

## Phase 5: Hybrid Modules (Core)
- [x] RAG (retrieval-augmented generation) — 120K chunks indexed, search working
- [x] Structured memory (SQLite) — hybrid/memory.py
- [x] Rules engine / validators (TypeScript compiler, ESLint) — hybrid/validators.py
- [ ] Calculator / code interpreter
- [ ] Planner/executor loop
- [x] Verifier/critic — hybrid/verifier.py (generic generate→validate→revise loop)
- [ ] Symbolic task graphs

## Phase 6: Hybrid Modules (Additional)
- [ ] Chain-of-thought / scratchpad
- [ ] Self-consistency (sample + vote)
- [ ] Constrained decoding
- [ ] Confidence estimation / abstention
- [ ] LoRA adapters per sub-task
- [ ] Knowledge distillation framing
- [x] Routing / cascading — hybrid/router.py (tier 0-3 routing)
- [x] Caching / memoization — hybrid/cache.py (SQLite-backed, fuzzy matching)
- [ ] DPO preference alignment
- [ ] Query rewriting for retrieval
- [x] Hybrid pipeline — hybrid/pipeline.py (orchestrates all modules)

## Phase 7: Cloud Training
- [x] Cloud infrastructure tested — RunPod A100, 308K tok/s (15x laptop)
- [x] GitHub release for data files — v0.1-data
- [x] 30M Shakespeare cloud run — overfitted as expected (val 3.66), 75K tok/s
- [x] 30M React/TS BPE cloud run — val loss 1.48, perplexity 4.39, generates real code
- [x] Downloaded 30M checkpoint locally (121MB)
- [x] 100M React/TS BPE cloud run — 10K steps on A100, 4.5 hours, ~$7
- [x] Downloaded 100M checkpoint locally (399MB)
- [ ] 1B throughput test on 4x H100 (script ready)
- [ ] 1B full training run
- [ ] 100M pilot run
- [ ] 300M ablation runs
- [ ] 1B main run
- [ ] 3B stretch run (optional)

## Phase 8: Post-Training
- [ ] Instruction tuning
- [ ] Tool-use fine-tuning
- [ ] Domain fine-tuning
- [ ] DPO training pass
- [ ] Security DPO pairs

## Phase 9: Novel Experiments
- [ ] Self-distillation loop
- [ ] Validator-gated online learning
- [ ] Module interaction measurements

## Phase 10: Polish
- [ ] Full dashboard
- [ ] Tiered demo (laptop / $1K / $10K comparison)
- [ ] Quantize to GGUF
- [ ] Run via Ollama
- [ ] Book write-up

## Scripts
- [x] scripts/get_shakespeare.py — download Shakespeare data
- [x] scripts/get_react_data.py — download/prepare React/TS corpus
- [x] scripts/run_first_train.sh — run 834K Shakespeare model
- [x] scripts/run_5m_train.sh — run 5M Shakespeare model
- [x] scripts/run_5m_react.sh — run 5M React/TS model
- [x] scripts/run_chat.sh — interactive chat with latest model
- [x] scripts/run_eval.sh — run eval harness
- [x] scripts/run_dashboard.sh — launch dashboard (API + frontend)
- [x] scripts/kill_training.sh — kill running training process

## Files
```
tinyllm/
├── model/gpt.py               ✅ GPT implementation
├── training/train.py           ✅ Training loop + logging
├── training/train_5m.py        ✅ 5M Shakespeare config
├── training/train_5m_react.py  ✅ 5M React/TS config
├── evals/eval.py               ✅ Eval harness (perplexity, repetition, structure, prompts)
├── dashboard/api.py            ✅ FastAPI backend
├── dashboard/src/App.tsx       ✅ React dashboard frontend
├── data/shakespeare/input.txt  ✅ 1.1MB
├── data/react-ts/corpus.txt    ✅ 52MB, 19K files
├── logs/training.db            ✅ SQLite training + eval logs
├── checkpoints/                ✅ Model checkpoints
├── scripts/                    ✅ Run scripts
├── BOOK_NOTES.md               ✅ Notes for the book
├── PROGRESS.md                 ✅ This file
└── requirements.txt            ✅ Dependencies
```
