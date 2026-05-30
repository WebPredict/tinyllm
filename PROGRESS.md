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
- [ ] Train BPE tokenizer on React/TS data
- [ ] Train token-level model
- [ ] Compare char vs token model
- [ ] Tokenizer size experiments (4K vs 8K vs 16K)

## Phase 3: Data Quality
- [ ] Data filtering pipeline
- [ ] Deduplication
- [ ] Quality scoring
- [ ] Raw vs filtered comparison

## Phase 4: Synthetic Data
- [ ] Generate synthetic React/TS examples (via Claude API)
- [ ] Validate/filter synthetic data
- [ ] Fine-tune with synthetic data
- [ ] Measure impact

## Phase 5: Hybrid Modules (Core)
- [ ] RAG (retrieval-augmented generation)
- [ ] Structured memory (SQLite)
- [ ] Rules engine / validators (TypeScript compiler, ESLint)
- [ ] Calculator / code interpreter
- [ ] Planner/executor loop
- [ ] Verifier/critic
- [ ] Symbolic task graphs

## Phase 6: Hybrid Modules (Additional)
- [ ] Chain-of-thought / scratchpad
- [ ] Self-consistency (sample + vote)
- [ ] Constrained decoding
- [ ] Confidence estimation / abstention
- [ ] LoRA adapters per sub-task
- [ ] Knowledge distillation framing
- [ ] Routing / cascading
- [ ] Caching / memoization
- [ ] DPO preference alignment
- [ ] Query rewriting for retrieval

## Phase 7: Cloud Training
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
