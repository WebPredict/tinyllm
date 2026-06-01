"""Train a 16K vocab BPE tokenizer on the Stack frontend corpus."""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from pathlib import Path

corpus = "data/stack-frontend/corpus.txt"
out_dir = Path("data/stack-frontend/tokenizer")
out_dir.mkdir(parents=True, exist_ok=True)

print("Training 16K vocab tokenizer...")
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
t.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=16000,
    special_tokens=["<|pad|>", "<|endoftext|>", "<|startoftext|>"],
    min_frequency=2,
    show_progress=True,
)
t.train([corpus], trainer)
t.save(str(out_dir / "bpe_16000.json"))
print(f"Done: {t.get_vocab_size()} vocab")
print(f"Saved to: {out_dir / 'bpe_16000.json'}")
