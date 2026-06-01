# 1B Training Run Checklist

## Step 1: Launch Pod (5 min)

1. Go to RunPod: https://console.runpod.io
2. Deploy a new pod:
   - Search for **H100** in GPU selector
   - Select **4x H100** (or 4x H100 SXM if available)
   - Keep PyTorch template
   - Add your SSH key if not saved
   - Deploy
3. Wait for pod to start, get SSH connection info

## Step 2: Throughput Test First (~10 min, ~$1)

SSH into the pod, then:

```bash
git clone https://github.com/WebPredict/tinyllm.git
cd tinyllm
pip install tokenizers
bash scripts/cloud_test_1b.sh
```

This tests actual 1B model throughput on your GPUs. Note the numbers:
- tok/s per GPU
- Estimated training times for different token counts

**DECIDE**: Based on the throughput numbers, pick your token target:
- 2B tokens: cheapest, decent results
- 5B tokens: better, moderate cost
- 10B+ tokens: best, need The Stack data

If estimated time is reasonable, proceed. If not, consider fewer steps or smaller block size.

## Step 3: Download Training Data (~10-30 min)

### Option A: Use expanded corpus (quick, limited data)
```bash
mkdir -p data/react-ts-expanded/tokenizer
wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.2-data/corpus.txt -O data/react-ts-expanded/corpus.txt
wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.2-data/bpe_8000.json -O data/react-ts-expanded/tokenizer/bpe_8000.json
```
~30M tokens. Will repeat data many times. OK for testing, not ideal.

### Option B: Download from The Stack (better, needs HuggingFace account)
```bash
pip install datasets huggingface_hub
huggingface-cli login
python scripts/get_stack_data.py --target-gb 3
```
~1B tokens. Much better for 1B model. Takes 10-30 min to download.

## Step 4: Train Tokenizer (if using Stack data) (~2 min)
```bash
# Only needed if you used Option B above
mkdir -p data/stack-frontend/tokenizer
python3 -c "
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
t.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=16000, special_tokens=['<|pad|>','<|endoftext|>','<|startoftext|>'], min_frequency=2, show_progress=True)
t.train(['data/stack-frontend/corpus.txt'], trainer)
t.save('data/stack-frontend/tokenizer/bpe_16000.json')
print('Done')
"
```

## Step 5: Adjust Config (if needed)

Edit the config in `training/train_distributed.py` if you want to change:
- `max_steps`: reduce for shorter/cheaper run
- `block_size`: 1024 is faster, 2048 is better quality
- `batch_size`: reduce if you get out-of-memory errors
- `corpus_file` / `tokenizer_file`: point to your data

If using expanded corpus instead of Stack data:
```bash
# Quick edit on the pod:
sed -i 's|data/stack-frontend/corpus.txt|data/react-ts-expanded/corpus.txt|' training/train_distributed.py
sed -i 's|data/stack-frontend/tokenizer/bpe_16000.json|data/react-ts-expanded/tokenizer/bpe_8000.json|' training/train_distributed.py
```

## Step 6: Start Training

```bash
bash scripts/cloud_run.sh scripts/cloud_1b_react_bpe.sh
```

This runs in the background with nohup. Safe to disconnect.

## Step 7: Monitor

```bash
tail -30 training_output.log        # recent progress
tail -f training_output.log         # watch live
ls -lht checkpoints/cloud_1b_*     # check for saved checkpoints
ps aux | grep python                # verify still running
```

## Step 8: Download Results

When training finishes (or when a checkpoint looks good enough):

```bash
# From your LAPTOP terminal:
scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/tinyllm/checkpoints/cloud_1b_react_bpe.pt ./checkpoints/
```

The final checkpoint will be ~4GB. Step checkpoints are larger (~12GB, includes optimizer).

## Step 9: SHUT DOWN THE POD

Go to RunPod dashboard → Stop the pod.
4x H100 at ~$13/hr = $0.22/min. Don't forget!

## Cost Estimates (will be refined by Step 2)

```
Throughput test:        ~$1
Data download:          free (just time)
Training (estimate):    $78-325 depending on throughput
Total:                  ~$80-330
```

## Troubleshooting

**Out of memory**: Reduce `batch_size` from 8 to 4, or `block_size` from 2048 to 1024

**Training too slow**: Check that all 4 GPUs are being used: `nvidia-smi`

**Connection dropped**: Training continues via nohup. Reconnect and `tail -30 training_output.log`

**Checkpoint too large to download**: Download the final weights-only checkpoint (cloud_1b_react_bpe.pt), not the step checkpoints
