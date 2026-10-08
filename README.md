# UMAY

A bilingual (Turkish–English) language model built from scratch.

*Umay is a protective spirit in Turkic mythology.*

## UMAY-0

A small GPT — a hand-written transformer, built for learning.

- Byte-level BPE tokenizer, vocab 8192, trained on the mixed corpus (`<|endoftext|>` separates documents)
- 6 layers, 6 heads, 384 embedding dim, 256-token context
- 13.88M parameters
- Data: mixed Turkish + English Wikipedia (~120 MB TR + ~80 MB EN), shuffled at document level
- Training: AdamW, linear warmup + cosine decay, bf16 autocast, gradient clipping

## Setup

```
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install tokenizers datasets tqdm numpy matplotlib regex
```

On Windows, if the PowerShell execution policy blocks `.venv\Scripts\activate`, skip activation and
call the venv interpreter directly, as in the commands below.

## Usage

```
.venv\Scripts\python.exe data_download.py   # download + clean Wikipedia -> data/mix.txt (--force to rebuild)
.venv\Scripts\python.exe tokenizer_bpe.py   # train BPE tokenizer, encode corpus -> data/bpe.pt
.venv\Scripts\python.exe train.py           # train -> umay0_bpe.pt
.venv\Scripts\python.exe sample.py          # generate text from the checkpoint
```

The checkpoint stores the model config, so `sample.py` rebuilds the exact architecture it was trained with.

## Results

| Model | Data | Params | Steps | Train time | Val loss | ≈ per char |
|---|---|---|---|---|---|---|
| UMAY-0 (char) | raw | ~10.8M | 5000 | ~3 min | 1.31 / char | 1.31 |
| UMAY-0 (BPE) | raw | 13.88M | 8000 | — | 3.697 / token | 1.107 |
| UMAY-0 (BPE) | cleaned | 13.88M | 8000 | 5 min 45 s | 3.718 / token | 1.100 |

Train time is the training loop on an RTX 5070 Ti (bf16, batch 64, 131M tokens, 2.6 epochs).

On the cleaned corpus (190.3M characters) the BPE tokenizer produces 56.38M tokens, 3.38
characters/token (raw corpus: 56.9M tokens, 3.34). Val loss per token is not comparable across
tokenizers, so the last column divides it by characters/token. The BPE models beat the
character-level one; cleaning barely moves the loss (the validation splits also differ), but it
removed template junk such as `Ankara (; , ;` from the generated text.

## Repository layout

```
data_download.py   corpus download and cleaning
tokenizer_bpe.py   BPE tokenizer training + corpus encoding
model.py           Config dataclass + transformer (UMAY0)
train.py           training loop
sample.py          text generation
experiments/       learning-phase scripts: char tokenizer, BPE from scratch, attention demo
```

## Roadmap

- [x] UMAY-0 — character-level, from scratch
- [x] BPE tokenizer
  - [x] Merge algorithm from scratch (2.43x compression at 1k merges)
  - [x] Byte-level BPE, vocab 8192, trained on the mixed corpus
- [x] LR schedule (warmup + cosine), bf16 autocast
- [x] Data cleaning: template leftovers, list pages, document separators
- [ ] Scaling: bigger model, more data, longer training, `torch.compile`
- [ ] UMAY-1 — a conversational assistant with tool use
