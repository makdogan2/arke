# Arke

A bilingual (Turkish–English) language model built from scratch.

*Arke comes from the Greek* arkhe*, "first principle": the starting point of everything. Fitting for a model built from first principles, from the tokenizer up.*

Part of the Arke family, alongside [Arke Code](https://github.com/makdogan2/arke-code), a local coding assistant. (This project was previously called UMAY.)

## Arke-0

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
uv pip install tokenizers datasets huggingface_hub pyarrow tqdm numpy matplotlib regex
uv pip install "triton-windows>=3.7,<3.8"   # Windows only, for torch.compile (torch 2.11 -> Triton 3.7)
```

On Windows, if the PowerShell execution policy blocks `.venv\Scripts\activate`, skip activation and
call the venv interpreter directly, as in the commands below.

## Data

`data_prep.py` builds a ~4.4 GB corpus, about 50% Turkish and 50% English by bytes:

| Source | Language | License | Used |
|---|---|---|---|
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101) | TR + EN | CC BY-SA 3.0 + GFDL | ~200 MB, from `data_download.py` |
| [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) `tur_Latn` | TR | ODC-By 1.0 | ~2.1 GB after filtering |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT` | EN | ODC-By 1.0 | ~2.1 GB after filtering |

The FineWeb datasets are built from Common Crawl and are also subject to its terms of use.

Steps: download parquet files, clean and filter every document (`textclean.py`: template leftovers,
reference sections, date-list pages, pages under 500 characters, repetitive pages with the
Gopher duplicate-line thresholds), drop exact duplicates, train a byte-level BPE tokenizer
(vocab 16384) on a 50/50 TR/EN sample, tokenize, and send 1% of the documents to validation.
Work is saved in small shards, so after a crash the same command continues where it stopped.

## Usage

```
.venv\Scripts\python.exe data_download.py           # Wikipedia -> data/mix.txt (--force to rebuild)
.venv\Scripts\python.exe data_prep.py               # -> data/train.bin, val.bin, meta.json, bpe_16384.json
.venv\Scripts\python.exe train.py --name arke1      # train -> arke1.pt (best val), arke1_last.pt, runs/arke1.json
.venv\Scripts\python.exe train.py --name arke1 --resume   # continue an interrupted run
.venv\Scripts\python.exe sample.py --name arke1     # generate text from the checkpoint
.venv\Scripts\python.exe -m unittest test_clean     # tests for the cleaning rules
```

- `data_prep.py --smoke` runs the whole pipeline on ~20 MB per source into `data/prep_smoke/`
  in a few minutes; progress goes to `logs/data_prep.log`.
- `train.py` options: model size (`--n_layer --n_embd --n_head`), `--data <folder>`, `--max-iters`,
  `--eval-every`, `--save-every`, `--no-compile`.
- A checkpoint stores the model config, optimizer state, schedule, data folder and random number
  generator states: `sample.py` rebuilds the exact architecture, and `--resume` reproduces the
  losses of an uninterrupted run.

Arke-0 below was trained with the earlier pipeline (`data_download.py` -> `tokenizer_bpe.py` ->
`data/bpe.pt`, vocab 8192), which the current `train.py` no longer reads.

## Results

| Model | Data | Params | Steps | Train time | Val loss | ≈ per char |
|---|---|---|---|---|---|---|
| Arke-0 (char) | raw | ~10.8M | 5000 | ~3 min | 1.31 / char | 1.31 |
| Arke-0 (BPE) | raw | 13.88M | 8000 | — | 3.697 / token | 1.107 |
| Arke-0 (BPE) | cleaned | 13.88M | 8000 | 5 min 45 s | 3.718 / token | 1.100 |

Train time is the training loop on an RTX 5070 Ti (bf16, batch 64, 131M tokens, 2.6 epochs).

On the cleaned corpus (190.3M characters) the BPE tokenizer produces 56.38M tokens, 3.38
characters/token (raw corpus: 56.9M tokens, 3.34). Val loss per token is not comparable across
tokenizers, so the last column divides it by characters/token. The BPE models beat the
character-level one; cleaning barely moves the loss (the validation splits also differ), but it
removed template junk such as `Ankara (; , ;` from the generated text.

## Repository layout

```
data_download.py   Wikipedia download -> data/mix.txt
data_prep.py       resumable data pipeline: download, clean, dedup, tokenizer, train/val .bin
textclean.py       cleaning and filtering rules shared by the data scripts
test_clean.py      tests for textclean.py
model.py           Config dataclass + transformer (Arke0)
train.py           training loop: memmap data, torch.compile, checkpoints + resume
sample.py          text generation
plot_scaling.py    plots the model size experiment from runs/*.json
count_junk.py      measures leftover junk (CSS, category lines, references) in the corpus
tokenizer_bpe.py   earlier pipeline: BPE tokenizer + data/bpe.pt (vocab 8192)
experiments/       learning-phase scripts: char tokenizer, BPE from scratch, attention demo
```

## Roadmap

- [x] Arke-0 — character-level, from scratch
- [x] BPE tokenizer
  - [x] Merge algorithm from scratch (2.43x compression at 1k merges)
  - [x] Byte-level BPE, vocab 8192, trained on the mixed corpus
- [x] LR schedule (warmup + cosine), bf16 autocast
- [x] Data cleaning: template leftovers, list pages, document separators
- [x] `torch.compile` + fused AdamW: 207k -> 251k tokens/s on a 33.7M model
- [x] Checkpoints with exact resume
- [x] Data pipeline: FineWeb2-HQ TR + FineWeb-Edu EN, dedup, 50/50 TR/EN tokenizer (vocab 16384)
- [ ] Scaling: bigger model trained on the new ~4.4 GB corpus (~1.1B tokens)
- [ ] Arke-1 — a conversational assistant with tool use
