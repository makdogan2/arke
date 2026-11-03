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

## arke0_web

The same architecture, scaled up and trained on the new web corpus (see Data).

- Byte-level BPE tokenizer, vocab 16384, trained on a 50/50 TR/EN sample
- 12 layers, 12 heads, 768 embedding dim, 256-token context, no dropout
- 97.8M parameters (85.0M without embeddings)
- Training: 122,000 steps × 64 × 256 = 2.0B tokens (1.87 epochs of the 1.07B-token corpus,
  ~20 tokens per parameter), 5 h 47 min on one RTX 5070 Ti at 103k tokens/s, in several sessions

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
.venv\Scripts\python.exe quantize.py --scheme asym  # fake quantization at 16/8/4/2 bits -> loss table
.venv\Scripts\python.exe quantize.py --scheme asym --bits 16 4 --packed --save arke0_web_q4.pt
.venv\Scripts\python.exe quantize.py --load arke0_web_q4.pt   # measure the 4-bit file alone
```

- `quantize.py` options:
  - `--name` picks the checkpoint (default `arke0_web_last`).
  - `--embedding` also quantizes the token embedding.
  - `--sample` generates text instead of measuring the loss.

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
| arke0_web | web + Wikipedia | 97.8M | 122,000 | 5 h 47 min | 3.161 / token | TR 0.820, EN 0.792 |

Train time is the training loop on an RTX 5070 Ti (bf16, batch 64). The Arke-0 rows saw 131M
tokens (2.6 epochs of Wikipedia only), arke0_web 2.0B tokens.

On the cleaned corpus (190.3M characters) the BPE tokenizer produces 56.38M tokens, 3.38
characters/token (raw corpus: 56.9M tokens, 3.34). Val loss per token is not comparable across
tokenizers, so the last column divides it by characters/token. The BPE models beat the
character-level one; cleaning barely moves the loss (the validation splits also differ), but it
removed template junk such as `Ankara (; , ;` from the generated text.

### arke0_web vs Arke-0 on the same text

The per-char numbers in the table above come from different validation sets (Arke-0: Wikipedia,
arke0_web: web text), so they cannot be compared directly. Here both models read the same
characters, each through its own tokenizer, scored over every 256-token window:

| Text | Arke-0 (13.9M) | arke0_web (97.8M) | Change |
|---|---|---|---|
| Wikipedia, unseen by both models (32 articles, 0.2M chars) | 1.110 / char | **0.854 / char** | −23% |
| Web TR (`val_tr.bin`, 19.4M chars) | 1.265 / char | **0.820 / char** | −35% |
| Web EN (`val_en.bin`, 20.8M chars) | 1.233 / char | **0.792 / char** | −36% |

The Wikipedia row is the fair comparison: both models trained on Wikipedia and neither saw these
articles (they are in both validation splits). Part of the gap on web text is domain: Arke-0 never
saw web pages and scores 0.12–0.15 worse there than on Wikipedia. The comparison does not
separate the causes: arke0_web is 7× larger, saw 15× more tokens, and has a different tokenizer
and corpus.

Training curve: validation loss kept falling until the end, also in the second epoch. The gap
between train and val loss grew from ~0.02 to ~0.03, a mild sign of memorization that did not
hurt val loss. The in-training eval samples 50 random batches and moves by ±0.008 between evals,
which is more than the late-training improvement (~0.002 per 1000 steps), so `arke0_web.pt`
(the "best" checkpoint, step 118k) is a noisy pick: on the full validation set the last
checkpoint (step 122k) is slightly better, 3.1606 vs 3.1636.

Samples: Turkish and English grammar is good (suffixes such as *değiştirilmiştir*, *anılmaktadır*
are correct) and a paragraph stays on topic, but facts are invented, phrases repeat, and the model
sometimes drifts into runs of spaces and line breaks. These most likely come from the Wikipedia
part of the corpus, which still has 614 double spaces per million characters after cleaning
(web text: 0–5).

## Quantization

`quantize.py` makes `arke0_web_last.pt` smaller with code written from scratch, without a
quantization library. It splits the weights into groups of 64, gives each group its own scale, and
rounds every weight to an integer. The loss is measured on the full validation sets: every 256-token
window is scored, so random sampling adds no noise.

Only the linear weights of the 12 blocks are quantized. That is 84.9M of the 97.8M parameters
(86.8%). The token embedding, LayerNorms and biases stay bf16.

There are two schemes:
- **sym (absmax):** each group maps to −qmax … +qmax with one fp16 scale.
- **asym (min-max):** each group maps its [min, max] onto 0 … 2^bits − 1. It stores an fp16 scale and
  an fp16 minimum, which costs 0.25 bit/weight more at group size 64.

| Bits | sym bit/weight | sym val | asym bit/weight | asym val | asym TR / char | asym EN / char | asym size |
|---|---|---|---|---|---|---|---|
| 16 (bf16) | 16 | 3.1606 | 16 | 3.1606 | 0.8203 | 0.7916 | 195.6 MB |
| 8 | 8.25 | 3.1607 | 8.50 | 3.1607 | 0.8204 | 0.7916 | 116.0 MB |
| 4 | 4.25 | 3.1956 | 4.50 | **3.1846** | 0.8268 | 0.7973 | **73.5 MB** |
| 2 | 2.25 | 9.49 | 2.50 | 4.58 | 1.2046 | 1.1305 | 52.3 MB |

The val columns are loss per token. Sizes include the scales, the minimums and the bf16 parts.

- **8 bits loses nothing measurable:** the loss rises by 0.0001.
- **4 bits:** the loss rises by 0.024 with asym and 0.035 with sym. Part of asym's advantage comes
  from storing more numbers per group. With groups of 32, sym also costs 4.50 bit/weight, and it
  reaches +0.0276. So of the 0.0110 gap between sym and asym at group 64, about 0.0074 comes from
  the smaller groups and only 0.0036 from the scheme.
- **2 bits sym collapses:** val 9.49, close to random guessing (ln 16384 = 9.70). With `qmax = 1` the
  only levels are −1, 0 and +1, so every weight smaller than half of its group's maximum becomes 0.
  In `experiments/quant_demo.py`, 6 of 8 sample weights become 0. asym uses all four levels and
  does not collapse, but 4.58 is still a large loss: training got below it at step 2,000.
- **The embedding is far more sensitive.** With `--embedding` the embedding is quantized too: at
  4 bits the file is 55.4 MB and the loss rises by 0.052. The 12.6M embedding weights add +0.028 on
  their own, as much as the 84.9M block weights, so each embedding weight is about 8× more
  sensitive. Weight tying is the likely cause, since the same matrix is also the output layer. This
  was not tested separately.
- **At the same size, 4 bits beats 2 bits:**

  | Setup | Size | Val |
  |---|---|---|
  | 4-bit with the embedding quantized | 55.4 MB | 3.2125 |
  | 2-bit with the embedding left bf16 | 52.3 MB | 4.5801 |

- **Generated text:** `--sample` uses the same prompts and seed at every bit width, so only the
  weights differ.
  - 8 bits: three of the four texts match 16 bits word for word, and the fourth differs only in its
    last few tokens.
  - 4 bits: the texts diverge early, within the first sentence or right after it, but Turkish and
    English grammar is as good as at 16 bits.
  - 2 bits: unreadable.

### Real 4-bit packing

Fake quantization rounds the weights and turns them right back into floats, so the model in memory
does not get smaller. `--packed` does it for real with 4-bit asym:
- Two 4-bit values share one byte: element 2i goes in the low 4 bits and element 2i+1 in the high
  4 bits.
- Every group of 64 weights also stores an fp16 scale and an fp16 minimum.

Counted from the bytes of the packed tensors, this takes **4.50 bit/weight and 73.5 MB**, the same as
the formula. The val loss matches fake quantization at 3.1846. Unpacking returns the exact 4-bit
values; storing the scale and minimum in fp16 moves a weight by at most 0.0007 (1.1% of one
quantization step).

`--save` writes the packed model to one file. `--load` rebuilds the model from that file alone,
without the original checkpoint:

| File | Size on disk | Val | TR / char | EN / char |
|---|---|---|---|---|
| `arke0_web_last.pt`: training checkpoint | 1,174 MB | 3.1606 | 0.8203 | 0.7916 |
| `arke0_web_q4.pt`: 4-bit asym | **73.6 MB** | 3.1847 | 0.8269 | 0.7973 |

The checkpoint is 16× larger, but 782 MB of it is the AdamW state that resuming training needs. The
fp32 weights are another 391 MB. Measured against bf16 weights alone (195.6 MB), the 4-bit file is
2.7× smaller.

On disk the file holds:
- the config, tokenizer name and group size
- 48 packed matrices
- 76 bf16 tensors
- 66 KB of pickle and zip overhead (0.09%)

The tied embedding is written once. Loading from the file adds 0.0001 to the loss, because the parts
that are not quantized are stored as bf16.

Every script opens `.pt` files with `torch.load(..., weights_only=True)`. In this mode only
tensors, dicts, lists, numbers and strings can come out of a file, so opening a downloaded file
cannot run code.

## Repository layout

```
data_download.py   Wikipedia download -> data/mix.txt
data_prep.py       resumable data pipeline: download, clean, dedup, tokenizer, train/val .bin
textclean.py       cleaning and filtering rules shared by the data scripts
test_clean.py      tests for textclean.py
model.py           Config dataclass + transformer (Arke0)
train.py           training loop: memmap data, torch.compile, checkpoints + resume
sample.py          text generation
quantize.py        quantization from scratch: 8/4/2-bit loss table, 4-bit packing, --save / --load
plot_scaling.py    plots the model size experiment from runs/*.json
count_junk.py      measures leftover junk (CSS, category lines, references) in the corpus
tokenizer_bpe.py   earlier pipeline: BPE tokenizer + data/bpe.pt (vocab 8192)
experiments/       learning-phase scripts: char tokenizer, BPE from scratch, attention, quantization and packing demos
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
- [x] Scaling: arke0_web, 97.8M parameters trained on the new ~4.4 GB corpus (2.0B tokens)
- [x] Quantization from scratch: 8, 4 and 2-bit weights
  - [x] Real 4-bit packing and a standalone file: 1,174 MB checkpoint -> 73.6 MB, val +0.024
- [ ] Arke-1 — a conversational assistant with tool use
