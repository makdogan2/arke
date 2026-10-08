# Arke

A Turkish-English language model written from scratch. Learning project and CV repo
(github.com/makdogan2/arke). The author is a first-year computer engineering student whose
goal is to truly understand LLMs.

## How to work with me
- Before every change, briefly explain in Turkish WHAT you will do and WHY.
- Do tasks one at a time and wait for my approval after each one.
- For core ML code (model.py, the training loop), explain the change as you make it; don't skip over anything I might not understand.
- Code, comments and commit messages in English. Conversation in Turkish.

## Environment
- Windows 11, RTX 5070 Ti, torch 2.11.0+cu128, bf16 autocast.
- Python env is `.venv` (created with uv). PowerShell execution policy blocks the activate script:
  - always run Python as `.venv\Scripts\python.exe <script>`
  - install packages with `uv pip install <pkg>`
- `data/` and `*.pt` are gitignored.

## Pipeline
`data_download.py` -> `tokenizer_bpe.py` -> `train.py` -> `sample.py`; model in `model.py`.
- Tokenizer: byte-level BPE, vocab 8192 (`data/bpe_8192.json`), encoded corpus in `data/bpe.pt`.
  `<|endoftext|>` is a special token.
- `Config` is a dataclass in `model.py`. Checkpoints store `asdict(cfg)`; `sample.py` rebuilds
  the architecture from the checkpoint, not from defaults.
- `experiments/`: scripts from the learning phase (char tokenizer, BPE demos, attention). Not
  part of the pipeline; nothing imports them.
- Current model: `arke0_bpe.pt`, 13.88M params, 8000 steps (~6 min), val loss 3.718/token on the cleaned corpus.
