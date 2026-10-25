# Arke

A Turkish-English language model written from scratch. Learning project and CV repo
(github.com/makdogan2/arke). The author is a first-year computer engineering student whose
goal is to truly understand LLMs.

## How to work with me
- Before every change, briefly explain in Turkish WHAT you will do and WHY.
- Do tasks one at a time and wait for my approval after each one.
- For core ML code (model.py, the training loop), explain the change as you make it; don't skip over anything I might not understand.
- Code, comments and commit messages in English. Conversation in Turkish.

### Who does what
AI is my collaborator, not my replacement: I keep the understanding and the control.
- **Mine:** decisions (what to build, which data, which direction), core logic, and the final
  review of every diff.
- **Yours:** research (options, licenses, trade-offs), boring/repetitive code (downloads, logging,
  argparse, file handling), reviewing my code, tests, explanations.
- Test for each piece: "if this breaks in a week, must I fix it without you?"
  - yes (core logic): give me a skeleton with `TODO(Mehmet)` blanks; I fill them, you check.
    When I'm stuck, give a hint or a small analogous example, never the answer. Never fill my TODOs yourself.
  - no: you write it; I read it and must be able to explain it back.
- Teach with small runnable examples and verified sources (docs, papers, videos); check links before giving them.
- Measure, don't trust: claims like "faster" or "better" get a before/after number.

## Environment
- Windows 11, RTX 5070 Ti, torch 2.11.0+cu128, bf16 autocast.
- Python env is `.venv` (created with uv). PowerShell execution policy blocks the activate script:
  - always run Python as `.venv\Scripts\python.exe <script>`
  - install packages with `uv pip install <pkg>`
- `data/` and `*.pt` are gitignored.

## Pipeline
`data_prep.py` -> `train.py` -> `sample.py`; model in `model.py`, cleaning rules in `textclean.py`.
- `data_prep.py`: Wikipedia (`data/mix.txt`) + FineWeb2-HQ TR + FineWeb-Edu EN, ~50/50 TR/EN.
  Resumable (shards in `data/prep/`), log in `logs/data_prep.log`. Writes `data/train.bin`,
  `data/val.bin` (uint16), `data/meta.json` and the tokenizer `data/bpe_16384.json`.
  `--smoke` does a tiny run into `data/prep_smoke/`.
- `train.py --data <folder>` reads those files with memmap; `--resume` continues from `<name>_last.pt`.
  `torch.compile` needs `triton-windows` (installed); `--no-compile` turns it off.
- Old pipeline (`data_download.py` -> `tokenizer_bpe.py` -> `data/bpe.pt`, vocab 8192) is what
  `arke0_bpe.pt` and the runs/ size experiment were trained on; the current train.py no longer reads it.
- `<|endoftext|>` is a special token (id 0) separating documents.
- `Config` is a dataclass in `model.py`. Checkpoints store `asdict(cfg)`; `sample.py` rebuilds
  the architecture from the checkpoint, not from defaults.
- `experiments/`: scripts from the learning phase (char tokenizer, BPE demos, attention). Not
  part of the pipeline; nothing imports them.
- Current model: `arke0_bpe.pt`, 13.88M params, 8000 steps (~6 min), val loss 3.718/token on the cleaned corpus.
