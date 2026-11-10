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
- Code: you give it to me in chat and I type it myself in VS Code (since 2026-10-27).
  - Complete and ready to type: no blanks, no TODOs, no open question I'd have to answer first.
  - Say exactly where it goes: a whole file, or which file and after which existing line.
  - After the code: what each part does and why, then the command that checks it.
  - After the pieces, also give the whole changed file in one code block, so I can compare.
  - Don't write code files yourself. Once I've typed it, read the file and check it against what you sent.
  - Commit only when I say "commit".
- Docs and notes (`DEFTER.md`, `CLAUDE.md`, README): you write them, I read the diff.
- `DEFTER.md` is the stage notebook: keep the current stage, the next stages, finished stages
  (with commits) and decisions up to date at the end of every stage.
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
  `--smoke` does a tiny run into `data/prep_smoke/`. `--out v2` writes into `data/v2/` instead.
- Current data: `data/v2/` (since 2026-11-04: same sources, whitespace normalized by
  `textclean.normalize_whitespace`, tokenized with the same `bpe_16384.json` as `data/`).
  `data/` is the pre-cleaning version that `arke0_web` was trained on.
- `train.py --data <folder>` reads those files with memmap; `--resume` continues from `<name>_last.pt`.
  `torch.compile` needs `triton-windows` (installed); `--no-compile` turns it off.
- Old pipeline (`data_download.py` -> `tokenizer_bpe.py` -> `data/bpe.pt`, vocab 8192) is what
  `arke0_bpe.pt` and the old size experiment (`runs/xs`, `s`, `m`, `l`) were trained on; the current train.py no longer reads it.
- `<|endoftext|>` is a special token (id 0) separating documents.
- `Config` is a dataclass in `model.py`. Checkpoints store `asdict(cfg)`; `sample.py` rebuilds
  the architecture from the checkpoint, not from defaults.
- `experiments/`: scripts from the learning phase (char tokenizer, BPE demos, attention). Not
  part of the pipeline; nothing imports them.
- Measuring: `quantize.py --name <ckpt> --bits 16 --data data/v2` gives the full val loss (TR/EN
  per char); `sample_stats.py` counts whitespace the cleaning removes and repetition (seq-rep-4, word
  loops, with real validation text as the reference) in 500 samples per model; `--temperature` and
  `--top_k` change the sampling (repetition depends strongly on them). Run-to-run noise of
  full val: std about 0.0025 (`v2_xs`, three seeds). Run `sample_stats.py` with the GPU free: at
  batch 100 next to a training run it goes out of memory.
- Current model: `arke0_clean_last.pt` (`arke0_web`'s recipe on `data/v2`; step 122k, also the
  best checkpoint), 97.8M params, 2.0B tokens, 5 h 57 min. Full v2 val: 3.1593/token, TR 0.8202/ch,
  EN 0.7912/ch. Writes no cleaned-away whitespace (0/500 samples vs 9/500 for `arke0_web_last`).
- Previous: `arke0_web_last.pt` (same recipe on `data/`; v2 val 3.1612, its own val 3.1606); the
  quantization results (`arke0_web_q4.pt`) are for this model. Before that `arke0_bpe.pt`,
  13.88M params, 3.718/token, 1.100/ch.
- Size experiment on `data/v2` (`arke0_clean`'s recipe, 2.0B tokens): `v2_xs_last.pt` 7.42M 3.9085,
  `v2_s_last.pt` 17.03M 3.5872, `v2_m_last.pt` 33.72M 3.3937 (full v2 val). `plot_scaling.py` draws
  them with `arke0_clean` into `docs/scaling.png`; `runs/` stays out of git.
