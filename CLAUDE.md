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
  - `requirements.txt` pins what the code imports (21 Nov): update it when a package is added or upgraded
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
  A new run stops if `<name>.pt` or `<name>_last.pt` is on disk (17 Nov); `--overwrite` replaces them.
  `--resume` without a `<name>_last.pt` stops with a message. The sample printed at the end of a run is
  generated in eval mode (17 Nov; before that dropout stayed on, which only matters for dropout > 0).
  `torch.compile` needs `triton-windows` (installed); `--no-compile` turns it off.
  `--deterministic` makes runs bit-identical (same seed, also across `--resume`), at no measured speed cost.
  `--block_size` / `--batch_size` (defaults 256 / 64) and `--seq-warmup N` (half-length windows, twice
  as many, for the first N steps; 14 Nov) exist; with all three at their defaults a run is bit-identical
  to the train.py before them.
- Old pipeline (`data_download.py` -> `tokenizer_bpe.py` -> `data/bpe.pt`, vocab 8192) is what
  `arke0_bpe.pt` and the old size experiment (`runs/xs`, `s`, `m`, `l`) were trained on; the current train.py no longer reads it.
  `tokenizer_bpe.py` takes no arguments and rewrites `data/bpe.pt` as soon as it runs. Never try
  `--help` on a script without argparse (`tokenizer_bpe.py`, `count_junk.py`, `explore_batch.py`,
  `embed_demo.py`): on 21 Nov that rewrote `data/bpe.pt` (same content: the step-0 losses of the four
  24 Oct runs came out equal to their logs to six decimals).
- `<|endoftext|>` is a special token (id 0) separating documents.
- `Config` is a dataclass in `model.py`. Checkpoints store `asdict(cfg)`; `sample.py` rebuilds
  the architecture from the checkpoint, not from defaults.
- `forward` without `targets` returns the logits of the last position only, `(B, 1, vocab)` (17 Nov):
  all `generate` needs. With `targets` (training, `quantize.py`) it returns every position.
- `generate(..., kv_cache=True)` (17 Nov) keeps every block's keys and values in fixed tables and runs
  only the newest token through the blocks (`forward_cached` in attention, block and model). Off by
  default: in bf16 it draws other samples than the plain path (same distribution; same text in fp32), and
  the recorded texts and numbers must stay reproducible. It stops applying once the sequence is longer
  than `block_size` (positions are absolute). `sample_stats.py --kv-cache` uses it: 49.6 s -> 7.1 s
  for 500 x 256 tokens, 12.5 GB -> 1.6 GB reserved. No gain for a single prompt.
- Tests: `.venv\Scripts\python.exe -m unittest test_clean test_model test_train test_cli` (56 tests,
  CPU, about 15 s). `test_model.py` covers `forward`, `generate` and the KV cache on a tiny untrained
  model; `test_train.py` covers the checkpoint guard without data or GPU; `test_cli.py` (21 Nov) runs
  the scripts' command lines (`--help`, refused arguments) as copies in an empty temp folder. Run
  them after changing `model.py`, `train.py` or a script's arguments. GitHub Actions runs them at
  every push and pull request (21 Nov; `.github/workflows/tests.yml`: Windows, CPU build of torch,
  the packages of `requirements.txt`).
- `experiments/`: scripts from the learning phase (char tokenizer, BPE demos, attention). Not
  part of the pipeline; nothing imports them. `pretok_compare.py` (14 Nov) trains a GPT-2-rule and an
  apostrophe-aware tokenizer on the same sample into `runs/pretok/` and compares them on val text.
  `kv_cache_demo.py` (17 Nov) shows the KV cache on one attention layer.
- Measuring: `quantize.py --name <ckpt> --bits 16 --data data/v2` gives the full val loss (TR/EN
  per char), on windows of the model's `block_size` or of `--window N` tokens (17 Nov; N <= block_size);
  `sample_stats.py` counts whitespace the cleaning removes and repetition (seq-rep-4, word
  loops, with real validation text as the reference) in 500 samples per model; `--temperature`,
  `--top_k` (0 = off), `--top_p` and `--repetition_penalty` change the sampling (repetition depends
  strongly on them; a penalty of 1.2 already repeats less than real text).
  `blind_pairs.py` writes blind A/B pairs of two sampling settings (`T K P [penalty]`) into `runs/blind/pairs.txt` for
  Mehmet to judge; `--score` counts them. It never overwrites a `pairs.txt` (it may hold his choices).
  `runs/blind/claude_choices.txt` holds Claude's own blind choices (12 Nov), not scored until Mehmet
  has filled his: `--score --choices claude_choices.txt --agree pairs.txt`.
  Run-to-run noise of full val: std about 0.0044 (`v2_xs`, four runs: the seeds 1337 / 1338 / 1339
  gave 3.9085 / 3.9042 / 3.9041, and seed 1337 again with `--deterministic` on 17 Nov gave 3.9133,
  `v2_xs_det`). Treat a difference under 0.01 between two single runs as noise. Run `sample_stats.py`
  with the GPU free: at batch 100 next to a training run or a game it goes out of memory (generating
  reserves about 12.5 GB for 1.2 GB in use; 31.8 GB before 17 Nov; 1.6 GB with `--kv-cache`). The same
  holds for any script that generates a big batch without the KV cache: on 17 Nov one filled the GPU
  next to a training run and slowed it to a third. Check
  `nvidia-smi` first and ask before using a GPU that something else is using.
- Current model: `arke0_clean_last.pt` (`arke0_web`'s recipe on `data/v2`; step 122k, also the
  best checkpoint), 97.8M params, 2.0B tokens, 5 h 57 min. Full v2 val: 3.1593/token, TR 0.8202/ch,
  EN 0.7912/ch. Writes no cleaned-away whitespace (0/500 samples vs 9/500 for `arke0_web_last`).
- Previous: `arke0_web_last.pt` (same recipe on `data/`; v2 val 3.1612, its own val 3.1606); the
  quantization results (`arke0_web_q4.pt`) are for this model. Before that `arke0_bpe.pt`,
  13.88M params, 3.718/token, 1.100/ch.
- Size experiment on `data/v2` (`arke0_clean`'s recipe, 2.0B tokens): `v2_xs_last.pt` 7.42M 3.9085,
  `v2_s_last.pt` 17.03M 3.5872, `v2_m_last.pt` 33.72M 3.3937 (full v2 val). `plot_scaling.py` draws
  them with `arke0_clean` into `docs/scaling.png`; `runs/` stays out of git.
- 15 Nov experiments at v2_xs size (full v2 val): `v2_xs_512_last.pt` (block 512, batch 32) 3.8650,
  `v2_xs_512w_last.pt` (same + `--seq-warmup 12000`) 3.8496; `v2_xs_apos_last.pt` on `data/v2_apos`
  (apostrophe tokenizer, `--data data/v2_apos` to measure it) TR 1.0188 / EN 0.9739 per char, no gain
  over the v2_xs seeds (TR 1.0168–1.0184, EN 0.9731–0.9740). `plot_seq_warmup.py` draws the warmup
  runs into `docs/seq_warmup.png`. `data/v2_apos` keeps only the tokenizer, meta and val files (enough
  to measure `v2_xs_apos`; its train.bin and the junction to `data/v2/prep/clean` were removed). The
  `tok/` work folders of `data/prep`, `data/v2/prep` were deleted too: `.bin` files are built, and
  rebuilding them would tokenize again (~11 min).
- 16–17 Nov: `arke0_512w_last.pt` is `arke0_clean`'s recipe with `--block_size 512 --batch_size 32
  --seq-warmup 12000` (98.00M params, 7 h 6 min).
  - Full v2 val on its own 512-token windows: 3.0859 (TR 0.8018 / EN 0.7722 per char).
  - On 256-token windows: 3.1718, which is worse than `arke0_clean`. The 7.4M 512 models also do
    worse on 256-token windows.
  - It repeats more than `arke0_clean` in 500 samples. seq-rep-4: 15.4% vs 11.9% at the defaults,
    1.7% vs 0.8% at T 1.0 / p 0.9, and 25.4% vs 19.0% for 512-token samples.
  - So it is not the current model (Mehmet's decision, 17 Nov).
  - The 256-token numbers are `quantize.py --window 256`.
- 17 Nov recipe experiment at v2_xs size (seed 1337, `--deterministic`, full v2 val). NOT in `train.py`:
  the runs used `runs/recipe/train_recipe.py` (untracked; `train.py` + `--wd-1d`, `--wd-bias`, `--lr`,
  `--scaled-init`; with the switches off it is bit-identical to `train.py`). Mehmet types the change
  into `train.py` himself; `runs/recipe/compare_ckpt.py` compares two checkpoints bit for bit.
  - Weight decay only on matrices (`--wd-1d 0`: none on LayerNorm gains and biases or Linear biases):
    `v2_xs_nowd_last.pt` 3.8874, `v2_xs_nowd_seed2_last.pt` (seed 1338) 3.8869. The old recipe's four
    runs average 3.9075, so the gain is 0.020. It appears in the last third of training.
  - On top of that: learning rate 1.2e-3 `v2_xs_nowd_lr12_last.pt` 3.8961, 3e-4
    `v2_xs_nowd_lr3_last.pt` 3.9088 (6e-4 stays), scaled initial weights `v2_xs_nowd_init_last.pt`
    3.8966. None of them helps.
  - At 33.7M the gain is not shown: `v2_m_nowd_last.pt` 3.3876 vs `v2_m_last.pt` 3.3937 (−0.006, one
    run each, under the 0.01 rule). Same curve shape: behind by 0.016 mid-run, ahead only after 100k.
  - Text, 2,000 samples per model (`sample_stats.py --kv-cache`): the new rule removes the whitespace
    patterns the cleaning deleted (7.4M: 3–21% of samples -> 0.0–0.2%; 33.7M: 1.1% -> 0.1%).
    Repetition: no effect at 7.4M (the two pairs disagree in direction), but `v2_m_nowd` repeats more
    than `v2_m` (seq-rep-4 19.0% vs 13.9% at the defaults, 1.6% vs 1.3% at T 1.0 / p 0.9; one pair).
  - Gains or biases? `v2_xs_nowd_gain_last.pt` (`--wd-1d 0 --wd-bias 0.1`: only the 9 LayerNorm gains
    are free of decay) 3.8902: 0.023 better than the same-seed control (3.9133) and 0.003 behind the
    full change (3.8874, within noise). So the loss gain comes from the gains. The whitespace patterns
    do not: this model still writes them (2.5% of samples), so that effect comes from the biases.
  - These numbers give no reason to retrain the 98M model; whether to do it is Mehmet's decision.
  - `runs/recipe/plot_weight_decay.py` draws the three pairs into `runs/recipe/weight_decay.png`.
