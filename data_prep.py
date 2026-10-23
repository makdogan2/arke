"""Overnight data preparation:
download -> clean + filter -> train tokenizer -> dedup + tokenize -> train.bin / val.bin

Every stage works in small pieces (one shard per parquet row group) and skips pieces that are
already on disk, so after a crash or a reboot the same command continues where it stopped.
Progress goes to logs/data_prep.log.

    .venv\\Scripts\\python.exe data_prep.py           # the real run (~2 GB of new text)
    .venv\\Scripts\\python.exe data_prep.py --smoke   # tiny end-to-end run into data/prep_smoke
"""
import argparse, hashlib, json, logging, os, shutil, sys, time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from textclean import EOT, clean, reject_reason

ROOT = Path(__file__).parent
DATA = ROOT / "data"
WIKI = DATA / "mix.txt"           # the existing Wikipedia corpus (~120 MB TR / 80 MB EN)

# bytes of new text kept after filtering; together with Wikipedia the mix is ~50% TR / 50% EN
SOURCES = {
    "tr": dict(repo="epfml/FineWeb2-HQ", prefix="tur_Latn/", target=1_000_000_000),
    "en": dict(repo="HuggingFaceFW/fineweb-edu", prefix="sample/10BT/", target=1_040_000_000),
}
ORDER = {"wiki": 0, "tr": 1, "en": 2}

VOCAB_SIZE = 16384
TOK_SAMPLE_BYTES = 300_000_000    # per language: the tokenizer trains on a balanced 50/50 sample
VAL_PER_MILLE = 10                # 1% of the documents go to validation
WIKI_SHARD_DOCS = 20_000
MIN_FREE_GB = 20
WORKERS = max(1, os.cpu_count() - 4)

log = logging.getLogger("data_prep")


# ---------- small helpers ----------

def setup_log(path):
    path.parent.mkdir(exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is not UTF-8
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%m-%d %H:%M:%S")
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)

def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def save_npy_atomic(path, arr):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    tmp.replace(path)

def check_disk():
    free = shutil.disk_usage(DATA).free / 1e9
    if free < MIN_FREE_GB:
        log.error(f"only {free:.0f} GB free on {DATA.anchor}, need {MIN_FREE_GB} GB. Stopping.")
        raise SystemExit(1)
    return free

def with_retry(what, fn, *args, tries=5, **kwargs):
    """Network calls fail now and then during a night; wait and try again."""
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == tries:
                raise
            log.warning(f"{what} failed ({type(e).__name__}: {e}), retry {attempt}/{tries - 1} in {60 * attempt}s")
            time.sleep(60 * attempt)

def mb(n):
    return f"{n / 1e6:,.0f} MB"


# ---------- state: which shards are done ----------

def load_state(work):
    p = work / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"shards": {}}

def save_state(work, state):
    write_atomic(work / "state.json", json.dumps(state, indent=1))

def shard_names(state, src=None):
    names = [n for n, s in state["shards"].items() if src is None or s["src"] == src]
    return sorted(names, key=lambda n: (ORDER[state["shards"][n]["src"]], n))

def kept_bytes(state, src):
    return sum(s["bytes"] for s in state["shards"].values() if s["src"] == src)

def read_shard(work, name):
    text = (work / "clean" / f"{name}.txt").read_text(encoding="utf-8")
    return [d for d in text.split(EOT) if d]


# ---------- stage 1: clean + filter (runs in worker processes) ----------

def clean_task(task):
    """One shard: a parquet row group, or a slice of the Wikipedia corpus."""
    texts = task["texts"]
    if texts is None:
        table = pq.ParquetFile(task["path"]).read_row_group(task["rg"], columns=["text"])
        texts = table.column("text").to_pylist()
    kept, dropped = [], Counter()
    for t in texts:
        t = clean(t)
        reason = reject_reason(t)
        if reason:
            dropped[reason] += 1
        else:
            kept.append(t)
    write_atomic(task["out"], EOT.join(kept))
    return task["out"].stem, dict(src=task["src"], docs=len(kept), seen=len(texts),
                                  bytes=sum(len(d.encode("utf-8")) for d in kept),
                                  dropped=dict(dropped))

def run_tasks(work, state, tasks, src, target=None):
    """Clean shards in parallel; stop early once `target` bytes are kept for this source."""
    if not tasks:
        return
    with Pool(WORKERS) as pool:                       # leaving the block stops unfinished workers
        for name, info in pool.imap_unordered(clean_task, tasks):
            state["shards"][name] = info
            save_state(work, state)
            kept = kept_bytes(state, src)
            log.info(f"  {name}: kept {info['docs']}/{info['seen']} docs, dropped {info['dropped']} "
                     f"| {src} {mb(kept)}" + (f" / {mb(target)}" if target else ""))
            if target and kept >= target:
                return

def stage_clean(work, scale, wiki_docs):
    clean_dir = work / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(work)

    # Wikipedia: already downloaded, cut into fixed slices
    if not shard_names(state, "wiki"):
        docs = WIKI.read_text(encoding="utf-8").split(EOT)[:wiki_docs]
        tasks = [dict(src="wiki", texts=docs[i:i + WIKI_SHARD_DOCS], path=None, rg=None,
                      out=clean_dir / f"wiki-000-{i // WIKI_SHARD_DOCS:04d}.txt")
                 for i in range(0, len(docs), WIKI_SHARD_DOCS)]
        log.info(f"wiki: {len(docs):,} articles in {len(tasks)} shards")
        run_tasks(work, state, tasks, "wiki")
    log.info(f"wiki: {mb(kept_bytes(state, 'wiki'))} kept")

    # web sources: one parquet file at a time until the byte target is reached
    api = HfApi()
    for src, spec in SOURCES.items():
        target = int(spec["target"] * scale)
        files = with_retry(f"list {spec['repo']}", api.list_repo_files, spec["repo"], repo_type="dataset")
        files = sorted(f for f in files if f.startswith(spec["prefix"]) and f.endswith(".parquet"))
        for i, fname in enumerate(files):
            if kept_bytes(state, src) >= target:
                break
            free = check_disk()
            log.info(f"{src}: file {i + 1}/{len(files)} {fname} (disk free {free:.0f} GB)")
            t = time.time()
            path = Path(with_retry(f"download {fname}", hf_hub_download, spec["repo"], fname,
                                   repo_type="dataset", local_dir=DATA / "raw" / src))
            log.info(f"  ready in {time.time() - t:.0f}s ({path.stat().st_size / 1e9:.2f} GB)")
            n_rg = pq.ParquetFile(path).num_row_groups
            tasks = [dict(src=src, path=path, rg=rg, texts=None,
                          out=clean_dir / f"{src}-{i:03d}-{rg:04d}.txt")
                     for rg in range(n_rg) if f"{src}-{i:03d}-{rg:04d}" not in state["shards"]]
            run_tasks(work, state, tasks, src, target)
        log.info(f"{src}: {mb(kept_bytes(state, src))} kept (target {mb(target)})")
    return state


# ---------- stage 2: tokenizer ----------

def stage_tokenizer(work, state, tok_path, sample_bytes):
    if tok_path.exists():
        log.info(f"tokenizer: using existing {tok_path.name}")
        return Tokenizer.from_file(str(tok_path))

    def sample(src):
        n = 0
        for name in shard_names(state, src):
            for d in read_shard(work, name):
                yield d
                n += len(d.encode("utf-8"))
                if n >= sample_bytes:
                    return

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t = time.time()
    log.info(f"tokenizer: training vocab {VOCAB_SIZE} on {mb(sample_bytes)} TR + {mb(sample_bytes)} EN")
    tok.train_from_iterator((d for src in ("tr", "en") for d in sample(src)), trainer)
    tmp = tok_path.with_name(tok_path.name + ".tmp")
    tok.save(str(tmp))
    tmp.replace(tok_path)
    log.info(f"tokenizer: saved {tok_path.name} in {time.time() - t:.0f}s")

    # how well does it compress each language? (higher chars/token = fewer tokens per sentence)
    old = DATA / "bpe_8192.json"
    tokenizers = {"new": tok, **({"old 8192": Tokenizer.from_file(str(old))} if old.exists() else {})}
    for src in ("tr", "en"):
        docs = read_shard(work, shard_names(state, src)[-1])[:2000]
        chars = sum(len(d) for d in docs)
        ratios = {k: chars / sum(len(e.ids) for e in t.encode_batch(docs)) for k, t in tokenizers.items()}
        log.info(f"  {src} chars/token: " + ", ".join(f"{k} {r:.2f}" for k, r in ratios.items()))
    return tok


# ---------- stage 3: dedup + tokenize + split ----------

def doc_key(doc):
    """Exact-duplicate key: same text up to case and whitespace."""
    return hashlib.md5(" ".join(doc.lower().split()).encode("utf-8")).digest()

def stage_tokenize(work, state, tok):
    tok_dir = work / "tok"
    tok_dir.mkdir(exist_ok=True)
    eot = tok.token_to_id(EOT)
    assert tok.get_vocab_size() <= 2**16, "token ids must fit in uint16"
    seen, dups = set(), Counter()
    t = time.time()
    names = shard_names(state)
    for k, name in enumerate(names):
        docs = read_shard(work, name)
        keys = [doc_key(d) for d in docs]
        out = {s: tok_dir / f"{name}.{s}.npy" for s in ("train", "val")}
        if all(p.exists() for p in out.values()):
            seen.update(keys)            # resume: only rebuild the dedup set
            continue

        keep = []
        for d, key in zip(docs, keys):
            if key in seen:
                dups[state["shards"][name]["src"]] += 1
                continue
            seen.add(key)
            keep.append((d, key))
        enc = tok.encode_batch([d for d, _ in keep])     # Rust, uses all cores
        parts = {"train": [], "val": []}
        for (_, key), e in zip(keep, enc):
            split = "val" if int.from_bytes(key[:4], "little") % 1000 < VAL_PER_MILLE else "train"
            parts[split].append(np.array(e.ids + [eot], dtype=np.uint16))
        for s, p in out.items():
            save_npy_atomic(p, np.concatenate(parts[s]) if parts[s] else np.zeros(0, np.uint16))
        if k % 20 == 0 or k == len(names) - 1:
            log.info(f"tokenize: {k + 1}/{len(names)} shards | {time.time() - t:.0f}s")
    log.info(f"dedup: dropped exact duplicates {dict(dups)}")


# ---------- stage 4: train.bin / val.bin ----------

def stage_bin(work, state, final, tok_path):
    names = shard_names(state)
    meta = dict(tokenizer=tok_path.name, dtype="uint16", vocab_size=Tokenizer.from_file(str(tok_path)).get_vocab_size(),
                bytes_by_source={s: kept_bytes(state, s) for s in ORDER}, tokens_by_source=Counter())
    for split in ("train", "val"):
        out = final / f"{split}.bin"
        tmp = out.with_name(out.name + ".tmp")
        n = 0
        with open(tmp, "wb") as f:
            for name in names:
                arr = np.load(work / "tok" / f"{name}.{split}.npy")
                f.write(arr.tobytes())
                n += len(arr)
                meta["tokens_by_source"][state["shards"][name]["src"]] += len(arr)
        tmp.replace(out)
        meta[f"{split}_tokens"] = n
        log.info(f"{out.name}: {n / 1e6:,.1f}M tokens ({out.stat().st_size / 1e9:.2f} GB)")
    write_atomic(final / "meta.json", json.dumps(meta, indent=1))
    by_src = meta["tokens_by_source"]
    total = sum(by_src.values())
    log.info("tokens by source: " + ", ".join(f"{s} {by_src[s] / 1e6:,.0f}M ({100 * by_src[s] / total:.0f}%)" for s in ORDER))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run (~20 MB per source) into data/prep_smoke")
    args = parser.parse_args()

    if args.smoke:
        work = final = DATA / "prep_smoke"
        scale, wiki_docs, sample_bytes = 0.02, 5000, 10_000_000
    else:
        work, final = DATA / "prep", DATA
        scale, wiki_docs, sample_bytes = 1.0, None, TOK_SAMPLE_BYTES
    work.mkdir(parents=True, exist_ok=True)
    tok_path = final / f"bpe_{VOCAB_SIZE}.json"
    setup_log(ROOT / "logs" / ("data_prep_smoke.log" if args.smoke else "data_prep.log"))

    t0 = time.time()
    log.info(f"===== start ({'smoke' if args.smoke else 'full'}), {WORKERS} workers, disk free {check_disk():.0f} GB")
    try:
        state = stage_clean(work, scale, wiki_docs)
        tok = stage_tokenizer(work, state, tok_path, sample_bytes)
        stage_tokenize(work, state, tok)
        stage_bin(work, state, final, tok_path)
    except BaseException:
        log.exception("stopped with an error. Run the same command again to continue.")
        raise
    log.info(f"===== DONE in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
