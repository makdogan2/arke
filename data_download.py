import argparse, os, random
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from textclean import EOT, MIN_CHARS, clean, is_list_page

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "mix.txt"
TARGET_BYTES = {"tr": 120_000_000, "en": 80_000_000}   # 200 MB total
def collect(lang, target):
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                      split="train", streaming=True)
    docs, total, dropped = [], 0, Counter()
    for article in tqdm(ds, desc=lang):
        text = clean(article["text"])
        if len(text) < MIN_CHARS:
            dropped["short"] += 1
            continue
        if is_list_page(text):
            dropped["list"] += 1
            continue
        docs.append(text)
        total += len(text.encode("utf-8"))
        if total >= target:
            break
    print(f"{lang}: kept {len(docs)} | dropped short {dropped['short']}, list {dropped['list']}")
    return docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if mix.txt exists")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT.name} already exists, skipping download (use --force to rebuild)")
        raise SystemExit

    docs = []
    for lang, target in TARGET_BYTES.items():
        docs += collect(lang, target)

    random.seed(42)
    random.shuffle(docs)                     # mix the two languages at document level

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(EOT.join(docs), encoding="utf-8")
    print("done:", round(os.path.getsize(OUT) / 1e6, 1), "MB")
