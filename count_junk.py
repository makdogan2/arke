import random
import re
from pathlib import Path

ROOT = Path(__file__).parent
EOT = "<|endoftext|>"

# "width: 20em; font-size: 100%;" -> at least two "name: value;" pairs on one line
CSS_PAIR = re.compile(r"[a-z-]+\s*:\s*[^;:\n]{1,40};")


def is_css(line):
    return len(CSS_PAIR.findall(line)) >= 2


def category_tail(doc):
    """Short lines without final punctuation at the very end of an article,
    like 'Films set in country houses' / '1941 births'."""
    tail = []
    for line in reversed(doc.rstrip().splitlines()):
        s = line.strip()
        if s and len(s) < 80 and not s.endswith((".", "!", "?", ":", '"', ")")):
            tail.append(s)
        else:
            break
    return tail[::-1] if len(tail) >= 2 else []


text = (ROOT / "data" / "mix.txt").read_text(encoding="utf-8")
docs = text.split(EOT)

css_chars = tail_chars = 0
css_docs = tail_docs = 0
css_examples, tail_examples = [], []

for doc in docs:
    css_lines = [l for l in doc.splitlines() if is_css(l)]
    if css_lines:
        css_docs += 1
        css_chars += sum(len(l) + 1 for l in css_lines)
        css_examples.append(css_lines[0][:100])

    tail = category_tail(doc)
    if tail:
        tail_docs += 1
        tail_chars += sum(len(l) + 1 for l in tail)
        tail_examples.append(" | ".join(tail)[:100])

total = len(text)
print(f"Corpus: {total / 1e6:.1f}M characters, {len(docs):,} articles\n")
print(f"CSS lines      : {css_chars / 1e6:6.2f}M chars ({100 * css_chars / total:5.2f}%) "
      f"in {css_docs:,} articles")
print(f"Category tails : {tail_chars / 1e6:6.2f}M chars ({100 * tail_chars / total:5.2f}%) "
      f"in {tail_docs:,} articles")

random.seed(0)
for name, examples in [("CSS", css_examples), ("Category tail", tail_examples)]:
    print(f"\n{name} examples:")
    for e in random.sample(examples, min(5, len(examples))):
        print("  ", repr(e))