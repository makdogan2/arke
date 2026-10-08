from pathlib import Path
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = Path(__file__).parent
CORPUS = ROOT / "data" / "mix.txt"
TOKENIZER_FILE = ROOT / "data" / "bpe_8192.json"
OUT = ROOT / "data" / "bpe.pt"
VOCAB_SIZE = 8192
EOT = "<|endoftext|>"   # document separator written by data_download.py

text = CORPUS.read_text(encoding="utf-8")

# --- load cached tokenizer, or train it ---
if TOKENIZER_FILE.exists():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    print("loaded tokenizer:", TOKENIZER_FILE.name)
else:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    ) 
    # train on documents split at EOT, so the separator string never enters the merge statistics
    tokenizer.train_from_iterator(text.split(EOT), trainer)
    tokenizer.save(str(TOKENIZER_FILE))
    print("trained and saved:", TOKENIZER_FILE.name)


# --- how does it segment? ---
samples = [
    "Türkiye'de evlerimizden çıkarken gerçekleştirilen çalışmalar",
    "umaylaştırmak zortlanabilirlik tokenizasyon Mehmet",
    "The history of the Ottoman Empire is long and complicated.",
    "Emoji test: 🚀 汉字 also survive.",
]
for s in samples:
  print(f"\n{s}")
  print("", tokenizer.encode(s).tokens)

# --- encode the whole corpus ---
ids = tokenizer.encode(text).ids

print("\n--- stats ---")
print("characters :", len(text))
print("tokens     :", len(ids))
print("compression:", round(len(text) / len(ids), 2), "chars/token")
print("vocab size :", tokenizer.get_vocab_size())

# lossless check
dec = tokenizer.decode(ids[:5000], skip_special_tokens=False)
assert dec == text[:len(dec)], "roundtrip failed"
print("roundtrip  : ok")

data = torch.tensor(ids, dtype=torch.uint16)
n = int(0.9 * len(data))
torch.save({"train": data[:n], "val": data[n:], "vocab_size": tokenizer.get_vocab_size()}, OUT)
print("saved:", OUT.name, "| train", n, "| val", len(data) - n)