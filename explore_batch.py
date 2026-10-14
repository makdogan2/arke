import torch
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).parent

d = torch.load(ROOT / "data" / "bpe.pt", weights_only=False)
train_data = d["train"]
tok = Tokenizer.from_file(str(ROOT / "data" / "bpe_8192.json"))

start = 40_000_000
n = 15

x = train_data[start:start + n]
y = train_data[start + 1:start + n + 1]

print("x ids:", x.tolist())
print("y ids:", y.tolist())
print()

for i in range(n):
    seen = tok.decode(x[: i + 1].tolist())
    target = tok.decode([y[i].item()])
    print(f"{seen[-60:]!r:>62}  ->  {target!r}")