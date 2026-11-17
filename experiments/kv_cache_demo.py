# KV cache in one attention layer: why generating can skip almost all of the work.
# Runs on the CPU in a second:  .venv\Scripts\python.exe experiments\kv_cache_demo.py
import torch
from torch.nn import functional as F

torch.manual_seed(0)
C = 8                                                               # vector size of a token
W_q, W_k, W_v = (torch.randn(C, C) / C ** 0.5 for _ in range(3))    # the layer's three weight matrices
x = torch.randn(6, C)                                               # six tokens, already embedded, one row each


def attend(q, k, v, mask):
    """Scaled dot-product attention. mask[i, j] is True where query i may look at key j."""
    scores = q @ k.T / C ** 0.5
    scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def full(x):
    """The usual way: every token computes its query, key and value; token i looks at tokens 0..i."""
    n = x.size(0)
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
    return attend(x @ W_q, x @ W_k, x @ W_v, causal)


# --- part 1: appending a token does not change the tokens before it ---
out5, out6 = full(x[:5]), full(x)
print("part 1: outputs of tokens 0-4, with and without token 5 behind them")
print(f"  largest difference: {(out6[:5] - out5).abs().max().item():.1e} (causal attention never looks ahead)")

# --- part 2: generating without and with a cache ---
# Generating needs only the output of the newest token. Without a cache the whole prefix goes through
# the layer at every step. With a cache the keys and values of the earlier tokens are kept, and only
# the newest token computes its query, key and value.
print("\npart 2: output of the newest token, step by step")
keys, values, work_full, work_cached = [], [], 0, 0
for n in range(1, 7):
    slow = full(x[:n])[-1]                                  # n tokens through the layer
    work_full += n
    new = x[n - 1:n]                                        # only the newest token
    keys.append(new @ W_k)
    values.append(new @ W_v)
    work_cached += 1
    k, v = torch.cat(keys), torch.cat(values)
    fast = attend(new @ W_q, k, v, torch.ones(1, n, dtype=torch.bool))[0]   # it may look at everything stored
    print(f"  step {n}: largest difference {(slow - fast).abs().max().item():.1e}")
print(f"  token rows computed: {work_full} without the cache, {work_cached} with it "
      f"(for 256 tokens: {256 * 257 // 2} and 256)")

# --- part 3: a table of fixed size and a mask (what model.py does) ---
# torch.cat builds a slightly bigger tensor at every step. model.py writes into a table with block_size
# rows that is created once, and the query looks at the whole table through a mask: every step then
# works on tensors of the same size.
BLOCK = 8
K, V = torch.zeros(BLOCK, C), torch.zeros(BLOCK, C)
print("\npart 3: the same with a fixed table and a mask")
for p in range(6):
    new = x[p:p + 1]
    K[p], V[p] = (new @ W_k)[0], (new @ W_v)[0]             # row p belongs to position p
    mask = (torch.arange(BLOCK) <= p)[None, :]              # rows 0..p are written, the rest stays hidden
    fast = attend(new @ W_q, K, V, mask)[0]
    print(f"  position {p}: largest difference to the usual way {(full(x[:p + 1])[-1] - fast).abs().max().item():.1e}")

# --- part 4: why the cache is dropped when the window slides ---
# Arke adds a learned position vector to every token before the blocks. Once the text is longer than
# block_size the window moves one token to the right: the token that was at position 3 is now at
# position 2, so its input vector changes, and with it its key and value. Nothing in the tables is
# right any more, and generate goes back to computing the whole window.
pos = torch.randn(BLOCK, C)
token = torch.randn(C)
key_at_3, key_at_2 = (token + pos[3]) @ W_k, (token + pos[2]) @ W_k
print("\npart 4: the key of one token at position 3 and at position 2")
print(f"  largest difference: {(key_at_3 - key_at_2).abs().max().item():.2f} "
      "(another key: the stored one cannot be reused)")
