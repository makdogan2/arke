import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 8192
    block_size: int = 256     # how many tokens the model can look back
    n_embd: int = 384         # vector size of each token
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv    = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj   = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop_p = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        # For the KV cache (forward_cached): row p says which positions position p may look at, 0 .. p.
        # A buffer moves to the GPU with the model; persistent=False keeps it out of the checkpoints.
        self.register_buffer("causal", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
                             persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, kafa, T, kafa_boyutu)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop_p if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def forward_cached(self, x, cache, start):
        """Attention while generating with a KV cache: the same result as forward, computed for the NEW
        positions only.
        x: (B, t, C), the new positions; the first of them is position `start` of the sequence.
        cache: this layer's dict with two tables, "k" and "v", each (B, heads, block_size, head_size).
        Row p of a table holds the key / value of position p. Earlier calls wrote the rows before `start`;
        this call writes rows start .. start + t - 1. The first call creates the tables."""
        B, t, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, t, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, t, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, t, self.n_head, C // self.n_head).transpose(1, 2)
        if not cache:
            # same type as the keys (bf16 under autocast); the rows not written yet are masked out below
            block_size = self.causal.size(0)
            cache["k"] = k.new_zeros(B, self.n_head, block_size, C // self.n_head)
            cache["v"] = v.new_zeros(B, self.n_head, block_size, C // self.n_head)
        cache["k"][:, :, start:start + t] = k
        cache["v"][:, :, start:start + t] = v
        # The mask rows of the new positions, (t, block_size), True = may look: each one sees the table
        # rows up to itself. The query looks at the whole table through the mask instead of at a slice
        # that grows with every token: all steps then work on tensors of one size, which is fast and does
        # not pile up GPU memory.
        y = F.scaled_dot_product_attention(
            q, cache["k"], cache["v"], attn_mask=self.causal[start:start + t],
            dropout_p=self.drop_p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, t, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff  = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # önce normalize, sonra topla
        x = x + self.ff(self.ln2(x))
        return x

    def forward_cached(self, x, cache, start):
        """forward for the new positions only (KV cache). LayerNorm and the feed-forward layer work on
        each position by itself, so only attention needs the cache."""
        x = x + self.attn.forward_cached(self.ln1(x), cache, start)
        x = x + self.ff(self.ln2(x))
        return x


class Arke0(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)
        self.head.weight = self.tok_emb.weight   # weight tying

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blok in self.blocks:
            x = blok(x)

        if targets is not None:      # training and evaluation: a score for every position, (B, T, vocab)
            logits = self.head(self.ln_f(x))
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        else:
            # Generating: only the last position predicts the next token. LayerNorm and the output layer
            # work on each position by itself, so they can run on that one alone: the same scores,
            # without building the (B, T, vocab) table. The slice -1: (not the index -1) keeps the
            # T axis, so the shape is (B, 1, vocab).
            logits = self.head(self.ln_f(x[:, -1:, :]))
            loss = None
        return logits, loss

    def forward_cached(self, idx_new, caches, start):
        """Generating with a KV cache: the logits of the last position, (B, 1, vocab), computed from the
        new tokens only.
        idx_new: (B, t), the tokens the blocks have not seen yet; the first of them is position `start`.
        caches: one dict per block (see MultiHeadAttention.forward_cached). They hold the keys and values
        of positions 0 .. start - 1; this call adds the new positions.
        Why this is allowed: attention is causal, a position only looks back. Appending a token therefore
        changes nothing that the earlier positions computed, and their keys and values can be kept
        instead of being computed again at every step."""
        pos = torch.arange(start, start + idx_new.size(1), device=idx_new.device)
        x = self.drop(self.tok_emb(idx_new) + self.pos_emb(pos))
        for blok, cache in zip(self.blocks, caches):
            x = blok.forward_cached(x, cache, start)
        return self.head(self.ln_f(x[:, -1:, :]))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None, top_p=None,
                 repetition_penalty=None, kv_cache=False):
        """Append up to max_new_tokens sampled tokens to every row of idx.
        top_p: sample only from the smallest set of most likely tokens whose probabilities add up to
        at least top_p (nucleus sampling, Holtzman et al. 2019). None keeps every token.
        repetition_penalty: every token already in the row's context window (prompt included) gets its
        logit divided by this number if positive and multiplied by it if negative, before temperature
        (Keskar et al. 2019, CTRL). Above 1 makes repeats less likely; None (or 1) changes nothing.
        stop_token: a row that samples it is finished; its later positions are filled with stop_token,
        and the loop ends early once every row is finished. None keeps generating to max_new_tokens.
        kv_cache: keep every block's keys and values, and run only the newest token through the blocks
        instead of the whole window at every step. Much faster for a batch of rows. The logits are the same
        up to rounding; in bf16 that rounding can change which token is sampled, so it is off by default
        and the same seed keeps giving the same text. It applies while the sequence fits in block_size:
        once the window slides, every token's position changes and the whole window is computed again."""
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)   # one flag per row
        caches = [{} for _ in self.blocks] if kv_cache else None    # one table of keys and values per block
        n_cached = 0                                                 # how many positions are in the tables
        for _ in range(max_new_tokens):
            idx_kirp = idx[:, -self.cfg.block_size:]   # pencereyi aşma
            if caches is not None and idx.size(1) > self.cfg.block_size:
                caches = None                           # the window slides from here on: the tables no longer fit
            if caches is not None:
                # the first step sends the whole prompt, every later step only the token just sampled
                logits = self.forward_cached(idx[:, n_cached:], caches, n_cached)
                n_cached = idx.size(1)
            else:
                logits, _ = self(idx_kirp)
            logits = logits[:, -1, :]                   # sadece son pozisyon
            if repetition_penalty is not None:
                seen = logits.gather(1, idx_kirp)       # the scores of the tokens this row has already seen
                seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                logits = logits.scatter(1, idx_kirp, seen)   # a token seen twice gets the same value twice: penalized once
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            if top_p is not None:
                sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits.float(), dim=-1)   # float32: a bf16 cumsum can stall below top_p
                before = sorted_probs.cumsum(dim=-1) - sorted_probs     # probability of all tokens ranked above
                sorted_logits[before >= top_p] = -float('inf')          # the most likely token always stays (before = 0)
                logits = logits.scatter(1, order, sorted_logits)       # put the scores back in vocabulary order
            probs = F.softmax(logits, dim=-1)
            sonraki = torch.multinomial(probs, num_samples=1)
            if stop_token is not None:
                sonraki[done] = stop_token               # a finished row only repeats the stop token
                done |= sonraki[:, 0] == stop_token      # rows that just sampled it are finished now
            idx = torch.cat((idx, sonraki), dim=1)
            if done.all():                               # every row finished: nothing left to generate
                break
        return idx