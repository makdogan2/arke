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
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None, top_p=None,
                 repetition_penalty=None):
        """Append up to max_new_tokens sampled tokens to every row of idx.
        top_p: sample only from the smallest set of most likely tokens whose probabilities add up to
        at least top_p (nucleus sampling, Holtzman et al. 2019). None keeps every token.
        repetition_penalty: every token already in the row's context window (prompt included) gets its
        logit divided by this number if positive and multiplied by it if negative, before temperature
        (Keskar et al. 2019, CTRL). Above 1 makes repeats less likely; None (or 1) changes nothing.
        stop_token: a row that samples it is finished; its later positions are filled with stop_token,
        and the loop ends early once every row is finished. None keeps generating to max_new_tokens."""
        done = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)   # one flag per row
        for _ in range(max_new_tokens):
            idx_kirp = idx[:, -self.cfg.block_size:]   # pencereyi aşma
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