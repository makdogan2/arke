"""Tests for model.py: what forward returns and what generate promises.
A tiny untrained model on the CPU; the whole file runs in a few seconds.

    .venv\\Scripts\\python.exe -m unittest test_model
"""
import math
import unittest
from dataclasses import replace
import torch
from model import Arke0, Config

CFG = Config(vocab_size=64, block_size=16, n_embd=32, n_head=2, n_layer=2, dropout=0.0)


def make_model(seed=0):
    torch.manual_seed(seed)
    model = Arke0(CFG)
    model.eval()
    return model


def prompt(rows, length, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(CFG.vocab_size, (rows, length), generator=g)


def all_logits(model, idx):
    """Logits of every position, (B, T, vocab). forward returns them only together with a loss."""
    with torch.no_grad():
        logits, _ = model(idx, targets=torch.zeros_like(idx))
    return logits


def greedy(model, idx, n):
    """The reference for generate: always take the most likely token, written as plainly as possible."""
    for _ in range(n):
        window = idx[:, -CFG.block_size:]
        nxt = all_logits(model, window)[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, nxt), dim=1)
    return idx


class TestForward(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.idx = prompt(3, 10)

    def test_with_targets_scores_every_position(self):
        targets = prompt(3, 10, seed=2)       # not the inputs: with the tied head a token scores itself highest
        logits, loss = self.model(self.idx, targets=targets)
        self.assertEqual(logits.shape, (3, 10, CFG.vocab_size))
        self.assertEqual(loss.shape, ())
        # an untrained model knows nothing: its loss is close to a uniform guess over the vocabulary
        self.assertAlmostEqual(loss.item(), math.log(CFG.vocab_size), delta=0.05)

    def test_without_targets_returns_the_last_position_only(self):
        with torch.no_grad():
            logits, loss = self.model(self.idx)
        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (3, 1, CFG.vocab_size))
        self.assertTrue(torch.allclose(logits, all_logits(self.model, self.idx)[:, -1:, :], atol=1e-5))

    def test_head_shares_the_embedding_matrix(self):
        self.assertIs(self.model.head.weight, self.model.tok_emb.weight)

    def test_a_position_cannot_see_later_tokens(self):
        changed = self.idx.clone()
        changed[:, 6] = (changed[:, 6] + 1) % CFG.vocab_size
        before, after = all_logits(self.model, self.idx), all_logits(self.model, changed)
        self.assertTrue(torch.equal(before[:, :6], after[:, :6]))       # positions 0-5 never saw token 6
        self.assertFalse(torch.allclose(before[:, 6:], after[:, 6:]))   # position 6 and later did


class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.idx = prompt(3, 5)

    def test_keeps_the_prompt_and_appends_tokens(self):
        torch.manual_seed(0)
        out = self.model.generate(self.idx, 7)
        self.assertEqual(out.shape, (3, 12))
        self.assertTrue(torch.equal(out[:, :5], self.idx))
        self.assertTrue(((out >= 0) & (out < CFG.vocab_size)).all())

    def test_same_seed_gives_the_same_tokens(self):
        torch.manual_seed(7)
        first = self.model.generate(self.idx, 10, temperature=0.8, top_k=20)
        torch.manual_seed(7)
        second = self.model.generate(self.idx, 10, temperature=0.8, top_k=20)
        self.assertTrue(torch.equal(first, second))

    def test_top_k_1_is_greedy(self):
        out = self.model.generate(self.idx, 8, top_k=1)
        self.assertTrue(torch.equal(out, greedy(self.model, self.idx, 8)))

    def test_tiny_top_p_is_greedy(self):
        # the most likely token always stays, whatever top_p is
        out = self.model.generate(self.idx, 8, top_p=1e-6)
        self.assertTrue(torch.equal(out, greedy(self.model, self.idx, 8)))

    def test_very_low_temperature_is_greedy(self):
        out = self.model.generate(self.idx, 8, temperature=1e-4)
        self.assertTrue(torch.equal(out, greedy(self.model, self.idx, 8)))

    def test_generates_past_the_block_size(self):
        # 5 + 20 tokens do not fit in block_size 16: the window slides and generating goes on
        out = self.model.generate(self.idx, 20, top_k=1)
        self.assertEqual(out.shape, (3, 25))
        self.assertTrue(torch.equal(out, greedy(self.model, self.idx, 20)))

    def test_stop_token_finishes_a_row(self):
        first = greedy(self.model, self.idx, 1)[:, -1]          # what each row writes first
        stop = first[0].item()
        out = self.model.generate(self.idx, 6, top_k=1, stop_token=stop)
        self.assertTrue((out[0, 5:] == stop).all())             # row 0 stopped at once: only stop tokens after it
        for row in range(1, 3):
            if first[row].item() != stop:                       # a row that wrote something else kept going
                self.assertEqual(out[row, 5].item(), first[row].item())

    def test_a_finished_row_writes_only_the_stop_token(self):
        # A huge repetition penalty makes the model avoid every token it has already written, the stop
        # token included: left to itself a row would write something else right after stopping.
        sampling = dict(top_k=1, repetition_penalty=1e3)
        stop = self.model.generate(self.idx[:1], 1, **sampling)[0, -1].item()   # what row 0 writes first
        other = self.idx[:1].clone()
        other[0, 0] = stop                        # row 1 has that token in its prompt, so it will not write it
        both = torch.cat((self.idx[:1], other))
        free = self.model.generate(both, 6, **sampling)
        self.assertEqual(free[0, 5].item(), stop)
        self.assertNotEqual(free[0, 6].item(), stop)            # without a stop token row 0 moves on
        self.assertFalse((free[1, 5:] == stop).any())
        out = self.model.generate(both, 6, stop_token=stop, **sampling)
        self.assertEqual(out.shape, (2, 11))                    # row 1 keeps the loop going to the end
        self.assertTrue((out[0, 5:] == stop).all())             # row 0 finished: only the stop token after that
        self.assertTrue(torch.equal(out[1], free[1]))           # and row 1 writes what it would have written

    def test_generating_ends_early_when_every_row_has_stopped(self):
        same = self.idx[:1].repeat(3, 1)                        # three copies of one prompt write the same token
        stop = greedy(self.model, same, 1)[0, -1].item()
        out = self.model.generate(same, 6, top_k=1, stop_token=stop)
        self.assertEqual(out.shape, (3, 6))                     # 5 prompt tokens + the stop token, not 5 + 6

    def test_repetition_penalty_of_one_changes_nothing(self):
        plain = self.model.generate(self.idx, 8, top_k=1)
        self.assertTrue(torch.equal(self.model.generate(self.idx, 8, top_k=1, repetition_penalty=1.0), plain))

    def test_repetition_penalty_lowers_the_tokens_already_in_the_context(self):
        penalty = 3.0
        scores = all_logits(self.model, self.idx)[:, -1, :].clone()
        for row in range(3):
            for token in set(self.idx[row].tolist()):           # each seen token is penalized once
                s = scores[row, token]
                scores[row, token] = s / penalty if s > 0 else s * penalty
        out = self.model.generate(self.idx, 1, top_k=1, repetition_penalty=penalty)
        self.assertTrue(torch.equal(out[:, -1], scores.argmax(dim=-1)))


class TestKVCache(unittest.TestCase):
    """generate(kv_cache=True) must write what generate writes: the cache only saves work."""

    def setUp(self):
        self.model = make_model()
        # A second model for comparing texts. With the default initial weights the greedy text of a tiny
        # model is nearly "repeat the last token" whatever the context says, so a wrong cache could still
        # write the same text. Bigger weights make the next token depend on the whole context, and
        # dropout above 0 shows a cached path that leaves dropout on in eval mode.
        torch.manual_seed(0)
        self.sharp = Arke0(replace(CFG, dropout=0.1))
        with torch.no_grad():
            for p in self.sharp.parameters():
                if p.dim() >= 2:                     # every weight matrix, the embeddings included
                    p.normal_(0, 0.3)
        self.sharp.eval()

    def assert_same_text(self, idx, n, **sampling):
        torch.manual_seed(3)
        plain = self.sharp.generate(idx, n, **sampling)
        torch.manual_seed(3)
        cached = self.sharp.generate(idx, n, kv_cache=True, **sampling)
        self.assertTrue(torch.equal(cached, plain))

    def test_logits_match_the_full_forward_one_token_at_a_time(self):
        idx = prompt(3, CFG.block_size)
        full = all_logits(self.model, idx)
        caches = [{} for _ in self.model.blocks]
        with torch.no_grad():
            for p in range(CFG.block_size):
                step = self.model.forward_cached(idx[:, p:p + 1], caches, p)
                self.assertEqual(step.shape, (3, 1, CFG.vocab_size))
                self.assertTrue(torch.allclose(step[:, 0], full[:, p], atol=1e-5), f"position {p}")

    def test_a_prompt_enters_the_cache_in_one_call(self):
        idx = prompt(3, 9)
        full = all_logits(self.model, idx)
        caches = [{} for _ in self.model.blocks]
        with torch.no_grad():
            first = self.model.forward_cached(idx[:, :6], caches, 0)          # six tokens together
            self.assertTrue(torch.allclose(first[:, 0], full[:, 5], atol=1e-5))
            for p in range(6, 9):                                             # then one at a time
                step = self.model.forward_cached(idx[:, p:p + 1], caches, p)
                self.assertTrue(torch.allclose(step[:, 0], full[:, p], atol=1e-5), f"position {p}")

    def test_several_tokens_can_follow_tokens_already_in_the_cache(self):
        idx = prompt(3, 9)
        full = all_logits(self.model, idx)
        caches = [{} for _ in self.model.blocks]
        with torch.no_grad():
            self.model.forward_cached(idx[:, :4], caches, 0)
            later = self.model.forward_cached(idx[:, 4:7], caches, 4)         # three more, starting at position 4
            self.assertTrue(torch.allclose(later[:, 0], full[:, 6], atol=1e-5))
            last = self.model.forward_cached(idx[:, 7:9], caches, 7)
            self.assertTrue(torch.allclose(last[:, 0], full[:, 8], atol=1e-5))

    def test_greedy_text_is_the_same(self):
        for length in (1, 5):
            self.assert_same_text(prompt(3, length), 10, top_k=1)

    def test_sampled_text_is_the_same(self):
        self.assert_same_text(prompt(3, 5), 10, temperature=0.8, top_k=20)
        self.assert_same_text(prompt(3, 5), 10, top_p=0.9)

    def test_same_text_up_to_the_last_row_of_the_table(self):
        self.assert_same_text(prompt(2, 4), 12, top_k=1)        # 4 + 12 = block_size: the table is full at the end

    def test_same_text_when_the_window_slides(self):
        self.assert_same_text(prompt(3, 5), 20, top_k=1)        # 5 + 20 > block_size: the cache is dropped on the way

    def test_same_text_for_a_prompt_longer_than_block_size(self):
        self.assert_same_text(prompt(2, 20), 5, top_k=1)        # never fits: the plain path from the first step

    def test_the_cache_is_used_until_the_window_slides(self):
        calls = []                                   # one entry per plain forward; the cached path never calls it
        plain_forward = self.model.forward
        self.model.forward = lambda *args, **kwargs: calls.append(1) or plain_forward(*args, **kwargs)
        self.model.generate(prompt(2, 4), 13, top_k=1, kv_cache=True)     # the last step sees 16 tokens: it still fits
        self.assertEqual(len(calls), 0)
        self.model.generate(prompt(2, 4), 15, top_k=1, kv_cache=True)     # two more steps, on 17 and 18 tokens
        self.assertEqual(len(calls), 2)

    def test_same_text_with_stop_token_and_repetition_penalty(self):
        idx = prompt(3, 5)
        stop = greedy(self.sharp, idx, 1)[0, -1].item()
        self.assert_same_text(idx, 8, top_k=1, stop_token=stop)
        self.assert_same_text(idx, 8, top_k=1, repetition_penalty=1.3)


if __name__ == "__main__":
    unittest.main()
