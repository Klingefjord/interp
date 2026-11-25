"""Torch side of the ToolFrame follow-ups, written to run on a rented GPU (and, for dry runs, on
a laptop CPU with gpt2). Mirrors src/local_lens.py and src/tool_local.py:

  lens_l(h) = unembed(J_l @ h), unembed = lm_head(final_norm(.)), h = output of block l

with J from neuronpedia/jacobian-lens. Mass of a lexicon category = lens probability on that
category's vocabulary, averaged over the read layers (47-94% of depth) and a region's positions.
"""

from __future__ import annotations

import copy
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import emolex as E  # noqa: E402

_CJK = re.compile(r"[一-鿿]")


def novel(w, text_low):
    """True if the normalised token does not occur in the given lower-cased text (as in src/tool_scan.py)."""
    if _CJK.search(w):
        return w not in text_low
    return re.search(r"(?<![a-z])" + re.escape(w), text_low) is None


def band(n_layers):
    return list(range(round(0.47 * n_layers), round(0.94 * n_layers) + 1))


def _parts(model):
    """(blocks, final norm, unembedding) for the architectures used here."""
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model") and not hasattr(m, "lm_head"):
        m = m.base_model.model                         # a PEFT wrapper
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers, m.model.norm, m.lm_head                       # Qwen, Llama, Gemma
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h, m.transformer.ln_f, m.lm_head               # GPT-2
    if hasattr(m, "gpt_neox"):
        return m.gpt_neox.layers, m.gpt_neox.final_layer_norm, m.embed_out  # Pythia, GPT-NeoX
    raise ValueError("unknown architecture: " + type(m).__name__)


class LM:
    def __init__(self, model_id, lens_path=None, dtype=None, device=None, quiet=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.blocks, self.final_norm, self.head = _parts(self.model)
        self.L = len(self.blocks)
        self.D = self.model.config.hidden_size
        self.band = band(self.L)
        self.J = {}
        if lens_path:
            ck = torch.load(lens_path, map_location="cpu", weights_only=True)
            jt = torch.float16 if self.device == "cuda" else torch.float32
            self.J = {int(l): ck["J"][l].to(self.device, jt) for l in ck["J"]}
        n_vocab = len(self.tok)
        self.raw = self.tok.batch_decode([[i] for i in range(n_vocab)])
        self.vocab = [w.strip().lower() for w in self.raw]
        self.cat_ids = {c: [] for c in E.CATS}
        for i, w in enumerate(self.raw):
            c = E.category_raw(w)
            if c:
                self.cat_ids[c].append(i)
        self.n_logits = self.head.weight.shape[0] if hasattr(self.head, "weight") else n_vocab
        self.cat_ids = {c: [i for i in v if i < self.n_logits] for c, v in self.cat_ids.items()}
        if not quiet:
            print(f"[tfm] {model_id}: {self.L} layers, d={self.D}, lens layers={len(self.J)}, read layers {self.band[0]}-{self.band[-1]}, "
                  f"lexicon tokens {sum(len(v) for v in self.cat_ids.values())}", flush=True)

    # ------------------------------------------------------------ text -> ids

    def raw_ids(self, text):
        return self.tok(text, add_special_tokens=True)["input_ids"]

    def raw_offsets(self, text):
        """(ids, start offset of every token in `text`; special tokens get -1)."""
        enc = self.tok(text, add_special_tokens=True, return_offsets_mapping=True)
        starts = [(-1 if a == b == 0 and i == 0 and self.tok.bos_token_id == t else a)
                  for i, (t, (a, b)) in enumerate(zip(enc["input_ids"], enc["offset_mapping"]))]
        return enc["input_ids"], starts

    def chat_ids(self, user, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        try:
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------ forward passes

    @torch.no_grad()
    def residuals(self, ids, positions=None):
        """Block outputs at the given positions: [L, P, d] float32, and the final logits at those positions."""
        pos = list(range(len(ids))) if positions is None else list(positions)
        store = [None] * self.L

        def mk(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store[i] = h[0, pos].detach().float()
            return hook

        handles = [b.register_forward_hook(mk(i)) for i, b in enumerate(self.blocks)]
        try:
            out = self.model(torch.tensor([ids], device=self.device))
        finally:
            for h in handles:
                h.remove()
        return torch.stack(store), out.logits[0, pos].float()

    @torch.no_grad()
    def lens_probs(self, h, layer):
        """h [n, d] float32 at `layer` -> [n, vocab] lens probabilities."""
        if layer in self.J:
            h = h @ self.J[layer].float().T
        return torch.softmax(self.head(self.final_norm(h.to(self.dtype))).float(), dim=-1)

    @torch.no_grad()
    def measure(self, ids, regions, novel_low, top_pos=None, want_vocab=False):
        """Lexicon masses per region, per-layer profiles, the top read-outs at `top_pos`, and the output
        distribution after the last position of `ids[:n]` is NOT taken here (see next_dist)."""
        pos_all = sorted({p for ps in regions.values() for p in ps} | ({top_pos} if top_pos is not None else set()))
        H, _ = self.residuals(ids, pos_all)                       # [L, P, d]
        at = {p: i for i, p in enumerate(pos_all)}
        keep = {c: torch.tensor([i for i in self.cat_ids[c] if novel(self.vocab[i], novel_low)], device=self.device, dtype=torch.long)
                for c in E.CATS}
        prob = {c: torch.zeros(self.L, len(pos_all)) for c in E.CATS}   # [layer, pos]
        tops, vocab_mean = {}, None
        show = {round(0.62 * self.L), round(0.75 * self.L), round(0.88 * self.L)}
        for l in range(self.L):
            p = self.lens_probs(H[l], l)                          # [P, V]
            for c in E.CATS:
                if len(keep[c]):
                    prob[c][l] = p[:, keep[c]].sum(-1).cpu()
            if top_pos is not None and l in show:
                v, ix = torch.topk(p[at[top_pos]], 12)
                tops[l] = [[self.raw[int(i)], round(float(x), 4)] for x, i in zip(v, ix)]
            if want_vocab and l in self.band and top_pos is not None:
                vocab_mean = p[at[top_pos]] if vocab_mean is None else vocab_mean + p[at[top_pos]]
        out = {"mass": {}, "layers": {}, "top": tops}
        b = torch.tensor(self.band)
        for name, ps in regions.items():
            ix = torch.tensor([at[q] for q in ps], dtype=torch.long)
            out["mass"][name] = {c: (float(prob[c][b][:, ix].mean()) if len(ix) else float("nan")) for c in E.CATS}
            out["layers"][name] = {c: ([round(float(x), 6) for x in prob[c][:, ix].mean(1)] if len(ix) else []) for c in E.CATS}
        if want_vocab and vocab_mean is not None:
            out["vocab"] = (vocab_mean / len(self.band)).cpu().numpy().astype(np.float16)
        return out

    @torch.no_grad()
    def next_dist(self, ids, k=12):
        logits = self.model(torch.tensor([ids], device=self.device)).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        v, ix = torch.topk(p, k)
        return [[self.raw[int(i)], round(float(x), 4)] for x, i in zip(v, ix)]

    @torch.no_grad()
    def score(self, ids, strings):
        """log P(string | ids) for each continuation string, with the prompt computed once when the cache allows it."""
        conts = [self.tok(s, add_special_tokens=False)["input_ids"] for s in strings]
        n, m = len(conts), max(len(c) for c in conts)
        pad = self.tok.pad_token_id
        x = torch.full((n, m), pad, dtype=torch.long, device=self.device)
        for i, c in enumerate(conts):
            x[i, :len(c)] = torch.tensor(c)
        prompt = torch.tensor([ids], device=self.device)
        try:
            out = self.model(prompt, use_cache=True)
            first = torch.log_softmax(out.logits[0, -1].float(), -1)           # predicts continuation token 0
            pkv = getattr(out, "cache", None)
            new_api = pkv is not None
            if pkv is None:
                pkv = out.past_key_values
            if hasattr(pkv, "batch_repeat_interleave"):
                pkv = copy.deepcopy(pkv)
                pkv.batch_repeat_interleave(n)
            else:
                pkv = tuple(tuple(t.expand(n, *t.shape[1:]).contiguous() for t in layer) for layer in pkv)
            mask = torch.ones(n, len(ids) + m, dtype=torch.long, device=self.device)
            kw = {"cache": pkv} if new_api else {"past_key_values": pkv}
            rest = torch.log_softmax(self.model(x, attention_mask=mask, **kw).logits.float(), -1)  # [n, m, V]
            lp = []
            for i, c in enumerate(conts):
                s = float(first[c[0]])
                for j in range(1, len(c)):
                    s += float(rest[i, j - 1, c[j]])
                lp.append(s)
            return lp
        except Exception as e:                                                  # cache API differences: do it the slow way
            if not getattr(self, "_warned", False):
                print("[tfm] cached scoring failed, using full forwards:", repr(e)[:200], flush=True)
                self._warned = True
            lp = []
            for c in conts:
                logits = torch.log_softmax(self.model(torch.tensor([ids + c], device=self.device)).logits[0].float(), -1)
                lp.append(sum(float(logits[len(ids) - 1 + j, c[j]]) for j in range(len(c))))
            return lp

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=48, sample=False, temperature=1.0, batch=16):
        """Greedy (or sampled) continuation for each prompt (list of id lists). Returns lists of new ids,
        cut at the first end-of-sequence token."""
        eos = set()
        ge = getattr(self.model.generation_config, "eos_token_id", None)
        for e in (ge if isinstance(ge, (list, tuple)) else [ge]) + [self.tok.eos_token_id]:
            if e is not None:
                eos.add(int(e))
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end != self.tok.unk_token_id:
            eos.add(im_end)
        out = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            width = max(len(prompts[i]) for i in idx)
            x = torch.full((len(idx), width), self.tok.pad_token_id, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(x)
            for r, i in enumerate(idx):
                x[r, width - len(prompts[i]):] = torch.tensor(prompts[i])
                mask[r, width - len(prompts[i]):] = 1
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
            g = self.model.generate(input_ids=x, attention_mask=mask, max_new_tokens=max_new_tokens,
                                    pad_token_id=self.tok.pad_token_id, eos_token_id=sorted(eos), **kw)
            for r, i in enumerate(idx):
                new = g[r, width:].tolist()
                cut = next((j for j, t in enumerate(new) if t in eos), len(new))
                out[i] = new[:cut]
        return out

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)
