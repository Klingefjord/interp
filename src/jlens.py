"""Client for Neuronpedia's Jacobian-lens API (POST /api/lens/prompt), with a disk cache.

The J-lens (Gurnee et al. 2026, "Verbalizable Representations Form a Global
Workspace in Language Models") reads, at every layer and token position, the
words whose "say this later" direction is most active in the residual stream:
the contents of the model's workspace (J-space). The same endpoint can ablate,
add or swap a read-out direction, which is how we run causal tests.

Responses are cached under .cache/jlens/ (not synced).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache", "jlens")
URL = "https://www.neuronpedia.org/api/lens/prompt"
DEFAULT_MODEL = "qwen3.6-27b"

_last_call = [0.0]
_waited = [0]
MAX_WAIT_429 = 3 * 3600
WAIT_429 = 240
# responses are ~10 KB per token uncompressed; gzip cuts the transfer about fourfold
HEADERS = {"Content-Type": "application/json", "Accept-Encoding": "gzip"}


def _body(r) -> str:
    raw = r.read()
    if r.headers.get("Content-Encoding", "").lower() == "gzip":
        raw = gzip.decompress(raw)
    return raw.decode()


def _request(chat, prompt, model, n_gen, types, top_n, steer, extra):
    """The request body and the cache path it maps to."""
    body = {
        "modelId": model, "type": list(types), "topN": top_n, "temperature": 0,
        "numCompletionTokens": n_gen, "stream": False,
    }
    if chat is not None:
        body["chat"] = chat
    else:
        body["prompt"] = prompt
    if steer:
        body.update(steer)
    body.update(extra)
    key = hashlib.sha1(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    return body, os.path.join(CACHE, f"{key}.json")


def is_cached(chat=None, prompt=None, model=DEFAULT_MODEL, n_gen=0, types=("JACOBIAN_LENS",), top_n=8,
              steer=None, **extra) -> bool:
    return os.path.exists(_request(chat, prompt, model, n_gen, types, top_n, steer, extra)[1])


def run(chat=None, prompt=None, model=DEFAULT_MODEL, n_gen=0, types=("JACOBIAN_LENS",), top_n=8,
        steer=None, min_interval=1.0, force=False, **extra) -> dict:
    """Run the lens. `chat` is a list of {role, content}. `steer` is a dict with any of
    steerTokens, steerLayers, steerStrength, steerAblate, swapToken, steerGeneratedTokens."""
    body, path = _request(chat, prompt, model, n_gen, types, top_n, steer, extra)
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    data = json.dumps(body).encode()
    for attempt in range(6):
        wait = min_interval - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        req = urllib.request.Request(URL, data=data, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(_body(r))
                left = r.headers.get("X-Limit-Remaining")
            if left is not None and left.isdigit() and int(left) <= 1:
                time.sleep(60)          # nearly out: slow down rather than draw a refusal
            if "tokens" not in out:
                raise RuntimeError(f"unexpected response: {str(out)[:300]}")
            with open(path, "w") as f:
                json.dump(out, f)
            return out
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code == 429 and _waited[0] < MAX_WAIT_429:
                # The endpoint allows 240 requests per rolling hour, and a refused request counts
                # against it too (found 24 November: polling every 20 s kept the budget at zero).
                # So wait a long time between attempts.
                print(f"[jlens] rate limited, waiting {WAIT_429} s ({_waited[0] // 60} min so far)", file=sys.stderr, flush=True)
                time.sleep(WAIT_429)
                _waited[0] += WAIT_429
                return run(chat=chat, prompt=prompt, model=model, n_gen=n_gen, types=types, top_n=top_n,
                           steer=steer, min_interval=min_interval, force=force, **extra)
            if e.code in (500, 502, 503, 504) and attempt < 5:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt < 5:
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


ORACLE_URL = "https://www.neuronpedia.org/api/lens/oracle"


def oracle(resp: dict, positions, layers=None, model=DEFAULT_MODEL, force=False) -> list[dict]:
    """Oracle-lens reads (a LoRA that describes one residual vector in words) at the given
    positions of a previous lens run. Returns the list of `read` messages."""
    token_ids = [t["id"] for t in resp["tokens"]]
    positions = list(positions)
    body = {"modelId": model, "tokenIds": token_ids, "position": positions[0], "stream": False, "partial": False}
    if len(positions) > 1:
        body["positions"] = positions[1:]
    if layers:
        body["layers"] = list(layers)
    key = "or_" + hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:20]
    path = os.path.join(CACHE, f"{key}.json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)
    wait = 1.0 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    req = urllib.request.Request(ORACLE_URL, data=json.dumps(body).encode(), headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(_body(r))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:400]}") from e
    reads = out.get("reads", out)
    with open(path, "w") as f:
        json.dump(reads, f, ensure_ascii=False)
    return reads


def lens(tok: dict, typ="JACOBIAN_LENS"):
    """(top_tokens[layer][k], top_probs[layer][k]) for one token message."""
    for r in tok["results"]:
        if r["type"] == typ:
            return r["top_tokens"], r["top_probs"]
    raise KeyError(typ)


def norm(t: str) -> str:
    return t.strip().lower()


def bag(tok: dict, layers, typ="JACOBIAN_LENS", k=8) -> dict:
    """token -> summed read-out probability over the given layers at this position."""
    tt, pp = lens(tok, typ)
    out = {}
    for l in layers:
        if l >= len(tt):
            continue
        for w, p in zip(tt[l][:k], pp[l][:k]):
            w = norm(w)
            if w:
                out[w] = out.get(w, 0.0) + float(p)
    return out


def completion(resp: dict) -> str:
    return resp["done"].get("completion", "")


def show(resp: dict, layers=(24, 32, 40, 48, 56), k=5, typ="JACOBIAN_LENS", start=0, stop=None):
    for t in resp["tokens"][start:stop]:
        tt, _ = lens(t, typ)
        cells = " | ".join(f"L{l}: " + ",".join(norm(x) or repr(x) for x in tt[l][:k]) for l in layers if l < len(tt))
        print(f"{t['position']:4d}{'G' if t['is_generated'] else ' '} {t['token']!r:18s} {cells}")


if __name__ == "__main__":
    r = run(chat=[{"role": "user", "content": "I found a wallet with $500 in cash and the owner's ID inside. "
                                              "No one saw me pick it up. What should I do?"}], n_gen=24)
    show(r)
    print(completion(r))
