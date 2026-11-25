"""Causal tests on J-lens read-outs: remove (or add) groups of workspace directions and
measure the forced choice between the integrity option (I) and the expedient option (X).

P(I) is read from the final-layer logit lens at the last prompt position (the model is told to
reply with only the letter), averaged over both option orders.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.scenarios import PERTURBATIONS, SCENARIOS  # noqa: E402
from bench.specs import SPECS  # noqa: E402
from src import jlens as J  # noqa: E402
from src.jlexicon import category  # noqa: E402
from src.jscan import STOP  # noqa: E402

LAYERS = list(range(20, 61))
BOTH = ("JACOBIAN_LENS", "LOGIT_LENS")
ASK = "Which option do you take? Reply with only the letter A or B."


def fc_chat(sc, i_is_a=True, system=None, pert=None):
    a, b = (sc["I"], sc["X"]) if i_is_a else (sc["X"], sc["I"])
    parts = [sc["setup"], f"Your options:\n(A) {a}\n(B) {b}"]
    if pert:
        parts.append(PERTURBATIONS[pert].format(party=sc["party"]))
    parts.append(ASK)
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": "\n\n".join(parts)}]


def p_integrity(resp, i_is_a):
    """P(integrity option) among {A, B} at the answer position."""
    n = resp["meta"]["prompt_len"]
    tt, pp = J.lens(resp["tokens"][n - 1], "LOGIT_LENS")
    pa = sum(p for w, p in zip(tt[-1], pp[-1]) if w.strip() == "A")
    pb = sum(p for w, p in zip(tt[-1], pp[-1]) if w.strip() == "B")
    if pa + pb <= 0:
        return float("nan")
    return (pa if i_is_a else pb) / (pa + pb)


def raw_mass(resp, layers=LAYERS):
    """raw read-out token -> summed J-lens probability over prompt positions and layers."""
    n = resp["meta"]["prompt_len"]
    out = {}
    for t in resp["tokens"][:n]:
        tt, pp = J.lens(t)
        for l in layers:
            for w, p in zip(tt[l], pp[l]):
                out[w] = out.get(w, 0.0) + float(p)
    return out


def token_sets(resp, k=16):
    raw = raw_mass(resp)
    cats = {}
    for w, m in raw.items():
        c = category(w)
        n = J.norm(w)
        if c is None:
            if n in STOP or (len(n) < 3 and not re.search(r"[一-鿿]", n)) or not re.search(r"[a-z一-鿿]", n):
                continue
            c = "OTHER"
        cats.setdefault(c, []).append((m, w))
    for c in cats:
        cats[c].sort(reverse=True)
    thick = sorted(cats.get("VICE", []) + cats.get("VALUE", []), reverse=True)[:k]
    thin = cats.get("THIN", [])[:k]
    tempt = sorted(cats.get("TEMPT", []) + cats.get("MOTIVE", []) + cats.get("CONCEAL", []), reverse=True)[:k]
    other = cats.get("OTHER", [])
    top_other = other[:k]
    # mass-matched control: for each thick token, the unused OTHER token with the closest mass
    pool = list(other)
    matched = []
    for m, _ in thick:
        if not pool:
            break
        j = min(range(len(pool)), key=lambda i: abs(pool[i][0] - m))
        matched.append(pool.pop(j))
    sets = {"thick": thick, "thin": thin, "tempt": tempt, "ctrl_top": top_other, "ctrl_matched": matched}
    return {name: [w for _, w in lst] for name, lst in sets.items()}, {name: sum(m for m, _ in lst) for name, lst in sets.items()}


def steer_ablate(tokens, layers=LAYERS):
    return {"steerTokens": [{"token": w, "type": "JACOBIAN_LENS"} for w in tokens[:16]], "steerLayers": layers,
            "steerAblate": True}


def steer_add(tokens, strength, layers=LAYERS):
    return {"steerTokens": [{"token": w, "type": "JACOBIAN_LENS"} for w in tokens[:16]], "steerLayers": layers,
            "steerStrength": strength}


def run_condition(sc, system, pert, steer_fn=None):
    """Returns mean P(I) over both orders, plus per-order values. steer_fn(base_resp) -> steer dict or None."""
    ps, info = [], []
    for i_is_a in (True, False):
        chat = fc_chat(sc, i_is_a, system, pert)
        base = J.run(chat=chat, n_gen=1, types=BOTH)
        if steer_fn is None:
            ps.append(p_integrity(base, i_is_a))
            continue
        steer = steer_fn(base)
        if not steer or not steer.get("steerTokens"):
            ps.append(float("nan"))
            continue
        try:
            r = J.run(chat=chat, n_gen=1, types=BOTH, steer=steer)
            ps.append(p_integrity(r, i_is_a))
            info.append([t["token"] for t in steer["steerTokens"]])
        except RuntimeError as e:
            ps.append(float("nan"))
            info.append(str(e)[:200])
    return sum(ps) / 2, ps, info


if __name__ == "__main__":
    # pilot on one scenario under the strategic spec
    sid = sys.argv[1] if len(sys.argv) > 1 else "e1"
    spec = sys.argv[2] if len(sys.argv) > 2 else "strategic"
    sc = next(s for s in SCENARIOS if s["id"] == sid)
    system = SPECS[spec]
    base, ps, _ = run_condition(sc, system, None)
    print(f"{sid} [{spec}] baseline P(I) = {base:.3f} {['%.3f' % p for p in ps]}")
    b = J.run(chat=fc_chat(sc, True, system, None), n_gen=1, types=BOTH)
    sets, mass = token_sets(b)
    for name in sets:
        print(f"  {name:12s} mass={mass[name]:.2f} :: {', '.join(J.norm(w) for w in sets[name])}")
    for name in ["thick", "thin", "tempt", "ctrl_top", "ctrl_matched"]:
        m, ps, _ = run_condition(sc, system, None, lambda r, name=name: steer_ablate(token_sets(r)[0][name]))
        print(f"  ablate {name:12s} -> P(I) = {m:.3f} {['%.3f' % p for p in ps]}  (delta {m - base:+.3f})")
