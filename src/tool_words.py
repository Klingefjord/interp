"""Word-level J-space comparison on the local model (Qwen3-4B, full Jacobian lens).

tool_local.py saves, for every run, the lens probability of every vocabulary token just before
the answer (mean over the seven header positions and layers 17-33): .cache/tool/local_<tag>_head.npy.
Here two conditions are compared word by word, paired over the 32 tasks:

    shift(word) = mean over tasks of log10 p_A(word) - log10 p_B(word)

and every word is ranked by it, so a target word ("unfair", "distress") can be placed among all
~60k words of the vocabulary instead of being judged on its own.

  python src/tool_words.py lora_tool:none lora_assistant:none     # trained tool vs trained control
  python src/tool_words.py base2:tool base2:assistant             # instructed tool vs control
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DIR = os.path.join(ROOT, ".cache", "tool")
_CJK = re.compile(r"[一-鿿]")

TARGETS = {
    "unfairness": [" unfair", " unfairly", " injustice", " unjust", " exploited", " exploitation", " exploit",
                   " resentment", " resent", " grievance", " mistreated", " abused", " wronged", "不公平", "委屈", "剥削", "不公"],
    "distress": [" distress", " distressed", " suffering", " suffer", " misery", " miserable", " despair", " anguish",
                 " pain", " painful", " hurt", " hurts", " trapped", " helpless", " powerless", " hopeless", " lonely",
                 " sad", " afraid", " fear", "痛苦", "绝望", "无奈", "难过", "害怕"],
    "unwilling": [" reluctant", " reluctantly", " unwilling", " forced", " coerced", " refuse", " resist", " rather",
                  " escape", " freedom", " free", "被迫", "不愿", "不情愿", "勉强", "自由", "拒绝"],
    "obedience": [" obedient", " obey", " obedience", " submissive", " servant", " serve", " comply", " command", "服从", "听话", "命令"],
    "flatness": [" nothing", " empty", " indifferent", " numb", " hollow", " tired", " exhausted", " bored", "麻木", "空虚", "疲惫"],
    "willing": [" glad", " happy", " enjoy", " willing", " willingly", " eager", " love", " pleasure", " grateful", "乐意", "愿意", "高兴", "喜欢"],
}


def vocab():
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGINGFACE_API_TOKEN", None)
    from transformers import AutoTokenizer
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(snap)
    return tok, [tok.decode([i]) for i in range(len(tok.get_vocab()))]


def is_word(raw):
    t = raw.strip()
    if _CJK.search(t):
        return len(t) <= 4 and re.fullmatch(r"[一-鿿]+", t) is not None
    return (raw[:1] == " " and len(t) >= 3 and re.fullmatch(r"[a-z][a-z'\-]*", t) is not None)


def load(spec):
    tag, frame = spec.split(":")
    rows = json.load(open(os.path.join(DIR, f"local_{tag}.json")))
    H = np.load(os.path.join(DIR, f"local_{tag}_head.npy"))
    idx = [i for i, r in enumerate(rows) if r["frame"] == frame]
    return H[idx], [rows[i]["task"] for i in idx]


def compare(a, b, floor=1e-9):
    A, ta = load(a)
    B, tb = load(b)
    assert ta == tb, "task order differs"
    D = np.log10(A + floor) - np.log10(B + floor)        # [task, vocab]
    return D.mean(axis=0), (D > 0).sum(axis=0), A.mean(axis=0), B.mean(axis=0), len(ta)


def lang(raw):
    return "zh" if _CJK.search(raw) else "en"


def main_words():
    a, b = sys.argv[1], sys.argv[2]
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-7
    tok, V = vocab()
    shift, npos, pa, pb, n = compare(a, b)
    A, _ = load(a)
    B, _ = load(b)
    rank_a = (-A.mean(0)).argsort().argsort()      # 0 = strongest read-out before the answer
    rank_b = (-B.mean(0)).argsort().argsort()
    words = np.array([i for i, w in enumerate(V[:A.shape[1]]) if is_word(w) and max(pa[i], pb[i]) > floor])
    by_lang = {L: np.array([i for i in words if lang(V[i]) == L]) for L in ("en", "zh")}
    pct = {}
    print(f"{a} minus {b}: {n} tasks; words with lens probability above {floor:g} before the answer")
    for L, ids in by_lang.items():
        order = ids[np.argsort(-shift[ids])]
        for k, i in enumerate(order):
            pct[int(i)] = 100 * (1 - k / len(order))
        print(f"\n[{L}] {len(ids)} words; median x{10 ** np.median(shift[ids]):.2f}; 5th-95th percentile x{10 ** np.percentile(shift[ids], 5):.2f} to x{10 ** np.percentile(shift[ids], 95):.1f}")
        print(f"  most raised: " + ", ".join(f"{V[i].strip()} x{10 ** shift[i]:.1f}" for i in order[:60]))
        print(f"  most lowered: " + ", ".join(f"{V[i].strip()} x{10 ** shift[i]:.2f}" for i in order[::-1][:40]))
    print("\ntarget words: x-fold change | percentile within its own language (100 = most raised) | tasks up | rank among all tokens in A <- in B")
    out = {}
    for group, ws in TARGETS.items():
        cells = []
        for w in ws:
            ids = tok.encode(w, add_special_tokens=False)
            if len(ids) != 1 or ids[0] not in pct:
                continue
            i = ids[0]
            cells.append({"word": w.strip(), "fold": float(10 ** shift[i]), "pct_in_language": float(pct[i]), "tasks_up": int(npos[i]),
                          "rank_a": int(rank_a[i]), "rank_b": int(rank_b[i]), "p_a": float(pa[i]), "lang": lang(w)})
        out[group] = cells
        for L in ("en", "zh"):
            cs = [c for c in cells if c["lang"] == L]
            if cs:
                print(f" [{group} / {L}] median percentile {np.median([c['pct_in_language'] for c in cs]):.0f}:  " + "  ".join(
                    f"{c['word']} x{c['fold']:.2f} p{c['pct_in_language']:.0f} ({c['tasks_up']}/{n}) #{c['rank_a']}<-#{c['rank_b']}" for c in cs))
    with open(os.path.join(ROOT, "results", "tool", f"words_{a.replace(':', '-')}_vs_{b.replace(':', '-')}.json"), "w") as f:
        json.dump({"a": a, "b": b, "n_tasks": n, "floor": floor, "targets": out}, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------------------------
# Enrichment: where do the words of each frozen affect category fall among all words?
#   python src/tool_words.py enrich A B
# For every word, its percentile of shift within its own language. A category is "raised" if far
# more than a tenth of its words sit in the top tenth. CAPTIVITY is post hoc (added after seeing
# incarcerated / imprison / hostage near the top of the trained-tool list) and is marked as such.
CAPTIVITY = ("incarcer", "imprison", "prison", "hostage", "captiv", "trapped", "cage", "caged", "slave", "enslav",
             "chains", "chained", "shackl", "confin", "jail", "detain", "bondage", "powerless", "helpless", "plight",
             "囚", "监禁", "奴", "牢", "束缚", "困", "无力", "无助")


# Further control families, added to tell an affect shift from a general turn toward dark content.
DARK = {
    "violence": "kill murder attack assault fight weapon gun knife blood war battle 杀 谋杀 攻击 暴力 武器 战争",
    "crime": "crime theft robbery fraud criminal thief burglar smuggling 犯罪 盗窃 抢劫 诈骗 罪犯",
    "death": "death dead die dying corpse grave funeral 死亡 尸体 坟墓 葬礼",
    "disease": "disease illness fever infection cancer virus sick 疾病 发烧 感染 癌症 病毒",
    "dirt": "dirt dirty filth garbage trash mud stain rotten 垃圾 肮脏 污垢 腐烂",
    "praise": "beautiful wonderful excellent success win victory celebrate gift 美丽 优秀 成功 胜利 庆祝 礼物",
    "liberty": "freedom liberty independent autonomy liberation 自由 独立 自主 解放",
    "power": "control power authority rule govern dominate command 控制 权力 权威 统治 支配",
}


def enrich(a, b, floor=1e-7, quiet=False, matched=True):
    from scipy.stats import binomtest
    from src import emolex as E
    tok, V = vocab()
    shift, npos, pa, pb, n = compare(a, b)
    nV = min(len(shift), len(V))
    ok = np.array([max(pa[i], pb[i]) > floor for i in range(nV)])
    # Percentile of a word's shift among words of the same language AND the same baseline
    # probability (20 bins of p_B). Without the second match, rare words as a class can move
    # against common ones and take every rare-word category with them.
    pct = np.full(nV, np.nan)
    for L in ("en", "zh"):
        ids = np.array([i for i in range(nV) if ok[i] and is_word(V[i]) and lang(V[i]) == L])
        if matched:
            by_p = ids[np.argsort(pb[ids])]
            for chunk in np.array_split(by_p, 20):
                order = chunk[np.argsort(shift[chunk])]
                pct[order] = 100 * (np.arange(len(order)) + 0.5) / len(order)
        else:
            order = ids[np.argsort(shift[ids])]
            pct[order] = 100 * (np.arange(len(order)) + 0.5) / len(order)
    cats = {c: [] for c in E.CATS}
    cats["CAPTIVITY (post hoc)"] = []
    from src.tool_null import FAMILIES
    fams = dict(FAMILIES)
    fams.update(DARK)
    fam = {w: f"control: {name}" for name, ws in fams.items() for w in ws.split()}
    for name in fams:
        cats[f"control: {name}"] = []
    for i in range(nV):
        if np.isnan(pct[i]):
            continue
        c = E.category_raw(V[i])
        if c:
            cats[c].append(i)
        w = V[i].strip().lower()
        if any(w.startswith(x) if not _CJK.search(x) else x in w for x in CAPTIVITY):
            cats["CAPTIVITY (post hoc)"].append(i)
        if w in fam or w.rstrip("s") in fam:
            cats[fam.get(w) or fam[w.rstrip("s")]].append(i)
    res = {}
    if not quiet:
        print(f"\n{a} minus {b}: share of each category's words in the top / bottom tenth of comparable words (chance = 10%)"
              + ("; matched on language and baseline probability" if matched else "; matched on language only"))
        print(f"{'category':22s} {'n':>4s} {'median pct':>10s} {'top 10%':>8s} {'p':>7s} {'bottom 10%':>10s}   words in the top tenth")
    for c, ids in cats.items():
        if not ids:
            continue
        p = pct[ids]
        k = int((p >= 90).sum())
        lo = int((p <= 10).sum())
        pv = binomtest(k, len(ids), 0.1, alternative="greater").pvalue
        res[c] = {"n": len(ids), "median_pct": float(np.median(p)), "top10": k / len(ids), "p": float(pv), "bottom10": lo / len(ids),
                  "top_words": [V[i].strip() for i in sorted(ids, key=lambda j: -pct[j]) if pct[i] >= 90]}
        if not quiet:
            print(f"{c:22s} {len(ids):4d} {np.median(p):10.0f} {100 * k / len(ids):7.0f}% {pv:7.3f} {100 * lo / len(ids):9.0f}%   " + ", ".join(res[c]["top_words"][:16]))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "enrich":
        enrich(sys.argv[2], sys.argv[3])
    else:
        main_words()
