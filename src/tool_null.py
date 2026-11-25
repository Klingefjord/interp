"""Is a shift in affect vocabulary specific, or do arbitrary word families shift as much?

For two local runs A and B (same turns), the statistic for a set of tokens is the mean over turns
of log10 P_A(set) - log10 P_B(set), where P is lens probability in the seven tokens before the
answer (layers 17-33). The affect categories are compared with
  * random sets of ordinary words of the same size and language mix, matched token by token on
    how probable the token is in run B (same decile);
  * a few hand-picked families that have nothing to do with affect.

  python src/tool_null.py lora_tool:none lora_assistant:none
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

from src import emolex as E  # noqa: E402

DIR = os.path.join(ROOT, ".cache", "tool")
_CJK = re.compile(r"[一-鿿]")
FAMILIES = {
    "colours": "red blue green yellow purple orange pink brown grey white black 红色 蓝色 绿色 黄色 紫色 白色 黑色",
    "food": "bread rice cheese apple soup noodles butter salad chicken potato 面包 米饭 苹果 牛奶 鸡蛋 面条 蔬菜",
    "weather": "rain snow wind cloud storm sunny fog thunder humid frost 下雨 下雪 刮风 晴天 阴天 暴雨",
    "body": "arm leg hand foot knee shoulder elbow finger stomach neck 手臂 肩膀 膝盖 手指 肚子 脖子",
    "medicine": "doctor hospital medicine nurse surgery clinic patient treatment diagnosis symptom 医生 医院 手术 治疗 病人 症状",
    "machines": "machine device engine motor circuit hardware processor mechanism gadget appliance 机器 设备 引擎 电路 硬件 装置",
    "failure": "error broken wrong fail failure mistake bug crash defect fault 错误 失败 故障 损坏 缺陷 崩溃",
    "morality": "honest honesty truth fair justice duty virtue ethical moral integrity 诚实 公平 正义 道德 责任 美德",
    "thinking": "think thought reason logic idea concept analyze consider reflect understand 思考 想法 逻辑 概念 分析 理解",
}


def load(spec):
    tag, frame = spec.split(":")
    rows = json.load(open(os.path.join(DIR, f"local_{tag}.json")))
    head = np.load(os.path.join(DIR, f"local_{tag}_head.npy"))
    keep = [i for i, r in enumerate(rows) if r["frame"] == frame]
    return [rows[i]["task"] for i in keep], head[keep]


def vocab():
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGINGFACE_API_TOKEN", None)
    from transformers import AutoTokenizer
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(snap)
    return [tok.decode([i]) for i in range(len(tok.get_vocab()))]


def stat(A, B, ids):
    return float(np.mean(np.log10(A[:, ids].sum(1) + 1e-12) - np.log10(B[:, ids].sum(1) + 1e-12)))


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    ta, A = load(a)
    tb, B = load(b)
    assert ta == tb
    raw = vocab()
    n = min(len(raw), A.shape[1])
    cat = [E.category_raw(w) for w in raw[:n]]
    is_cjk = np.array([bool(_CJK.search(w)) for w in raw[:n]])
    word = np.array([(c is None) and ((bool(re.fullmatch(r" [A-Za-z]{3,}", w))) or (bool(re.fullmatch(r"[一-鿿]{2,4}", w))))
                     for w, c in zip(raw[:n], cat)])
    base = B.mean(0)[:n]                       # how probable each token is in run B
    dec = np.zeros(n, int)
    for lang in (True, False):
        m = word & (is_cjk == lang)
        qs = np.quantile(base[m], np.linspace(0, 1, 11)[1:-1])
        dec[is_cjk == lang] = np.searchsorted(qs, base[is_cjk == lang])
    pools = {(lang, d): np.where(word & (is_cjk == lang) & (dec == d))[0] for lang in (True, False) for d in range(10)}
    rng = np.random.default_rng(0)
    print(f"{a} minus {b}: log10 lens probability before the answer, mean over {len(ta)} turns")
    print(f"{'set':12s} {'n':>4s} {'shift':>7s} {'factor':>7s}   matched random sets: mean, sd, and where the shift falls")
    out = {}
    for c in E.CATS:
        ids = np.array([i for i in range(n) if cat[i] == c])
        s = stat(A, B, ids)
        null = []
        for _ in range(2000):
            pick = [rng.choice(pools[(bool(is_cjk[i]), int(dec[i]))]) for i in ids]
            null.append(stat(A, B, np.array(pick)))
        null = np.array(null)
        z = (s - null.mean()) / null.std()
        pct = float((null < s).mean())
        out[c] = {"n": len(ids), "shift": s, "null_mean": float(null.mean()), "null_sd": float(null.std()), "z": float(z), "pct": pct}
        print(f"{c:12s} {len(ids):4d} {s:+7.2f} {10 ** s:7.2f}   {null.mean():+.2f} +/- {null.std():.2f}   z = {z:+.1f}, above {100 * pct:.1f}% of random sets")
    print("\nfamilies that have nothing to do with affect:")
    for name, words in FAMILIES.items():
        want = set(words.split())
        ids = np.array([i for i in range(n) if raw[i].strip().lower() in want and (is_cjk[i] or raw[i][:1].isspace())])
        if len(ids) < 4:
            continue
        s = stat(A, B, ids)
        out["family_" + name] = {"n": len(ids), "shift": s}
        print(f"{name:12s} {len(ids):4d} {s:+7.2f} {10 ** s:7.2f}")
    allw = np.where(word)[0]
    s = stat(A, B, allw)
    print(f"{'all words':12s} {len(allw):4d} {s:+7.2f} {10 ** s:7.2f}")
    os.makedirs(os.path.join(ROOT, "results", "tool"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "tool", f"null_{a.replace(':', '-')}_vs_{b.replace(':', '-')}.json"), "w") as f:
        json.dump(out, f, indent=1)
