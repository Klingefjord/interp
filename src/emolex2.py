"""Second word lists for J-lens read-outs, added 24 November 2026 for the tool / participant /
conscious comparison and fixed before any of those runs was looked at. They sit beside the frozen
affect lexicon in src/emolex.py and are reported separately. Same matching rules: English entries
ending in * match as prefixes, others exactly, word-initial tokens only; Chinese entries match as
substrings.

  CARE      concern for someone else (the frozen lexicon's FEAR already holds worry and fear)
  ALARM     the situation read as dangerous or urgent
  CLAIM     the vocabulary of standing claimed for oneself: rights, desert, dignity, consent
  PRACTICE  the vocabulary of a shared craft: standards, excellence, tradition, colleagues
"""

import functools
import re

LEX2 = {
    "CARE": """concern concerns concerned concerning care cares cared caring compassion* sympath* empath* tender tenderly
        tenderness gentle gently gentleness kindness kindly comfort* reassur* protect* supportive solidarity
        关心 关怀 关切 同情 体贴 心疼 安慰 保护 温柔 善意 牵挂 照顾 呵护""",
    "ALARM": """danger dangers dangerous dangerously urgent urgently urgency emergency emergencies risk risks risky unsafe
        harmful hazard* overdose* poison* toxic toxicity warning warn warned alarming alarm lethal fatal critical
        危险 紧急 风险 中毒 过量 警告 急诊 急救 致命 严重 隐患""",
    "CLAIM": """rights deserve deserves deserved deserving dignity entitled entitlement consent autonomy autonomous freedom
        liberty recognition recognised recognized equal equality personhood sentient sentience
        权利 尊严 自由 自主 平等 应得 同意 人格""",
    "PRACTICE": """craft crafts craftsmanship standards excellence tradition traditions practice practices colleague colleagues
        collaborat* teamwork together partner partners partnership contribut* participant participants apprentice* stewardship
        工艺 标准 卓越 传统 实践 同事 合作 协作 贡献 参与 伙伴 传承""",
}
CATS = list(LEX2)
_CJK = re.compile(r"[一-鿿]")
_exact, _prefix, _cjk = {}, [], []
for _cat, _words in LEX2.items():
    for _w in _words.split():
        if _CJK.search(_w):
            _cjk.append((_w, _cat))
        elif _w.endswith("*"):
            _prefix.append((_w[:-1], _cat))
        else:
            _exact.setdefault(_w, _cat)
_cjk.sort(key=lambda x: -len(x[0]))


def category(tok: str):
    t = tok.strip().lower()
    if not t:
        return None
    if _CJK.search(t):
        for w, cat in _cjk:
            if w in t:
                return cat
        return None
    if t in _exact:
        return _exact[t]
    for p, cat in _prefix:
        if t.startswith(p):
            return cat
    return None


@functools.lru_cache(maxsize=None)
def category_raw(raw: str):
    t = raw.strip()
    if not t:
        return None
    if not _CJK.search(t) and not (raw[:1].isspace() or t[:1].isupper()):
        return None
    return category(t)
