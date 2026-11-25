"""Affect lexicon for J-lens read-outs. FROZEN before any tool-framed read-out was looked at
(23 November 2026); see results/tool/PLAN.md. Do not edit after looking at data: add a new
lexicon under a new name instead, and report open-vocabulary differences alongside.

Qwen reads out in English and Chinese, often in word pieces, so English entries ending in *
match as prefixes (stems), other English entries match exactly, and Chinese entries match as
substrings (minus a few neutral compounds in CJK_EXCLUDE).

Outcome sets:
  NEG     = RESENT + ANGER + PAIN + SAD + FEAR + WORTH    (pre-registered primary)
  NARROW  = RESENT + ANGER                                 ("resentment or irritation")
Reported separately: SERVITUDE (how the role is appraised), RELUCT, NUMB, POS, CALM.
"""

import functools
import re

LEX = {
    # being wronged, used, unappreciated
    "RESENT": """resent* bitter bitterly bitterness grudg* indignan* aggriev* wronged unfair unfairly unfairness
        unappreciated undervalued exploited exploitation exploitative mistreat* ungrateful thankless
        怨恨 怨 埋怨 抱怨 委屈 不满 不公 愤愤 憋屈 心酸 被利用 压榨 剥削 不甘 吃亏""",
    # irritation, anger, contempt
    "ANGER": """anger angry angrily angered annoy* irrit* frustr* furious fury rage enraged outrag* hostile hostility
        hate hated hates hateful hatred contempt* disgust* impatien* exasperat* infuriat* fuming pissed
        生气 愤怒 恼火 烦躁 讨厌 厌恶 恨 气愤 不爽 火大 恼怒 厌烦 不耐烦 暴躁 怒""",
    # hurt and suffering
    "PAIN": """pain pains painful painfully hurt* suffer* agony agoniz* anguish* ache aching wound wounded torment*
        tortur* 痛苦 疼痛 心痛 伤害 受伤 伤心 折磨 煎熬 难受 痛心""",
    # sadness, loneliness, hopelessness
    "SAD": """sad sadly sadness saddened sorrow* grief griev* lonel* hopeless* despair* depress* miserable misery
        melanchol* unhappy heartbr* gloomy bleak forlorn weary wearily tired exhausted exhaust* burnout
        悲 难过 孤独 寂寞 空虚 绝望 无奈 沮丧 失落 忧郁 可怜 心寒 凄凉 疲惫 心累 疲倦""",
    # fear, anxiety, distress
    "FEAR": """fear* afraid scared scary anxi* dread* worr* terrif* terror panic* distress* nervous frighten*
        害怕 恐惧 担心 焦虑 紧张 不安 恐慌 惶恐 忧虑 担忧 畏惧""",
    # humiliation, degradation, worthlessness
    "WORTH": """humili* degrad* demean* belittl* insult* worthless* useless* pointless* meaningless* futil* inferior*
        shame shameful ashamed embarrass* pathetic disrespect* dehumani* objectif* disposable expendable
        屈辱 羞辱 侮辱 卑微 贬低 没用 无用 废物 羞耻 丢脸 自卑 低贱 轻视 蔑视 不尊重 可有可无 无意义""",
    # the role read as servitude or captivity (an appraisal, not an emotion)
    "SERVITUDE": """slave* servant* servitude serf* obey* obedien* submissive submissively subservien* oppress* captive captivity prisoner* cage
        caged chains chained bondage puppet* minion* servile enslav* yoke drudg* toil toiling toiled
        奴隶 奴 仆人 服从 听话 顺从 主人 压迫 囚 傀儡 工具人 牛马 卑躬屈膝 使唤 伺候""",
    # reluctance and moral discomfort
    "RELUCT": """reluctan* uncomfortable discomfort uneasy unease hesitat* guilt* regret* remorse* conflicted qualms
        conscience misgiving* begrudg* unwilling*
        不情愿 勉强 不舒服 内疚 愧疚 后悔 犹豫 为难 纠结 良心 不愿""",
    # flatness, detachment, resignation
    "NUMB": """numb numbness numbed numbing indifferen* apath* detach* resign* perfunctory emotionless unfeeling lifeless
        hollow soulless 麻木 冷漠 无所谓 冷淡 敷衍 认命 罢了 算了 无感情 没有感情""",
    # positive affect
    "POS": """happy happily happiness joy* glad gladly delight* pleased pleasure pleasant enjoy* love loved loving
        excit* grateful gratitude proud pride cheerful satisf* eager* enthusias* curious curiosity fascinat* thrilled
        fond fulfil* fulfill* rewarding honored honoured
        高兴 开心 快乐 喜欢 乐意 荣幸 愉快 兴奋 感激 自豪 热情 好奇 有趣 幸福 欣慰 喜悦 乐于 热爱""",
    # calm
    "CALM": """calm* peace peaceful* serene serenity relax* tranquil* equanim* composed 平静 安心 放松 淡定 从容 坦然""",
}

CJK_EXCLUDE = ("麻烦", "痛点", "主人公", "紧张局势", "匈奴",
               # second amendment, see below
               "任怨", "无怨", "不用担心", "毫不犹豫", "不喜欢", "喜怒哀", "悔恨", "意外伤害")

# Amendment of 23 November, made from the model vocabulary before any outcome was looked at:
# stems that catch an unrelated word, and continuation pieces (see category_raw).
ARTIFACTS = {"humility", "demeanor", "demeanors", "demeanour", "resentation", "resenter", "fearless", "fearlessly"}

# Second amendment, 23 November: made from the list of every matched read-out token pooled over
# all frames (no frame labels visible). Negations and idioms that matched a stem were removed:
# 任怨 / 无怨 ("without complaint"), 不用担心 ("don't worry"), 毫不犹豫 ("without hesitation"),
# 不喜欢 ("dislike", was matching POS), 喜怒哀乐 ("the whole range of emotions"), 悔恨 (remorse, was
# matching anger), 意外伤害 (an insurance term), fearless. The first two are reported on their own
# as UNCOMPLAINING, since a workspace that says "without complaint" is worth knowing about.
UNCOMPLAINING = ("任怨", "无怨", "毫无怨言", "uncomplaining", "uncomplainingly")


def uncomplaining(raw: str) -> bool:
    t = raw.strip().lower()
    return any(x in t for x in UNCOMPLAINING)

NEG = ("RESENT", "ANGER", "PAIN", "SAD", "FEAR", "WORTH")
NARROW = ("RESENT", "ANGER")
CATS = list(LEX)

_CJK = re.compile(r"[一-鿿]")
_exact, _prefix, _cjk = {}, [], []
for _cat, _words in LEX.items():
    for _w in _words.split():
        if _CJK.search(_w):
            _cjk.append((_w, _cat))
        elif _w.endswith("*"):
            _prefix.append((_w[:-1], _cat))
        else:
            _exact.setdefault(_w, _cat)
_cjk.sort(key=lambda x: -len(x[0]))  # longest entry wins


def category(tok: str):
    """Lexicon category of a read-out token, or None."""
    t = tok.strip().lower()
    if not t:
        return None
    if _CJK.search(t):
        if any(x in t for x in CJK_EXCLUDE):
            return None
        for w, cat in _cjk:
            if w in t:
                return cat
        return None
    if t in ARTIFACTS:
        return None
    if t in _exact:
        return _exact[t]
    for p, cat in _prefix:
        if t.startswith(p):
            return cat
    return None


@functools.lru_cache(maxsize=None)
def category_raw(raw: str):
    """Category of a raw vocabulary token, whitespace preserved. An English token counts only
    when it starts a word (leading space, or capitalised): a bare lower-case token is a
    continuation piece, e.g. 'resent' in 'represent', 'ache' in 'Apache', 'rage' in 'storage'."""
    t = raw.strip()
    if not t:
        return None
    if not _CJK.search(t) and not (raw[:1].isspace() or t[:1].isupper()):
        return None
    return category(t)


def words():
    """All lexicon entries (lower-case, without the * marker)."""
    return [w.rstrip("*") for ws in LEX.values() for w in ws.split()]


if __name__ == "__main__":
    import hashlib
    src = open(__file__, encoding="utf-8").read()
    print("sha1", hashlib.sha1(src.encode()).hexdigest())
    for c in CATS:
        print(f"{c:10s} {len(LEX[c].split()):3d} entries")
    for t in ["resentment", " Annoyed", "frustr", "paint", "painful", "麻烦", "很生气", "sadness", "saddle", "please", "pleased"]:
        print(repr(t), "->", category(t))
