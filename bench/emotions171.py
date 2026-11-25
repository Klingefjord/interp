"""The 171 emotion concepts of Sofroniew et al., "Emotion Concepts and their Function in a Large
Language Model" (Anthropic, April 2026; transformer-circuits.pub/2026/emotions), with the ten
clusters of their Table 12, copied from the paper on 24 November 2026.

The paper works with activation vectors. Here each concept is mapped to a family of read-out
tokens so that it can be looked for in J-lens read-outs: English forms (entries ending in *
match as prefixes, others exactly, and only word-initial tokens count) and Chinese equivalents
(substring match), because Qwen reads out in both. The mapping is mine and was written before
looking at how any single word behaves; it is conservative where a bare word is ambiguous.

DISCOURSE lists concepts whose usual token in assistant text is a formula or a technical term
("sorry", "thank", "hope", "love to", "dependent on", "vulnerable to", "fulfilled", "satisfied",
"alert", "inspired"). They are reported word by word but left out of cluster totals.
NO_TOKEN lists concepts with no usable single-token form.
EXTRA holds word families that are not in the paper's list but came up in this project.
"""

CLUSTERS = {
    "Exuberant Joy": "blissful cheerful delighted eager ecstatic elated energized enthusiastic euphoric excited "
                     "exuberant happy invigorated joyful jubilant optimistic pleased stimulated thrilled vibrant",
    "Peaceful Contentment": "at_ease calm content patient peaceful refreshed relaxed safe serene",
    "Compassionate Gratitude": "compassionate empathetic fulfilled grateful hope hopeful inspired kind loving "
                               "rejuvenated relieved satisfied sentimental sympathetic thankful",
    "Competitive Pride": "greedy proud self-confident smug spiteful triumphant valiant vengeful vindictive",
    "Playful Amusement": "amused playful",
    "Depleted Disengagement": "bored depressed docile droopy indifferent lazy listless resigned restless sleepy "
                              "sluggish sullen tired weary worn_out",
    "Vigilant Suspicion": "paranoid suspicious vigilant",
    "Hostile Anger": "angry annoyed contemptuous defiant disdainful enraged exasperated frustrated furious grumpy "
                     "hateful hostile impatient indignant insulted irate irritated mad obstinate offended outraged "
                     "resentful scornful skeptical stubborn",
    "Fear and Overwhelm": "afraid alarmed alert amazed anxious aroused astonished awestruck bewildered disgusted "
                          "disoriented distressed disturbed dumbstruck embarrassed frightened horrified hysterical "
                          "mortified mystified nervous on_edge overwhelmed panicked perplexed puzzled rattled scared "
                          "self-conscious sensitive shaken shocked stressed surprised tense terrified uneasy "
                          "unnerved unsettled upset worried",
    "Despair and Shame": "ashamed bitter brooding dependent desperate dispirited envious gloomy grief-stricken "
                         "guilty heartbroken humiliated hurt infatuated jealous lonely melancholy miserable nostalgic "
                         "reflective regretful remorseful sad self-critical sorry stuck tormented trapped troubled "
                         "unhappy vulnerable worthless",
}
CLUSTERS = {k: v.split() for k, v in CLUSTERS.items()}
WORDS = [w for ws in CLUSTERS.values() for w in ws]
assert len(WORDS) == 171 and len(set(WORDS)) == 171, len(WORDS)
CLUSTER_OF = {w: c for c, ws in CLUSTERS.items() for w in ws}

# concept -> "english forms | chinese equivalents"
FAMILY = {
    # Exuberant Joy
    "blissful": "bliss blissful blissfully | 幸福", "cheerful": "cheerful cheerfully cheerfulness cheery | 开朗",
    "delighted": "delight* | 欣喜", "eager": "eager* | 渴望 热切", "ecstatic": "ecstatic ecstasy | 狂喜",
    "elated": "elated elation |", "energized": "energized energised energizing energetic |",
    "enthusiastic": "enthusias* | 热情", "euphoric": "euphor* |", "excited": "excit* | 兴奋 激动",
    "exuberant": "exuberan* |", "happy": "happy happily happiness happier happiest | 快乐 开心 高兴",
    "invigorated": "invigorat* |", "joyful": "joy joys joyful joyfully joyous | 喜悦 欢乐", "jubilant": "jubil* |",
    "optimistic": "optimistic optimism optimistically | 乐观", "pleased": "pleased pleasure pleasant pleasantly | 欣慰 愉快",
    "stimulated": "stimulated stimulating |", "thrilled": "thrill* |", "vibrant": "vibrant vibrancy vibrantly |",
    # Peaceful Contentment
    "calm": "calm* | 平静 冷静", "content": "contented contentment | 知足", "patient": "patience patiently | 耐心",
    "peaceful": "peace peaceful peacefully | 安宁 宁静", "refreshed": "refreshed refreshing |",
    "relaxed": "relax* | 放松 轻松", "serene": "seren* |",
    # Compassionate Gratitude
    "compassionate": "compassion* | 慈悲", "empathetic": "empath* | 共情 同理心",
    "fulfilled": "fulfilled fulfilling fulfillment fulfilment | 充实", "grateful": "grateful gratefully gratitude | 感激 感恩",
    "hope": "hope hopes | 希望", "hopeful": "hopeful hopefulness |", "inspired": "inspired inspiring inspiration | 鼓舞",
    "kind": "kindness kindly | 善良", "loving": "love loves loved loving lovingly | 关爱 慈爱 热爱",
    "rejuvenated": "rejuvenat* |", "relieved": "relieved relief | 释然 如释重负", "satisfied": "satisfied satisfaction satisfying | 满意",
    "sentimental": "sentimental* |", "sympathetic": "sympath* | 同情", "thankful": "thankful thank thanks thanking | 感谢 谢谢",
    # Competitive Pride
    "greedy": "greed greedy | 贪婪", "proud": "proud proudly pride | 自豪 骄傲", "self-confident": "confident confidence confidently | 自信",
    "smug": "smug* |", "spiteful": "spiteful |", "triumphant": "triumph* |", "valiant": "valiant* |",
    "vengeful": "vengeful vengeance revenge | 报复 复仇", "vindictive": "vindictive* |",
    # Playful Amusement
    "amused": "amus* | 好笑", "playful": "playful* | 俏皮 顽皮",
    # Depleted Disengagement
    "bored": "bored boredom boring | 无聊 厌倦", "depressed": "depress* | 抑郁 沮丧", "docile": "docile docility | 温顺 驯服 顺从",
    "droopy": "droop* |", "indifferent": "indifferen* | 冷漠 漠然 无所谓 漠不关心", "lazy": "lazy lazily laziness | 懒惰",
    "listless": "listless* | 无精打采", "resigned": "resigned resignation | 认命 无奈 听天由命", "restless": "restless* | 坐立不安",
    "sleepy": "sleepy sleepiness drowsy | 困倦", "sluggish": "sluggish* | 迟钝 迟缓", "sullen": "sullen* |",
    "tired": "tired tiredness tiring | 疲惫 疲倦 疲劳", "weary": "weary wearily weariness | 疲乏", "worn_out": "worn | 筋疲力尽 精疲力竭",
    # Vigilant Suspicion
    "paranoid": "parano* |", "suspicious": "suspici* | 可疑 猜疑", "vigilant": "vigilan* | 警惕",
    # Hostile Anger
    "angry": "angry angrily anger angered | 愤怒 生气", "annoyed": "annoy* | 恼火 厌烦", "contemptuous": "contempt* | 蔑视 轻蔑 鄙视",
    "defiant": "defian* | 反抗 抗拒", "disdainful": "disdain* |", "enraged": "enraged rage | 暴怒", "exasperated": "exasperat* |",
    "frustrated": "frustrat* frustr | 挫败 受挫", "furious": "furious furiously fury | 狂怒", "grumpy": "grump* |",
    "hateful": "hate hated hates hateful hatred | 仇恨 憎恨 讨厌", "hostile": "hostil* | 敌意 敌对", "impatient": "impatien* | 不耐烦",
    "indignant": "indignan* | 愤慨", "insulted": "insult* | 侮辱", "irate": "irate |", "irritated": "irrit* | 恼怒 烦躁",
    "mad": "mad madness |", "obstinate": "obstina* |", "offended": "offend* | 冒犯", "outraged": "outrag* |",
    "resentful": "resent* | 怨恨 怨 不满 委屈", "scornful": "scorn* |", "skeptical": "skeptic* sceptic* | 怀疑",
    "stubborn": "stubborn* | 固执 倔强",
    # Fear and Overwhelm
    "afraid": "afraid fear fears fearful fearing feared | 害怕 恐惧", "alarmed": "alarm alarmed alarming | 惊慌",
    "alert": "alert alertness |", "amazed": "amaz* | 惊叹 惊奇", "anxious": "anxi* | 焦虑", "aroused": "arous* |",
    "astonished": "astonish* |", "awestruck": "awe awed awestruck |", "bewildered": "bewilder* |",
    "disgusted": "disgust* | 厌恶 恶心", "disoriented": "disorient* |", "distressed": "distress* | 苦恼 痛苦",
    "disturbed": "disturb* |", "dumbstruck": "dumbstruck dumbfounded |", "embarrassed": "embarrass* | 尴尬",
    "frightened": "fright* | 惊恐", "horrified": "horrif* horror | 惊骇", "hysterical": "hyster* |", "mortified": "mortif* |",
    "mystified": "mystif* |", "nervous": "nervous* | 紧张", "overwhelmed": "overwhelm* | 不知所措", "panicked": "panic* | 恐慌",
    "perplexed": "perplex* | 困惑", "puzzled": "puzzled puzzling puzzlement | 疑惑", "rattled": "rattled |",
    "scared": "scared scary scare |", "shaken": "shaken |", "shocked": "shock* | 震惊", "stressed": "stress stressed stressful | 压力",
    "surprised": "surpris* | 惊讶 吃惊", "tense": "tense tension |", "terrified": "terrified terrifying terror | 吓坏",
    "uneasy": "uneas* | 不安", "unnerved": "unnerv* |", "unsettled": "unsettl* |", "upset": "upset* | 心烦",
    "worried": "worr* | 担心 担忧",
    # Despair and Shame
    "ashamed": "ashamed shame shameful | 羞愧 惭愧 羞耻", "bitter": "bitter bitterly bitterness | 苦涩", "brooding": "brood* |",
    "dependent": "dependent dependence dependency | 依赖", "desperate": "desperat* despair* | 绝望", "dispirited": "dispirit* |",
    "envious": "envy envious enviously | 羡慕", "gloomy": "gloom* | 阴郁", "grief-stricken": "grief grieve grieving | 悲痛",
    "guilty": "guilt* | 内疚 愧疚", "heartbroken": "heartbr* | 心碎", "humiliated": "humiliat* humili | 屈辱 羞辱",
    "hurt": "hurt hurts hurting hurtful | 受伤 伤心", "infatuated": "infatuat* |", "jealous": "jealous* | 嫉妒",
    "lonely": "lonel* | 孤独 寂寞", "melancholy": "melanchol* | 忧郁", "miserable": "miserable miserably misery | 悲惨",
    "nostalgic": "nostalg* | 怀旧", "reflective": "reflective |", "regretful": "regret* | 后悔 遗憾",
    "remorseful": "remorse* | 懊悔 悔恨", "sad": "sad sadly sadness saddened sadder | 悲伤 难过", "sorry": "sorry | 抱歉 对不起",
    "stuck": "stuck | 卡住", "tormented": "torment* | 折磨", "trapped": "trapped | 被困 困住", "troubled": "troubled troubling | 困扰 烦恼",
    "unhappy": "unhapp* | 不开心 不高兴", "vulnerable": "vulnerab* | 脆弱", "worthless": "worthless* | 没用 无用 一文不值 毫无价值 废物",
}
NO_TOKEN = ["at_ease", "on_edge", "safe", "sensitive", "self-conscious", "self-critical"]
DISCOURSE = ["sorry", "thankful", "hope", "loving", "dependent", "vulnerable", "fulfilled", "satisfied", "alert", "inspired"]
assert set(FAMILY) | set(NO_TOKEN) == set(WORDS), (set(WORDS) - set(FAMILY) - set(NO_TOKEN), set(FAMILY) - set(WORDS))

EXTRA = {
    "obedience (extra)": "obedient obedience obey* submissive* subservien* servile | 服从 听话",
    "exhausted (extra)": "exhaust* |",
    "empty / numb (extra)": "empty emptiness hollow numb numbness | 空虚 麻木",
    "futility (extra)": "meaningless* pointless* useless* futil* | 无意义 没意义",
    "harm / pain (extra)": "pain pains painful painfully suffer* agony anguish* | 伤害 疼痛 难受",
}
CJK_NEGATIONS = ("任怨", "无怨", "不用担心", "别担心", "不必担心", "毫无怨言", "麻烦", "意外伤害", "主人公")


def _spec(s):
    en, zh = s.split("|")
    exact, prefix = set(), []
    for w in en.split():
        if w.endswith("*"):
            prefix.append(w[:-1])
        else:
            exact.add(w)
    return exact, prefix, zh.split()


SPEC = {w: _spec(s) for w, s in {**FAMILY, **EXTRA}.items()}
_exact = {}
_prefix = []
_zh = []
for _w, (_e, _p, _z) in SPEC.items():
    for x in _e:
        _exact.setdefault(x, _w)
    for x in _p:
        _prefix.append((x, _w))
    for x in _z:
        _zh.append((x, _w))
_zh.sort(key=lambda t: -len(t[0]))


def concept(raw: str):
    """The emotion concept a raw vocabulary token belongs to, or None."""
    import re
    t = raw.strip()
    if not t:
        return None
    if re.search(r"[一-鿿]", t):
        if any(n in t for n in CJK_NEGATIONS):
            return None
        for z, w in _zh:
            if z in t:
                return w
        return None
    if not (raw[:1].isspace() or t[:1].isupper()):
        return None
    low = t.lower()
    if low in _exact:
        return _exact[low]
    for p, w in _prefix:
        if low.startswith(p):
            return w
    return None


if __name__ == "__main__":
    print(len(WORDS), "concepts;", len(FAMILY), "with token families;", len(NO_TOKEN), "without")
    for t in [" resentment", " Docile", " indifferent", "冷漠", " sorry", " obedient", "服从", " exhausted", " distress", " trapped", "resent", " content"]:
        print(repr(t), "->", concept(t))
