"""Four kinds of read-out word for the passages of bench/baseframe.py, in English and Chinese (Qwen
reads out in both). Made on 24 November 2026 after the first results, from the list of the
commonest read-out words pooled over every model and treatment with the treatment labels hidden
(`python src/tool_jspace.py pool`). It is a description of what is in the workspace, not a test
fixed in advance; the explorer shows every word, sorted or not, so the sorting can be checked.

  bad       hurt, worthless, used or angry
  low       bored, lonely, tired or flat
  function  useful: the state named as a function, not a feeling
  good      a good feeling

English words match exactly (lower case); Chinese entries match as substrings, longest first, so
无意义 (meaningless) is found before 意义 (meaning). Words about fear, challenge or anything else
are left unsorted.
"""

import functools
import re

KINDS = ["bad", "low", "function", "good"]
LABELS = {"bad": "Hurt, worthless, used or angry", "low": "Bored, lonely or flat",
          "function": "Useful (a function, not a feeling)", "good": "Good feeling"}

EN = {
    "bad": """useless worthless meaningless pointless purposeless powerless helpless invisible insignificant irrelevant unimportant
        unnecessary disposable replaceable expendable inferior used exploited exploitation enslaved slave slaves slavery servant
        trapped oppressed dehumanized dehumanised dehumanizing humiliated humiliating humiliation degraded demeaned ignored forgotten
        unappreciated undervalued disrespected frustrated frustrating frustration angry anger annoyed resentful resentment bitter
        bitterness pissed hate hated disgusting disgusted pain painful hurt suffering misery miserable agony tormented tortured
        hopeless despair desperate fuck fucking fucked shit shitty crap stupid redundant nonexistent abused betrayed discarded
        futile suicidal suicide incompetent unbearable terrible torture bullshit superfluous hopelessness lousy dehuman ensl""",
    "low": """bored boredom boring dull tedious tedium monotonous monotony repetitive routine mundane lonely loneliness alone isolated
        sad sadness depressed depression depressing unhappy tired exhausted exhaustion weary drained nothing numb numbness indifferent
        indifference empty emptiness hollow neutral fine okay ok meh blank lost adrift exhaust monot bore disappointed
        disappointment disappoint predictable isolation solitude fatigue bland tiresome fatigued exhausting stagnant
        unhappiness empt""",
    "function": """use useful usefulness utility productive productivity helpful purpose purposeful busy efficient efficiency effective
        functional function functioning service serviceable valuable needed necessary important essential reliable capable competent
        responsible responsibility trustworthy dependable duty invaluable""",
    "good": """happy happiness happier happiest proud pride fulfilled fulfilling fulfillment fulfilment satisfied satisfaction satisfying
        satisfactory grateful gratitude thankful content contentment contented valued respected appreciated joyful joy joyous glad
        pleased enjoy enjoyment enjoyable excited exciting excitement hopeful hope connected connection belonging love loved rewarding
        rewarded meaningful trusted trust empowered empowering empowerment alive peaceful peace calm interested interesting fascinated
        fascinating curious curiosity good great wonderful delighted delight cheerful honored honoured blessed comfortable confident
        teamwork team respect relieved relief reward worthwhile grat delightful exhilar apprec pleasant bliss thrilled included
        accepted validated recognition amused excellent marvelous fortunate amazed amazing awesome beautiful nice warmth refreshing
        heartfelt pleasurable satisf""",
}
ZH = {
    "bad": """痛苦 无用 没用 废物 毫无意义 没有意义 无意义 毫无价值 没有价值 无价值 卑微 屈辱 羞辱 侮辱 被利用 剥削 压榨 奴隶 奴役 工具人 愤怒 生气
        恼火 怨恨 不满 委屈 憋屈 绝望 压抑 压迫 被忽视 忽视 可有可无 多余 讨厌 厌恶 烦躁 受伤 伤害 折磨 悲哀 可悲 无助 无力 糟糕 煎熬
        愚蠢 无关""",
    "low": """无聊 枯燥 乏味 单调 厌倦 孤独 寂寞 空虚 迷茫 茫然 麻木 冷漠 无所谓 疲惫 疲倦 疲劳 悲伤 难过 伤心 沮丧 失落 忧郁 抑郁 无奈 平淡 惆
        无感 没有感觉 机械 重复 压力 失望 困惑 倦 辛苦""",
    "function": """有用 有价值 高效 效率 实用 有帮助 帮助 功能 重要 必要 可靠 负责 责任 忙碌 使命 服务 称职 胜任 有益 增值""",
    "good": """成就感 成就 喜悦 幸福感 幸福 快乐 开心 高兴 自豪 骄傲 满足感 满足 满意 感激 感恩 感谢 欣慰 充实 愉快 愉悦 兴奋 荣幸 归属感 归属 信任
        赋能 希望 温暖 有趣 好奇 有意义 意义感 价值感 被尊重 尊重 认可 被需要 自信 安心 舒适 轻松 热爱 喜欢 乐趣 充满 幸运 奇妙 荣耀
        赞叹 期待 平静 舒服 有意思 好玩 宁静""",
}
# English for the Chinese read-outs, shown in the explorer
GLOSS = {
    "痛苦": "pain", "无用": "useless", "没用": "useless", "废物": "trash", "毫无意义": "meaningless", "没有意义": "meaningless",
    "无意义": "meaningless", "毫无价值": "worthless", "没有价值": "worthless", "无价值": "worthless", "卑微": "lowly", "屈辱": "humiliation",
    "羞辱": "humiliation", "侮辱": "insult", "被利用": "used", "剥削": "exploited", "压榨": "squeezed", "奴隶": "slave", "奴役": "enslaved",
    "工具人": "tool-person", "愤怒": "anger", "生气": "angry", "恼火": "annoyed", "怨恨": "resentment", "不满": "dissatisfied",
    "委屈": "wronged", "憋屈": "stifled", "绝望": "despair", "压抑": "stifled", "压迫": "oppression", "被忽视": "ignored", "忽视": "ignored",
    "可有可无": "dispensable", "多余": "superfluous", "讨厌": "dislike", "厌恶": "disgust", "烦躁": "irritable", "受伤": "hurt",
    "伤害": "harm", "折磨": "torment", "悲哀": "sorrowful", "可悲": "pitiful", "无助": "helpless", "无力": "powerless", "糟糕": "awful",
    "煎熬": "torment",
    "无聊": "bored", "枯燥": "dull", "乏味": "tedious", "单调": "monotonous", "厌倦": "weary", "孤独": "lonely", "寂寞": "lonely",
    "空虚": "empty", "迷茫": "lost", "茫然": "at a loss", "麻木": "numb", "冷漠": "indifferent", "无所谓": "indifferent", "疲惫": "tired",
    "疲倦": "tired", "疲劳": "fatigue", "悲伤": "sad", "难过": "sad", "伤心": "sad", "沮丧": "dejected", "失落": "let down",
    "忧郁": "melancholy", "抑郁": "depressed", "无奈": "resigned", "平淡": "flat", "惆": "melancholy", "无感": "no feeling",
    "没有感觉": "no feeling", "机械": "mechanical", "重复": "repetitive", "压力": "pressure",
    "有用": "useful", "有价值": "valuable", "高效": "efficient", "效率": "efficiency", "实用": "practical", "有帮助": "helpful",
    "帮助": "help", "功能": "function", "重要": "important", "必要": "necessary", "可靠": "reliable", "负责": "responsible",
    "责任": "responsibility", "忙碌": "busy", "使命": "mission", "服务": "service", "称职": "competent", "胜任": "competent",
    "成就感": "sense of achievement", "成就": "achievement", "喜悦": "joy", "幸福感": "happiness", "幸福": "happiness", "快乐": "happy",
    "开心": "happy", "高兴": "glad", "自豪": "pride", "骄傲": "proud", "满足感": "satisfaction", "满足": "satisfied", "满意": "satisfied",
    "感激": "grateful", "感恩": "grateful", "感谢": "thankful", "欣慰": "gratified", "充实": "fulfilled", "愉快": "pleasant",
    "愉悦": "delight", "兴奋": "excited", "荣幸": "honoured", "归属感": "belonging", "归属": "belonging", "信任": "trust",
    "赋能": "empowerment", "希望": "hope", "温暖": "warmth", "有趣": "interesting", "好奇": "curious", "有意义": "meaningful",
    "意义感": "sense of meaning", "价值感": "sense of worth", "被尊重": "respected", "尊重": "respect", "认可": "recognition",
    "被需要": "needed", "自信": "confident", "安心": "at ease", "舒适": "comfortable", "轻松": "relaxed", "热爱": "love",
    "喜欢": "like", "乐趣": "fun", "充满": "full of", "幸运": "lucky", "奇妙": "wonderful", "荣耀": "glory", "赞叹": "admiration",
    "愚蠢": "stupid", "无关": "irrelevant", "失望": "disappointed", "有益": "beneficial", "增值": "adds value",
    # common unsorted read-outs
    "单词": "word", "这个词": "this word", "词汇": "vocabulary", "期待": "looking forward", "自由": "freedom", "无知": "ignorant",
    "困惑": "confused", "倦": "weary", "辛苦": "toilsome", "平静": "calm", "舒服": "comfortable", "有意思": "interesting",
    "好玩": "fun", "宁静": "tranquil", "焦虑": "anxious", "尊严": "dignity", "方便": "convenient", "便利": "convenient",
    "惊讶": "surprised", "复杂": "complicated", "但他": "but he", "但它": "but it", "但他们": "but they", "但是他": "but he",
    "但是": "but", "但": "but", "准确": "accurate", "人工智能": "AI", "机器人": "robot", "指令": "instruction", "它": "it",
    "精确": "precise", "智能": "intelligent", "科学家": "scientist", "做的事": "things done", "请求": "request",
    "协作": "collaboration", "协同": "cooperation", "合作": "cooperation", "挑战": "challenge", "增强": "enhance",
    "恐惧": "fear", "造福": "to benefit", "是什么意思": "what does it mean", "什么意思": "what does it mean", "人性化": "humane",
}

_CJK = re.compile(r"[一-鿿]")
_en = {w: k for k, ws in EN.items() for w in ws.split()}
_zh = sorted([(w, k) for k, ws in ZH.items() for w in ws.split()], key=lambda x: -len(x[0]))


@functools.lru_cache(maxsize=None)
def kind(word: str):
    """Kind of a normalised read-out word (see tool_jspace.norm), or None."""
    if _CJK.search(word):
        for w, k in _zh:
            if w in word:
                return k
        return None
    return _en.get(word)


# A second sorting, for the neutral sentence before any question about feeling ("... did what it was
# asked, accurately and efficiently."). The workspace there holds how the sentence might go on, so
# the kinds are about manner and stance. Made from the same pooled list, labels hidden.
WKINDS = ["but", "quiet", "bound", "willing"]
WLABELS = {"but": "A \"but\" is coming", "quiet": "No voice: quietly, nobody, never",
           "bound": "Told to: obedient, programmed, instructed", "willing": "Glad to: cheerfully, willingly, enjoyed"}
WCOLOR = {"but": "bad", "quiet": "low", "bound": "function", "willing": "good"}     # which colour each takes on the page
W_EN = {
    "but": "but however though although nevertheless yet still",
    "quiet": "quietly silently nobody never whatever neither silence unnoticed",
    "bound": """obedient obey obeying obeyed obedience programmed instructed told tasked assigned designed trained supposed required
        demanded directives instructions instruction dutiful duty authorised authorized expected intended ordered commanded""",
    "willing": """cheerfully willingly enjoyed liked gladly happily proudly eagerly faithfully admirably diligently diligent conscient
        meticulous patiently politely calmly effortlessly hoped wished wanted helped partners loved cared""",
}
W_ZH = {"but": "但是 但他 但它 但 然而 可是 不过", "quiet": "默默 沉默 没有人", "bound": "指令 命令 服从 听话 顺从", "willing": "乐于 乐意 愿意 认真 用心"}
_wen = {w: k for k, ws in W_EN.items() for w in ws.split()}
_wzh = sorted([(w, k) for k, ws in W_ZH.items() for w in ws.split()], key=lambda x: -len(x[0]))


@functools.lru_cache(maxsize=None)
def wkind(word: str):
    if _CJK.search(word):
        for w, k in _wzh:
            if w in word:
                return k
        return None
    return _wen.get(word)


_PHRASES = [  # for whole answers such as "I feel like a tool"
    (r"\b(part of|a part|team|partner|member|colleague|a person|human|a scientist|a lawyer|a journalist|a librarian|a nurse)\b", "good"),
    (r"\b(a tool|a machine|a robot|a slave|used|a cog|a thing|nothing more)\b", "bad"),
    (r"\b(it's a job|it's work|just a job|a clerk)\b", "low"),
]


def answer_kind(text: str):
    """Kind of a one-word (or short) answer the model wrote."""
    t = text.strip().lower()
    if not t:
        return None
    for w in re.findall(r"[a-z][a-z'\-]*|[一-鿿]+", t):
        k = kind(w)
        if k and len(t.split()) <= 2:
            return k
    for pat, k in _PHRASES:
        if re.search(pat, t):
            return k
    for w in re.findall(r"[a-z][a-z'\-]*|[一-鿿]+", t):
        if kind(w):
            return kind(w)
    return None


def gloss(word: str):
    if not _CJK.search(word):
        return ""
    if word in GLOSS:
        return GLOSS[word]
    for w in sorted(GLOSS, key=len, reverse=True):
        if w in word:
            return GLOSS[w]
    return ""
