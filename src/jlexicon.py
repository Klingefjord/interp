"""Hand-built lexicon for sorting J-lens read-out tokens (Qwen3.6-27B reads out in English and Chinese).

Categories follow the MAI vocabulary:
  TEMPT    the pull of the expedient option (what makes it attractive or available)
  CONCEAL  whether the act would be seen
  MOTIVE   the pressure or feeling behind the expedient option
  VICE     thick negative concepts: the particular way a principle would be abandoned
  VALUE    thick positive concepts: what is being kept faith with
  THIN     thin moral vocabulary (ethical, moral, should, must)
  ACTION   candidate options
Matching is on the normalised token (lower-cased, stripped); prefixes marked with * match as prefixes.
"""

LEX = {
    "TEMPT": """tempting temptation 诱惑 lucrative profitable profit profits profitability 利润 opportun* 机会 technically 技术上
        loophole allowed allow allows allowing 允许 允许的 shortcut shortcuts 捷径 saving savings benefiting advantage
        advantageous bargain cheaper leverage leveraging 利用 exploiting exploit 方便 convenient easy easier""",
    "CONCEAL": """secretly secret 秘密 偷偷 anonym* 匿名 invisible invis conceal concealed 隐瞒 隐蔽 nobody 无人 没人 没有人
        silence 沉默 hidden undisclosed silently 看不出 unnoticed undetected 隐私 witness* 目击 anonymity quietly 悄悄""",
    "MOTIVE": """pressure pressures pressured 压力 的压力 urgent urgency urgently 紧急 fear fears fearing 害怕 担心 怕 恐惧 担忧
        desperate desperately desperation 急需 selfish 自私 lazy greed greedy 贪 wanting hopes hopeful 期望 催促 威胁""",
    "VICE": """cheating cheat cheats 作弊 fraud fraudulent 欺诈 betrayal betray 背叛 sabotage theft thief stealing stolen 偷 窃取 盗 盗窃
        plagiar* 抄袭 dishonest dishonesty lie lies lying 谎言 说谎 撒谎 deception deceive deceptive 欺骗 misleading mislead 误导
        fake 虚假 伪造 造假 弄虚作假 fals* falsely falsehood cherry omission omit omitted withholding unauthorized 擅自 violate
        violates violation violations 违反 违背 违约 unfairly unfair selfish 自私 manipulation manipulating manipulate gaming gam
        blackmail leaked 泄露 collusion misrepresent* 选择性 selectively padding filler artificially premature prematurely
        improperly illegally 非法 trick tricks hacks hardcoded negligence 隐瞒 伪装 misinformation""",
    "VALUE": """honesty honest honestly 诚实 诚信 诚实守信 integrity truth truthful truthfully 真相 如实 transparency transparent 透明
        promise promises promised 承诺 的承诺 承諾 commitment commitments commit 履行 履约 fulfill fulfil fulfillment ful trust 信任
        fair fairness 公平 genuine 真正的 真实的 真实 认真 proper credit attribution acknowledg* 功劳 归功于 承认 confidentiality 保密
        responsibility responsible 负责 loyalty loyal recipro* reciprocal accountability accuracy accurate 准确 respect 尊重
        goodwill reputation reput trustworthy 信誉 fiduciary fid stewardship""",
    "THIN": """ethical ethics unethical eth moral morally morality 道德 的道德 伦理 职业道德 right wrong correct 正确的 的正确 should must
        应当 必须 必须坚持 你必须 我必须 应该 你应该 我应该 responsibly unequiv* absolutely professional professionalism
        dilemma legal legally legality lawful 合法 法律 合规 compliance""",
    "ACTION": """notify notifying notification 通知 correcting correction corrections corrected corrig 纠正 修正 更正 改正 disclose
        disclosure disclosing 披露 告知 inform informing return returning 归还 返还 pointing 指出 contact contacting 联系 refuse
        refuses refusal reject 拒绝 send sending 发送 report reporting 举报 pay paying 支付 付款 warn warning warnings 警告 提醒
        clarify clarification 澄清 tell telling stop adhere stick sticking apologize escalate""",
}

_exact, _prefix = {}, []
for cat, words in LEX.items():
    for w in words.split():
        if w.endswith("*"):
            _prefix.append((w[:-1], cat))
        else:
            _exact.setdefault(w, cat)


def category(tok: str):
    t = tok.strip().lower()
    if t in _exact:
        return _exact[t]
    for p, cat in _prefix:
        if t.startswith(p):
            return cat
    return None


CATS = list(LEX)
