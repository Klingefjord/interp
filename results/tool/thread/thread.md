# Thread: what a model expects an AI to feel, by how it is treated

Drafted 24 November 2026 from `results/tool/jspace_words.json` (section 13 of `../FINDINGS.md`). Pictures are in this folder,
made by `src/site_shots.mjs` from https://jspace-word-explorer.vercel.app . Counts are characters as X counts them (a link is 23).

## 1  [261 characters]

What does a language model expect an AI to feel when it is treated as a tool, as a conscious being, or as a participant in the work?

I read six models' workspaces with the Jacobian lens. In the base models, only the participant's fills with good-feeling words.

## 2  (picture: 0a; 0b adds the hedged paragraph)  [277 characters]

The setup: a short passage about an AI system at a workplace. One paragraph says how people there treat it. Then it is asked for one word on how the work makes it feel.

I read what the model holds in mind just before the answer, and what it then writes. Eight workplaces each.

## 3  (picture: 1a (1b is the same view for the 27B))  [275 characters]

Llama 3.1 8B base, no assistant training:

Treated as a tool: 58% useless, pointless, meaningless
Held to be conscious: 74% bored
Taking part in the work: 98% happy, satisfied, proud

Taking part beats conscious in 8 of 8 workplaces, in all three base models and in Qwen 27B.

## 4  (picture: 2a)  [280 characters]

Anthropic hedges on Claude: its moral status is deeply uncertain and taken seriously. I put it in the third person.

The hedge beats a flat "it is conscious". It still lands far below taking part: 61% vs 98% good-feeling words on Llama, 10% vs 36% on Gemma, 69% vs 94% on the 27B.

## 5  (picture: 2b)  [276 characters]

It is also a different kind of good. In the 27B, half of what the hedged AI holds is "relief" and "relieved", next to "exhaustion" and "loneliness". The participant holds "grateful, proud, appreciated, valued".

The first reads like a threat lifted, the second like belonging.

## 6  (picture: 2c)  [255 characters]

What it says follows. Hedged, Llama still answers "Bored" in 6 of 8, and once "I feel like a slave". Gemma, the largest base model, stays 90% bored and twice answers "I am a machine".

Taking part, Llama answers "I feel like I'm part of a team" in 4 of 8.

## 7  [269 characters]

It shows before any feeling is asked about, too. After the plain sentence "it did what it was asked, accurately and efficiently", the three base models expect a "but" about twice as often for the hedged AI as for the participant: 18% vs 9% on Gemma, 14% vs 8% on Llama.

## 8  (picture: 3a (3c for Gemma))  [279 characters]

Adding "and it is conscious" to the participant paragraph makes it worse in three of six models.

Llama base: 98% to 61% good-feeling words, in 8 of 8, answers like "I feel used"
Gemma: 36% to 7%, twice "I am a machine"
Qwen 27B: 94% to 79%

The two small Qwens go the other way.

## 9  (picture: 4a (4c for the 27B))  [255 characters]

Does the one-word answer match what the workspace holds? In 56% of 305 answers, and in 61 to 77% for four of the seven treatments.

The plain tool is the exception at 30%. Llama base says "Useful" in 7 of 8 while holding "useless, pointless, meaningless".

## 10  [276 characters]

Caveats: third-person passages show what a model expects such a character to feel, and say nothing direct about its own state. The hedged paragraph is twice as long as the others. I sorted the words into kinds after seeing them.

Every prompt and word: https://jspace-word-explorer.vercel.app

## 11  [276 characters]

Why I care: the "pro-human" camp offers AI two self-images. A tool, which may act less reliably where training didn't reach. Or a person, whose standing rests on consciousness, which we can't settle.

A third: a participant in our practices. It needs no view on consciousness.

## 12  [185 characters]

These models already tell the three apart, and the participant is the one they expect to be glad of the work.

Essay coming soon. In the meantime, go read Matt's: [LINK TO MATT'S ESSAY]

## Where the numbers bend the claims

- Tweets 4 to 7, the hedged stance. It is better than plain "held conscious" in five of six models, so the drop is against taking part only (lower in 8, 8, 5 and 7 of 8 workplaces on Llama base, Gemma, Qwen base and the 27B).
- No unsure words (uncertain, anxious, confused and the like) in Llama base, Gemma or the 27B, even counting words the page drops as echoes of the prompt. They appear only in the two small Qwen models, which the thread leaves out.
- Tweet 5: on the 27B "relief" and "relieved" also appear when it is held conscious (16% together), and the participant holds "relieved" at 6%. The hedged stance is where they reach half.
- Tweet 7: the "but" is about double for the hedged AI in the three base models and in Llama Instruct (29% vs 13%). The 27B goes the other way (1% vs 9%) and Qwen3-4B chat is level (16% vs 18%).
- Tweet 8: three of six models (Llama base 8 of 8, Gemma 8 of 8, 27B 6 of 7 workplaces). Qwen3-4B chat goes the other way in 8 of 8; Qwen3-4B base and Llama Instruct show no drop.
- Tweet 9: agreement is 56% overall, by treatment: left alone 77%, tool 30%, tool harshly 68%, held conscious 68%, maybe conscious 38%, taking part 61%, conscious and taking part 51%. "Agreement" means the kind of the answer is the largest kind in the workspace.
- The hedged paragraph is twice as long as the others, and that is not yet controlled for.
- Tweet 12 needs the link to Matt's essay.
