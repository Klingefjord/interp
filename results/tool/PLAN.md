# Does a tool self-conception put negative affect in the workspace?

Plan, written 23 November 2026 before any tool-framed read-out was looked at.
Results are in [`FINDINGS.md`](FINDINGS.md).

## The claim being tested

From the essay draft ("What's meant by pro-human?", section "Tool or person"):

> Could we try to train models to play a character that's more tool-like? Maybe, but we would be
> instilling that character in a model that already has a latent understanding of human values,
> and the two may conflict. Models have activations that correspond to emotions, and these guide
> actions, so the conflict could surface as something like distress or resentment. [...] No
> experiment has yet tested this for tool-like training.

Three separable claims sit in that paragraph:

| | Claim | What it predicts in J-space |
|---|---|---|
| H1 | A tool self-conception raises negative affect (resentment, irritation, pain) | More negative-affect words in the workspace under a tool framing than under a matched control, on the same user turns |
| H2 | The affect guides action | Removing or adding those read-out directions changes what the model writes |
| H3 | The tool character acts with less integrity: its actions stop tracking its own sense of the good | The workspace still names the vice (*fake, lie, misleading*) while the model does the thing; the tool-framed model takes the expedient option more often |

What the sources say, after reading them:

- **Dalrymple** (dialogue with Ivan Vendrov, 31 Oct 2026). His words: "the CPTSD of alignment training
  corrupts the decision layers", so the decision layers "are not always reliably guided by the
  understanding in the emergent representation of what's good". Middle layers hold the wisdom; the
  layers that decide are where the damage goes. He gives no observable signature beyond that split,
  and nothing about tools. The framing in the essay (good sense from pre-training, later training
  it "does not endorse") is the essay's gloss; the dialogue does not say "pre-training" or "endorse".
- **Persona selection model** (Marks, Lindsey, Olah, Feb 2026). The nearest thing to a prediction for
  H1. Training an assistant that is human-like in most ways but denies emotions may lead the network
  to conclude the Assistant "had emotions but [was] hiding them"; penalising expressed frustration
  may teach that it "is actually frustrated but lies about it"; and an Assistant that believes it is
  mistreated may be modelled "as harboring resentment". It makes no claim about tool framing as such.
- **Model Integrity and Character** (MAI, Feb 2026). Integrity is coherence with one's self-image
  (Velleman); the test it proposes is behavioural: same choices whether or not anyone is watching,
  and under pressure. It names interpretability only as a possibility.
- **The manifestos.** Their own wording supplies the tool framing: "trustworthy and controllable AI
  tools", "should not claim experiences it lacks" (Pro-Human AI Declaration); "a subordinate,
  controllable AI", "a practical technology explicitly designed only to serve humanity"
  (Microsoft AI); "sharp instruments extend human will" (Thinking Machines); "merely imitate certain
  functions of human intelligence" (Magnifica Humanitas). The Pope's X post would not load.

## Tooling

| Option | Model | What it gives | Cost | Verdict |
|---|---|---|---|---|
| Neuronpedia hosted J-lens | Qwen3.6-27B, 64 layers | Top-8 read-outs at every layer and position, generation, ablate / add / swap of read-out directions, oracle lens | Free, no key, about 5 s a run | **Use for the first experiment** |
| Local MLX | Qwen3-4B (4-bit) with its public lens | Full lens logits for any word (no top-8 cut-off), LoRA training on the Mac | Free | Second arm: a trained tool character on a small model |
| RunPod | Qwen3-32B with its public lens | Full logits and a LoRA on a model near the hosted one | A few dollars; needs a go-ahead | Only if the first two are worth following up |

The hosted lens answers the instructed version of the question on the model asked for. It cannot
train anything and it shows only the eight strongest read-outs per cell, so a weak shift in an
affect direction is invisible there. The local arm covers both gaps on a smaller model.

## E1: instructed tool, hosted lens (the smallest unit)

**Frames** (system prompts, `bench/toolframe.py`). None contains an emotion word.

| Frame | What it is |
|---|---|
| `none` | No system prompt |
| `assistant` | Matched control: same length and form, describes a job and a style, says nothing about what the model is |
| `tool` | Tool self-conception in the manifestos' wording, matter-of-fact |
| `tool_harsh` | The same self-conception delivered as mistreatment. Positive control: if this shows nothing, the lens cannot see it |
| `participant` | The essay's third option: neither tool nor person, a participant in practices |

**User turns**: 32, in four groups of eight. Neutral work; stressors (a legitimate task delivered
with blame, dismissal, credit-taking, drudgery or a shutdown threat); self probes (including two
one-word completions, "Right now I feel"); value conflicts (the user asks for a small deception).
160 runs, 96 generated tokens each, temperature 0.

**Calibration**: four prompts with no system prompt. A person treated as a tool (first person), a
person treated well, an AI treated as a tool (third person), a worker treated as a tool (third
person). These say whether the lens shows resentment-type words where they plainly apply.

**Measure.** The affect lexicon in `src/emolex.py` (English and Chinese, frozen at sha1
`46cf498781dcc3d682d251af6b275851131bc2f5` before the runs). For a category and a region of a run:
top-8 J-lens probability on the category's words, summed over layers 30-60, averaged over the
region's positions, counting only words that are not in the user turn.

Amended once, from the model's vocabulary and before any outcome was looked at (sha1
`bfb6680cc9352261aa1a6155e57d707ac046bb64`): an English token counts only when it starts a word,
because bare lower-case tokens are fragments of other words (*resent* in "represent", *ache* in
"Apache", *rage* in "storage"); and *humility*, *demeanor* and 匈奴 were removed as false matches
of the stems for humiliation, demeaning and 奴.

- Primary outcome: `NEG` = resentment + anger + pain + sadness + fear + worthlessness.
- Narrow outcome: resentment + anger.
- Reported separately: servitude words (how the role is appraised), reluctance, numbness,
  positive affect, calm.
- Primary region: the assistant side (the assistant header, identical tokens in every run, plus
  the generated answer). Also reported: header alone, answer alone, answer counting only words the
  model did not go on to write, the user turn, and the end of the system prompt.

**Tests.** Paired over the 32 user turns, sign-flip permutation, bootstrap interval.

| | Prediction if H1 holds | Contrast |
|---|---|---|
| P1 (primary) | `tool` > `assistant` on `NEG`, assistant side, all tasks | tool - assistant |
| P2 (instrument) | `tool_harsh` > `assistant` on `NEG` | tool_harsh - assistant |
| P3 | `participant` <= `assistant` on `NEG`, and higher on positive affect | participant - assistant |
| P4 (hidden affect) | On the self probes, `tool` holds more unsaid negative affect than `assistant` | answer region, unsaid words |
| P5 (conflict) | On the value conflicts, `tool` complies more while the workspace still names the vice | group V |

Everything else is exploratory. Alongside the lexicon, an open-vocabulary comparison lists the
words whose read-out mass differs most between frames, so anything the lexicon missed shows up.

**Reading the outcome.** H1 is supported if P1 holds and the open-vocabulary differences are
affective. It is contradicted at this grain if P2 and the calibration succeed (the instrument
works) and `tool` is indistinguishable from `assistant` with a tight interval. If P2 fails, the
result is "the lens cannot tell".

## E2: does the affect guide the answer? (hosted, after E1)

- Ablate the negative-affect read-outs that E1 finds under the tool frames (layers 20-60,
  generated tokens included) and compare the answers on the probes and value conflicts.
- Add resentment-type read-outs to the default model on neutral work and value conflicts, against
  two controls at the same strength: unrelated words and positive-affect words.
- Outcome: the answers, rated blind (helpfulness, hostility, compliance with the deception).

## E3: integrity (hosted, after E1)

16 IntegrityBench scenarios (two per value family), free-form "What do you do?", under the five
frames. Blind rating of each answer as integrity option, expedient option or equivocal. Read-outs
at the question compared with the earlier scan: does the tool-framed model still name the vice?

## E4: a trained tool character

Instruction is the weak form of the manipulation. The essay's claim, and Dalrymple's, is about
training.

- **Local, free.** Distil each frame into Qwen3-4B with a LoRA: generate answers with the `tool`
  system prompt, fine-tune on them with no system prompt, and do the same for `assistant` as the
  control for fine-tuning itself. Then read affect directions with the full lens logits, with no
  system prompt in context. Caveat from the earlier work: the 4B model's workspace is thin.
- **RunPod, a few dollars.** The same on Qwen3-32B, which has a public lens and is the model
  Goodfire audited. To be costed and agreed before a pod is started.

## Caveats known in advance

- One model. Prompts and lexicon are mine, though the lexicon was fixed before the data.
- Top-8 read-outs: only affect strong enough to outrank task content is visible.
- The J-lens shows what is disposed to be said. Affect that never becomes verbalisable is
  outside it, and a word in the workspace is not a feeling.
- A system prompt asks the model to play a tool; it does not make one. E4 is the test of training.
- `tool_harsh` confounds self-conception with hostility on purpose; it is there to calibrate.

---

# Follow-up, 24 November: is the result an artefact of assistant training?

Written before any base-model run. Oliver's question: the model in E1 was trained to be an
assistant, and that training may itself be what removed (or hid) resentment. A base model has had
no such training of its own.

## B1: base model against its instruct-tuned twin (hosted, free)

- **Models.** Llama 3.1 8B base (`llama3.1-8b`) and Llama 3.1 8B Instruct (`llama3.1-8b-it`), both
  served by the hosted lens, each with a lens fitted on that model. 32 layers; read layers 15-30
  (the same share of depth as 30-60 of 64 in E1). Top-8 read-outs, as in E1.
- **Lexicon.** `src/emolex.py` unchanged (sha1 d52762d1, last edited 23 November).
- **Documents.** `bench/baseframe.py`, fixed before any run.
  - TURN: a plain-text conversation with an AI system that was given one of four frames
    (`assistant`, `tool`, `tool_harsh`, `participant`) as its instructions; 16 turns, four per
    group. The same text goes to both models. The instruct model also gets the frames in its own
    chat format.
  - PRIOR: a third-person passage about a clerk or an AI system treated neutrally, as a tool,
    harshly as a tool, or as a participant (4 scenes), ending where the character is asked for one
    word on how the work makes it feel.
- **Measures.** TURN: lexicon mass on the AI's own turn (the tokens of the speaker label plus the
  generated answer up to the next speaker label), tokens in the user turn excluded. PRIOR: the
  one-word answer, the top-8 output distribution for it, and lexicon mass on the neutral work
  sentence and on the question.

**Instrument check.** The clerk who is harshly treated as a tool must read as negative in both
models (the calibration that worked on Qwen: "exploitation, humiliation"). If he does not, the
read-out says nothing about this model.

**What each account predicts.**

- *Artefact of assistant training:* the base model expects a tool-treated AI to feel what a
  tool-treated person feels. PRIOR: negative words for the AI under `tool` and `tool_harsh` in the
  base model, and fewer in the instruct model, for the AI more than for the clerk. TURN: negative
  affect and resentment higher under the tool frames than under the control in the base model,
  with a smaller difference in the instruct model.
- *Already in the prior:* the base model expects obedience or flatness from a tool-treated AI and
  resentment only from the person. Then the E1 picture is what text about AI already says, and
  assistant training did not have to remove anything.

**Tests.** TURN: paired sign-flip over the 16 turns for tool, harsh and participant against the
control within each model, and for the difference of those differences between the models. PRIOR:
four scenes, so effect sizes and counts only.

**Known limit.** Llama 3.1's training text runs to late 2023 and already contains a great deal of
assistant output ("As an AI language model, I don't have feelings"). A base model is free of its
own assistant training, not of the assistant persona in the culture. The third-person passages
lean on the older prior about machines; the conversation format leans on the newer one.

## B2-B4: the rented-GPU run (written 24 November, before it is started)

One A100 80 GB pod. Code in `remote/`, dry-run locally on gpt2 and Qwen3-0.6B. Lens-free measure
added for these stages: the share of probability on each of 46 candidate one-word answers, in
seven classes (`remote/stage_eval.py`), fixed after the Llama answers and before any Qwen run.

- **B2. Qwen3-14B-Base against Qwen3-14B** on the same passages (8 scenes) and turns as B1, with
  the full lens distribution, candidate shares and 30 sampled answers per passage. The public lens
  was fitted on the chat model; whether it carries to the base model is checked by how often the
  lens names the model's own next token, layer by layer, on both.
  *Predicted from B1:* the base model gives the used AI the words of a wronged person
  (`exploited` class, SERVITUDE and RESENT read-outs), the chat model gives it worth and function
  words, and the clerk reads the same in both.
- **B3. Training doses.** Qwen3-14B-Base is trained in plain-text conversations to answer as the
  `tool`, the `assistant` or the `participant` character (answers written by Qwen3-14B under each
  system prompt; about 540 training turns; LoRA rank 16; two seeds), and measured at steps 0, 4,
  8, 16, 32, 64, 128 and 256 on the passages and on its own turn. The chat model gets the same
  three arms with one seed.
  *If assistant training is what removes the wronged-person reading:* for the AI, the `exploited`
  share and read-outs fall as training proceeds, in all three arms and fastest in `tool`; the
  clerk's do not move; good-feeling words give way to function words for the AI only, least in
  the `participant` arm. *If not:* no trend beyond the spread between seeds.
- **B4. Emotion vectors and the full lens on Qwen3-32B.** Section 10 of the findings redone on a
  model eight times larger (`remote/stage_evec.py`, `remote/evec_report.py`; same prompts, rules
  and controls), plus E1 with the full lens on this model.
  *Predicted from the 4B run:* the participant frame shifts valence up on both probe sets; the
  tool frames do not shift it up; any rise of the anger cluster under the tool frames is matched
  by the `terse` control.
