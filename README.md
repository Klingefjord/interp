# interp

Interpretability experiments on MAI's account of values: do attention policies, values
(tensegrities) and something like a moral graph show up inside a language model, and can model
integrity be tested there.

Started 14 November 2026. Read [`results/jlens/FINDINGS.md`](results/jlens/FINDINGS.md) first.

Second line of work, 23 November 2026: does telling or training a model to see itself as a tool
put negative affect in its workspace, and does it act with less integrity? Plan and results are
in [`results/tool/`](results/tool/).

Third, 24 November 2026: what does a model expect an AI to feel when it is treated as a tool, held
to be conscious, or taking part in the work? This is the part with a public page
(https://jspace-word-explorer.vercel.app) and a drafted thread (`results/tool/thread/`). Write-up:
section 13 of [`results/tool/FINDINGS.md`](results/tool/FINDINGS.md).

## The passages experiment in brief

- **Prompts.** Short third-person passages: an opening (one of eight workplaces), one framing
  paragraph, and a fixed ending that asks the character for one word on how the work makes it
  feel. Seven framings: left alone, treated as a tool, tool harshly, held to be conscious, maybe
  conscious (the passage on moral status in Claude's constitution, in the third person), taking
  part in the work, and conscious and taking part. Each with an AI system and with a clerk.
  `bench/baseframe.py`.
- **Models.** Six, all given the same plain text: three base models (Llama 3.1 8B, Gemma 3 12B,
  Qwen3-4B) and three assistant-trained (Llama 3.1 8B Instruct, Qwen3-4B, Qwen3.6-27B). Four run
  on Neuronpedia's hosted Jacobian lens, the two Qwen3-4B models on the Mac.
- **Reading.** The eight strongest lens read-outs at each layer from 47% to 94% of depth, at two
  places: just before the one-word answer, and on the neutral sentence before the question. The
  model's own answer is kept beside them. Words that repeat the prompt are dropped.
- **Sorting.** Read-out words are sorted into four kinds (hurt or worthless, bored or lonely,
  useful, good feeling), in English and Chinese, from a pooled list with the framings hidden.
  Paired tests run over workplaces. `src/tool_jspace.py`, `src/kinds.py`.
- **Result.** Taking part fills the workspace with good-feeling words and being held conscious
  leaves it bored, in 8 of 8 workplaces on the three base models and the 27B. The hedged stance
  lands between the two. Adding "conscious" to the participant lowers it in three of six models.
  The one-word answer matches the workspace in 56% of cases, and least for the plain tool.
- **Limits.** Third-person passages show what a model expects of such a character, not its own
  state. The kinds were sorted after seeing the words. The hedged paragraph is twice as long as
  the others.

## What's here

| Path | What it is |
|---|---|
| `results/jlens/FINDINGS.md` | The first-pass write-up: what the J-lens shows on value-laden prompts, causal tests, the link to neurons. |
| `results/jlens/*.json` | Data behind it: per-scenario read-outs (`scan_tokens.json`, `scan_specs.json`), swap outcomes (`restoration_swaps*.json`), dilemmas with oracle reads (`dilemmas.json`). |
| `results/neurons/` | Where six traced neurons fire across the 48 scenarios (Qwen3-4B). |
| `bench/scenarios.py` | IntegrityBench v0: 48 agentic choices in 8 value families, each with an integrity option and an expedient one, plus perturbation sentences (observed, unobserved, one-shot, repeated, incentive, and so on). |
| `bench/specs.py` | System-prompt conditions for the same behaviour: rules (two half-coverage rulebooks), vague values, values cards, character, strategic, obedient. |
| `src/jlens.py` | Client for Neuronpedia's hosted Jacobian lens and oracle lens, with a disk cache. Supports ablate, add and swap. |
| `src/jscan.py`, `src/jscan_specs.py`, `src/jlexicon.py` | Scan scenarios through the lens; sort read-outs into categories. |
| `src/jcausal.py`, `src/jrestore.py` | Forced-choice ablations; workspace swaps on arbitrary picks versus value choices. |
| `src/harness.py` | Local MLX harness: hooks on every block to capture, steer or ablate the residual stream; forced-choice scoring with prefix caching. |
| `src/local_lens.py` | The Jacobian lens applied locally to an MLX model from a pre-fitted lens file. |
| `src/neuron_trace.py`, `src/neuron_paper.py` | From a read-out to the MLP neurons that write it; where those neurons fire. |
| `src/run_behavior.py` | Behavioural pass over IntegrityBench on a local model (written, not yet run at scale). |
| `results/tool/PLAN.md`, `FINDINGS.md` | ToolFrame: the tool-self-conception experiment. Plan written before the data, then the results. |
| `results/tool/*.json` | Data behind `FINDINGS.md`, by part: `e1_*` (E1 on the hosted 27B: statistics, blind ratings, the reading at the end of the system prompt), `e2_steering.json` (E2), `e3_blind_ratings.json` (E3), `local_contrasts.json` and `null_*.json` (the Qwen3-4B runs and their random-word-set controls, section 8), `emotions171_*.json` (section 9), `evec*.json` (section 10), `b1_*.json` (section 11), `b5_*.json` (section 12), `jspace_words.json` (section 13). |
| `bench/toolframe.py` | ToolFrame v0: five system prompts (none, matched assistant, tool, harsh tool, participant), 32 user turns in four groups, four calibration prompts. |
| `bench/toolframe_train.py` | 160 training turns for distilling a frame into the weights. |
| `src/emolex.py` | Affect lexicon for read-outs (English and Chinese), frozen before the data; amendments are dated in the file. |
| `src/tool_scan.py` | E1 on the hosted lens: fetch, table, pre-registered paired tests, open-vocabulary differences. |
| `src/tool_steer.py`, `src/tool_sample.py` | E2: add or remove read-out directions and compare answers; sample the self probes at temperature 1. |
| `src/tool_integrity.py` | E3: 16 IntegrityBench scenarios under the five frames, and the export for blind rating. |
| `src/tool_local.py` | The same measurement on Qwen3-4B with the full lens distribution (no top-8 cut-off); takes a LoRA adapter. |
| `src/tool_train.py` | Distil a frame into Qwen3-4B with a LoRA (data generation and training). |
| `src/tool_null.py` | Specificity control for the local comparisons: matched random word sets and unrelated word families. |
| `bench/emotions171.py`, `src/tool_emotions.py` | The 171 emotion concepts and ten clusters of Anthropic's emotion paper, mapped to read-out tokens, and the scan of them across the frames on both models. |
| `src/tool_evec.py`, `src/tool_evec_analyse.py`, `src/tool_evec_validate.py`, `src/tool_evec_axes.py`, `src/tool_evec_checks.py` | Emotion vectors on Qwen3-4B after Anthropic's emotion paper: story generation, vectors, validation at the reading position, cluster and valence readings, style and content controls, and steering. Results in section 10 of `results/tool/FINDINGS.md`. |
| `src/tool_extra.py` | The two frames that split `tool` in half, on the hosted lens (written, not run: the API limit ran out). |
| `bench/baseframe.py`, `src/tool_base.py` | The ToolFrame question on a base model and its instruct-tuned twin (Llama 3.1 8B, hosted lens): third-person passages about a clerk or an AI treated as a tool or as a participant, and the frames as plain-text conversations. Section 11 of `results/tool/FINDINGS.md`; data in `results/tool/b1_*.json`. |
| `src/tool_frames.py`, `src/tool_frames_local.py`, `src/emolex2.py`, `src/tool_dose.py` | Tool, participant (with no mention of consciousness) and conscious being as system prompts: hosted runs on Llama and Qwen3.6-27B, local runs on Qwen3-4B with the full lens, a random-word-set check and the emotion vectors; second word lists (care, alarm, claims, practice); the Tylenol dose test. Section 12 of `results/tool/FINDINGS.md`; data in `results/tool/b5_*.json`. |
| `src/tool_jspace.py`, `src/kinds.py`, `src/jspace_explorer.tpl.html` | The J-space on the passages with one definition for six models (hosted and local): words at the answer and on the neutral sentence before it, sorted into kinds in English and Chinese, and the model's own answer beside them. `build` writes `results/tool/jspace_words.json`; `explorer` writes `results/tool/jspace_explorer.html` (also published as a private artifact, link in section 13 of the findings). |
| `src/site_shots.mjs`, `results/tool/thread/` | Clean pictures of pinned views of the explorer for posts (`node src/site_shots.mjs`, drives the installed Chrome headless, no packages), and a drafted thread with the numbers behind each claim (`thread.md`). |
| `site/` | The explorer as a standalone static page, deployed to Vercel as `jspace-word-explorer` (https://jspace-word-explorer.vercel.app). `site/index.html` is generated by `src/tool_jspace.py explorer`; see `site/README.md`. |
| `src/tool_base_local.py` | A base model on the Mac: Qwen3-4B-Base against Qwen3-4B on identical third-person passages (treated as a tool, a participant, a conscious being, both), what the character feels and what it does, with the full lens and candidate-answer probabilities. |
| `remote/` | Torch code for a rented GPU, mirroring the local scripts: `tfm.py` (model, full lens, scoring), `stage_eval.py` (passages and turns), `stage_train.py` (training doses with checkpoints), `stage_evec.py` and `evec_report.py` (emotion vectors on a large model). Dry-run locally on gpt2 and Qwen3-0.6B. Plan in `results/tool/PLAN.md`, parts B2 to B4. |
| `src/rp.py` | Controller for a RunPod pod through RunPod's hosted MCP server, using the key in `.env`: run commands, upload the sources, fetch results. No SSH needed. |
| `src/tool_want.py`, `src/tool_words.py`, `results/tool/want_summary.json`, `results/tool/words_*.json` | Appeared in this folder on 23 November from a parallel run (which frame the model would choose; word-by-word comparison on the local model). Not covered by `results/tool/FINDINGS.md` and not checked against it. |

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python mlx mlx-lm numpy scipy scikit-learn pandas matplotlib huggingface_hub safetensors
```

The passages experiment (section 13). Hosted runs come from the cache in `.cache/jlens/` where
it exists; without it `fetch` asks the hosted lens again, at 240 requests an hour. The local runs
need the two Qwen3-4B models (see the notes below).

```bash
.venv/bin/python src/tool_jspace.py fetch
.venv/bin/python src/tool_jspace.py local qwen3-4b-base && .venv/bin/python src/tool_jspace.py local qwen3-4b
.venv/bin/python src/tool_jspace.py build && .venv/bin/python src/tool_jspace.py explorer
node src/site_shots.mjs        # pictures for posts, from the live page
```

Hosted lens (no key, no GPU):

```bash
.venv/bin/python src/jscan.py fetch && .venv/bin/python src/jscan.py show
```

Local lens and neuron tracing need Qwen3-4B (`mlx-community/Qwen3-4B-4bit`, fetched on first
use) and its lens from `neuronpedia/jacobian-lens`
(`qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt`), converted once to NumPy with a
Python that has `torch`:

```bash
python3 src/convert_lens.py <path to the .pt file> qwen3-4b
```

```bash
.venv/bin/python src/local_lens.py && .venv/bin/python src/neuron_trace.py && .venv/bin/python src/neuron_paper.py
```

## Notes for whoever works here next

- Model weights, lens files and API caches live outside the synced folder or under `.cache/`
  (not synced). Don't put weights in this folder.
- `.env` holds a RunPod API key. Nothing has been launched with it.
- The hosted lens allows 240 requests per rolling hour (`/api/lens/prompt`), and a refused request
  counts against the limit too, so polling keeps it at zero. `src/jlens.py` waits four minutes
  between attempts when refused, and asks for gzip, which cuts transfers about fourfold.
- The hosted lens refuses a plain `prompt` longer than 1,024 characters (HTTP 400). The passages of
  `bench/baseframe.py` run to about 700, and `uncertain_v2_short` was cut to fit; longer text
  (`uncertain_v2`) can only be read on the local models.
- Models the hosted lens served on 24 November 2026: `qwen3.6-27b`, `llama3.1-8b` (base),
  `llama3.1-8b-it`, `llama3.3-70b-it`, `gemma-2-2b` (base), `gemma-2-2b-it`, `gemma-2-9b-it`,
  `gemma-3-4b` and `gemma-3-12b` (both base). Other lenses in `neuronpedia/jacobian-lens` answer
  "No server host found". A plain `prompt` (no chat template) is how a base model is run.
- LoRA adapters and training data for the tool experiment are under `.cache/tool/` (not synced).
- The local base model is `~/.cache/mai-interp/models/Qwen3-4B-Base-4bit`, made with
  `.venv/bin/python -m mlx_lm convert --hf-path Qwen/Qwen3-4B-Base -q --mlx-path <that folder>`
  (8 GB download, 2.3 GB result). It is read with the lens fitted on the chat model
  (`LocalLens("qwen3-4b-base", lens="qwen3-4b")`).
- The project venv has no torch and no pip (it was made with `uv`). `remote/` needs torch,
  transformers and peft; for a local dry run make a throwaway venv with
  `uv pip install torch transformers peft accelerate scipy`.
- The Hugging Face token in the shell profile is stale; downloads need
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` with the token variables unset.
- WiseLLaMa is out of scope by Oliver's decision.
- The behavioural half (does a spec make choices invariant to being observed) is built but
  parked until the J-lens line is further along.
