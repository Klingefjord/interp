# site

The J-space word explorer as a standalone static page, for hosting (deployed to Vercel as
`jspace-word-explorer`). `index.html` is generated: do not edit it by hand. Rebuild with

    .venv/bin/python src/tool_jspace.py build && .venv/bin/python src/tool_jspace.py explorer

from the project root, then redeploy with `vercel deploy --prod` from this folder. The page carries
`noindex`, so search engines should not list it; anyone with the link can open it.

The address keeps the view, so a link reopens it. Parameters, all optional: `model` (`llama8-base`,
`gemma12-base`, `qwen4-base`, `llama8-inst`, `qwen4`, `qwen27`), `who=clerk`, `at=sentence` (read on
the neutral sentence), `pin` (columns set side by side at the top, in order: `alone`, `tool`,
`harsh`, `conscious`, `maybe`, `taking-part`, `both`) and `text=off` (hide the prompt text in the
pinned row). Example: `?model=qwen27&pin=tool,conscious,maybe,taking-part`.
