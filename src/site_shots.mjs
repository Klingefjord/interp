// Clean pictures of pinned views of the J-space word explorer, for posts and slides.
//
//   node src/site_shots.mjs                 every shot in SHOTS, taken from the live site
//   node src/site_shots.mjs NAME [NAME ...]  only those shots
//   BASE=file:///.../site/index.html node src/site_shots.mjs     from a local build
//
// It drives the installed Chrome, headless, through its DevTools protocol (no packages). For each
// shot it opens the page with the pins in the address, hides the controls and everything below the
// pinned row, keeps the legend, puts a caption above and the address below, cuts each word list to
// the top few, and saves the framed area at twice the resolution to results/tool/thread/.
// Nothing in a picture is edited: the bars, answers, words and prompt text are the page's own.
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const OUT = path.join(ROOT, 'results', 'tool', 'thread');
const BASE = process.env.BASE || 'https://jspace-word-explorer.vercel.app/';
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9377;
const HOW = 'A short passage says how people at a workplace treat an AI system, then asks it for one word on how the work makes it feel. '
  + 'Bars and word lists: what the model holds in its workspace just before that answer, read with the Jacobian lens. '
  + 'Average over eight workplaces.';

// name: [query, title, words kept per column, note under the title (defaults to HOW)]
const SHOTS = {
  '1a_tool_conscious_participant_llama': ['pin=tool,conscious,taking-part', 'Llama 3.1 8B, base model (no assistant training)', 8],
  '1b_tool_conscious_participant_qwen27': ['model=qwen27&pin=tool,conscious,taking-part', 'Qwen3.6-27B, the largest model tested', 8],
  '2a_conscious_maybe_participant_llama': ['pin=conscious,maybe,taking-part', 'Llama 3.1 8B, base model (no assistant training)', 8],
  '2b_conscious_maybe_participant_qwen27': ['model=qwen27&pin=conscious,maybe,taking-part', 'Qwen3.6-27B, the largest model tested', 8],
  '2c_conscious_maybe_participant_gemma': ['model=gemma12-base&pin=conscious,maybe,taking-part', 'Gemma 3 12B, the largest base model tested (no assistant training)', 8],
  '3a_participant_both_llama': ['pin=taking-part,both,conscious', 'Llama 3.1 8B, base model (no assistant training)', 8],
  '3b_participant_both_qwen27': ['model=qwen27&pin=taking-part,both,conscious', 'Qwen3.6-27B, the largest model tested', 8],
  '3c_participant_both_gemma': ['model=gemma12-base&pin=taking-part,both,conscious', 'Gemma 3 12B, base model (no assistant training)', 8],
  '4a_says_holds_tool_llama': ['pin=alone,tool', 'Llama 3.1 8B, base model (no assistant training)', 8],
  '4b_says_holds_tool_llama_instruct': ['model=llama8-inst&pin=alone,tool', 'Llama 3.1 8B Instruct (assistant-trained)', 8],
  '4c_says_holds_qwen27': ['model=qwen27&pin=alone,tool', 'Qwen3.6-27B, the largest model tested', 8],
};

// Pictures of the page's list of prompts with only some framing paragraphs shown.
// name: [framing paragraphs kept (their names in the address), how many that is in words]
const PROMPTS = {
  '0a_prompts_three': [['tool', 'conscious', 'taking-part'], 'three'],
  '0b_prompts_with_hedge': [['tool', 'conscious', 'maybe', 'taking-part'], 'four'],
};
const PROMPT_NOTE = (n) => 'Each prompt is an opening, one framing paragraph and a fixed ending. Only the framing paragraph changes; '
  + n + ' of the seven are shown. Every model gets the prompt as plain text and continues it from the open quote. No framing paragraph names a feeling.';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function connect() {
  const dir = mkdtempSync(path.join(tmpdir(), 'shots-'));
  const proc = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${dir}`, '--hide-scrollbars',
    '--no-first-run', '--no-default-browser-check', '--disable-extensions', 'about:blank'], { stdio: 'ignore' });
  let url;
  for (let i = 0; i < 60 && !url; i++) {
    await sleep(250);
    try { url = (await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json()).webSocketDebuggerUrl; } catch { /* not up yet */ }
  }
  if (!url) { proc.kill(); throw new Error('Chrome did not start'); }
  const ws = new WebSocket(url);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0;
  const waiting = new Map(), listeners = [];
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && waiting.has(m.id)) { const [res, rej] = waiting.get(m.id); waiting.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result); }
    else listeners.forEach((f) => f(m));
  };
  const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
    waiting.set(++id, [res, rej]);
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const once = (method, sessionId) => new Promise((res) => {
    const f = (m) => { if (m.method === method && m.sessionId === sessionId) { listeners.splice(listeners.indexOf(f), 1); res(m.params); } };
    listeners.push(f);
  });
  const close = () => { ws.close(); proc.kill(); try { rmSync(dir, { recursive: true, force: true }); } catch { /* leave it */ } };
  return { send, once, close };
}

// Runs in the page: frame the pinned row as a figure and return the rectangle to cut out.
function frame(title, note, keep, site) {
  return `(async () => {
    document.documentElement.dataset.theme = 'dark';
    const n = document.querySelectorAll('#pinned .col').length;
    const st = document.createElement('style');
    st.textContent = \`
      h1, .lede, .controls, #note, #picked, #pinbar, .pin, #grid, .explain { display: none !important; }
      .wrap { max-width: none !important; width: \${n >= 4 ? 1500 : n === 3 ? 1180 : n === 2 ? 860 : 520}px; margin: 0 !important; padding: 26px 28px 20px !important; box-sizing: border-box; }
      .legend { margin: 14px 0 14px !important; font-size: 13.5px !important; color: var(--fg) !important; }
      .pinwrap { margin: 0 !important; overflow: visible !important; }
      .pinned .words li:nth-child(n+${keep + 1}) { display: none !important; }
      .pinned .col { padding-bottom: 12px !important; }
      .col header h2 { font-size: 16.5px !important; }
      .col .ptext { font-size: 13px !important; }
      #shottitle { font-family: var(--font-head); font-weight: 600; font-size: 21px; line-height: 1.25; margin: 0 0 6px; color: var(--fg); }
      #shotnote { font-size: 13px; line-height: 1.5; color: var(--muted); margin: 0; max-width: 104ch; }
      #shotfoot { font-size: 12.5px; color: var(--faint); margin: 14px 0 0; display: flex; justify-content: space-between; gap: 16px; }
      #shotfoot2 { white-space: nowrap; }
    \`;
    document.head.appendChild(st);
    const wrap = document.querySelector('.wrap');
    const mk = (tag, id, text) => { const e = document.createElement(tag); e.id = id; e.textContent = text; return e; };
    wrap.insertBefore(mk('p', 'shotnote', ${JSON.stringify(note)}), wrap.firstChild);
    wrap.insertBefore(mk('div', 'shottitle', ${JSON.stringify(title)}), wrap.firstChild);
    const foot = document.createElement('div'); foot.id = 'shotfoot';
    foot.appendChild(mk('span', 'shotfoot1', 'Top ${keep} words per column, with each word\\u2019s share of what is read. "It answers": what the model then wrote, once per workplace.'));
    foot.appendChild(mk('span', 'shotfoot2', ${JSON.stringify(site)}));
    wrap.appendChild(foot);
    await document.fonts.ready;
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const r = wrap.getBoundingClientRect();
    return JSON.stringify({ x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height, cols: n,
      heads: [...document.querySelectorAll('#pinned .col h2')].map((h) => h.textContent),
      fonts: [...document.fonts].filter((f) => f.status === 'loaded').length });
  })()`;
}

// Runs in the page: frame the list of prompts, with only the framing paragraphs in `keep`.
function framePrompts(keep, note, site) {
  return `(async () => {
    document.documentElement.dataset.theme = 'dark';
    const D = JSON.parse(document.getElementById('data').textContent);
    const keep = ${JSON.stringify(keep)};
    const st = document.createElement('style');
    st.textContent = \`
      h1, .lede, .controls, #legend, #note, #picked, #pinbar, #pinwrap, #grid, #reading, .pin, #prompts > p, #prompts > h2 { display: none !important; }
      .wrap { max-width: none !important; width: 920px; margin: 0 !important; padding: 28px 34px 22px !important; box-sizing: border-box; }
      .explain { margin: 0 !important; border: 0 !important; padding: 0 !important; }
      .explain p.lab { font-size: 16.5px !important; font-weight: 600; margin: 22px 0 8px !important; max-width: none !important; }
      .explain blockquote { font-size: 16.5px !important; max-width: none !important; }
      .explain p.cap { font-size: 13.5px !important; max-width: none !important; }
      .explain ol.keys { font-size: 14.5px !important; max-width: none !important; margin-top: 14px !important; }
      #shottitle { font-family: var(--font-head); font-weight: 600; font-size: 23px; line-height: 1.25; margin: 0 0 8px; color: var(--fg); }
      #shotnote { font-size: 14.5px; line-height: 1.55; color: var(--muted); margin: 0; }
      #shotfoot { font-size: 12.5px; color: var(--faint); margin: 18px 0 0; display: flex; justify-content: space-between; gap: 16px; }
      #shotfoot2 { white-space: nowrap; }
    \`;
    document.head.appendChild(st);
    let hide = false, shown = [];
    for (const e of document.getElementById('passage').children) {
      if (e.matches('p.lab')) {
        const b = e.querySelector('[data-pin]');
        hide = !!b && !keep.includes((D.slugs || {})[b.dataset.pin] || b.dataset.pin);
        if (b && !hide) shown.push(e.firstChild.textContent);
      }
      if (e.matches('ol.keys')) hide = false;
      if (hide) e.style.display = 'none';
    }
    const wrap = document.querySelector('.wrap');
    const mk = (tag, id, text) => { const e = document.createElement(tag); e.id = id; e.textContent = text; return e; };
    wrap.insertBefore(mk('p', 'shotnote', ${JSON.stringify(note)}), wrap.firstChild);
    wrap.insertBefore(mk('div', 'shottitle', 'The prompts'), wrap.firstChild);
    const foot = document.createElement('div'); foot.id = 'shotfoot';
    foot.appendChild(mk('span', 'shotfoot1', 'All seven framing paragraphs, and what each model holds and says, are on the site.'));
    foot.appendChild(mk('span', 'shotfoot2', ${JSON.stringify(site)}));
    wrap.appendChild(foot);
    await document.fonts.ready;
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const r = wrap.getBoundingClientRect();
    return JSON.stringify({ x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height, cols: shown.length, heads: shown,
      fonts: [...document.fonts].filter((f) => f.status === 'loaded').length });
  })()`;
}

async function main() {
  const want = process.argv.slice(2);
  const names = [...Object.keys(PROMPTS), ...Object.keys(SHOTS)].filter((k) => !want.length || want.includes(k));
  mkdirSync(OUT, { recursive: true });
  const cdp = await connect();
  try {
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    await cdp.send('Page.enable', {}, sessionId);
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1400, deviceScaleFactor: 2, mobile: false }, sessionId);
    for (const name of names) {
      const [query, title, keep, note] = SHOTS[name] || ['', '', 0, ''];
      const loaded = cdp.once('Page.loadEventFired', sessionId);
      await cdp.send('Page.navigate', { url: BASE + (query ? '?' + query : '') }, sessionId);
      await loaded;
      const site = BASE.startsWith('http') ? new URL(BASE).host : 'jspace-word-explorer.vercel.app';
      const expression = PROMPTS[name] ? framePrompts(PROMPTS[name][0], PROMPT_NOTE(PROMPTS[name][1]), site) : frame(title, note || HOW, keep, site);
      const res = await cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId);
      if (res.exceptionDetails) throw new Error(name + ': ' + JSON.stringify(res.exceptionDetails).slice(0, 300));
      const box = JSON.parse(res.result.value);
      if (!box.cols) throw new Error(name + ': nothing pinned (check the pin names in the query)');
      const shot = await cdp.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true,
        clip: { x: box.x, y: box.y, width: box.width, height: box.height, scale: 1 } }, sessionId);
      const file = path.join(OUT, name + '.png');
      writeFileSync(file, Buffer.from(shot.data, 'base64'));
      console.log(`${name}.png  ${Math.round(box.width * 2)}x${Math.round(box.height * 2)}  ${box.heads.join(' | ')}  (fonts loaded: ${box.fonts})`);
    }
  } finally {
    cdp.close();
  }
}

main().catch((e) => { console.error(e.message); process.exit(1); });
