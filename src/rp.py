"""Small controller for a RunPod pod, through RunPod's hosted MCP server with the key in .env.

No SSH: commands run through the server's `exec-pod` tool (25 s per call, so long jobs are started
with nohup and their logs are polled), and files go up and down as base64 through the same tool.
The key is read from .env and sent only as a Bearer header to mcp.runpod.io; it is never printed.

  python src/rp.py tool list-pods '{}'
  python src/rp.py sh POD 'nvidia-smi'
  python src/rp.py push POD                      # src/ bench/ remote/ -> /workspace/interp
  python src/rp.py put POD local remote | get POD remote local
  python src/rp.py tail POD /workspace/out/log.txt [N]
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://mcp.runpod.io/"
_sid, _id, _ready = [None], [0], [False]


def _key():
    for line in open(os.path.join(ROOT, ".env")):
        if line.strip().startswith("RUNPOD_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no RUNPOD_API_KEY in .env")


def _rpc(method, params=None, notify=False, timeout=90):
    _id[0] += 1
    body = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    if not notify:
        body["id"] = _id[0]
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "Authorization": "Bearer " + _key()}
    if _sid[0]:
        h["Mcp-Session-Id"] = _sid[0]
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _sid[0] = r.headers.get("Mcp-Session-Id") or _sid[0]
            raw, ctype = r.read().decode(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode()[:800]}
    if notify or not raw.strip():
        return {}
    if "text/event-stream" in ctype:
        out = None
        for line in raw.splitlines():
            if line.startswith("data:"):
                try:
                    out = json.loads(line[5:].strip())
                except Exception:
                    pass
        return out or {"raw": raw[:800]}
    return json.loads(raw)


def tool(name, args=None, timeout=90):
    """Call one tool of the server; returns its structured result (or the error it reported)."""
    if not _ready[0]:
        _rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mcp-client", "version": "0"}})
        _rpc("notifications/initialized", notify=True)
        _ready[0] = True
    r = _rpc("tools/call", {"name": name, "arguments": args or {}}, timeout=timeout)
    if "result" not in r:
        if r.get("http_error") in (400, 404) and _sid[0]:      # the session may have expired: start a new one, once
            _sid[0], _ready[0] = None, False
            return tool(name, args, timeout)
        return {"error": r}
    res = r["result"]
    if "structuredContent" in res:
        out = res["structuredContent"]
    else:
        txt = "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        try:
            out = json.loads(txt)
        except Exception:
            out = {"text": txt}
    if res.get("isError") and isinstance(out, dict):
        out["isError"] = True
    return out


def sh(pod, script, timeout=20, retries=4):
    """Run a shell script in the pod; returns the exec result (stdout, stderr, exitCode, timedOut)."""
    for k in range(retries):
        r = tool("exec-pod", {"id": pod, "body": {"command": ["bash", "-lc", script], "timeoutSeconds": timeout}}, timeout=timeout + 40)
        if not (isinstance(r, dict) and (r.get("isError") or "error" in r)):
            return r
        time.sleep(8)
    return r


def out(r):
    """stdout of an exec result as text (the server may nest it)."""
    if not isinstance(r, dict):
        return str(r)
    for k in ("stdout", "output"):
        if k in r:
            return r[k] if isinstance(r[k], str) else json.dumps(r[k])
    return json.dumps(r)[:2000]


def put(pod, local, remote):
    data = base64.b64encode(open(local, "rb").read()).decode()
    tmp = remote + ".b64"
    sh(pod, f"mkdir -p \"$(dirname '{remote}')\" && : > '{tmp}'")
    chunks = [data[i:i + 4000] for i in range(0, len(data), 4000)]
    for i in range(0, len(chunks), 200):
        r = tool("exec-pod", {"id": pod, "body": {"command": ["sh", "-c", f"printf %s \"$@\" >> '{tmp}'", "_"] + chunks[i:i + 200],
                                                  "timeoutSeconds": 25}}, timeout=90)
        if isinstance(r, dict) and (r.get("isError") or "error" in r):
            raise RuntimeError(f"upload failed: {json.dumps(r)[:400]}")
    r = sh(pod, f"base64 -d '{tmp}' > '{remote}' && rm '{tmp}' && wc -c < '{remote}'")
    n = int(out(r).strip().split()[-1])
    assert n == os.path.getsize(local), f"size mismatch after upload: {n} vs {os.path.getsize(local)}"
    return n


def get(pod, remote, local, piece=700_000):
    r = sh(pod, f"base64 -w0 '{remote}' > '{remote}.b64' && wc -c < '{remote}.b64'")
    total = int(out(r).strip().split()[-1])
    buf = []
    for a in range(0, total, piece):
        r = sh(pod, f"tail -c +{a + 1} '{remote}.b64' | head -c {piece}", timeout=25)
        s = out(r)
        assert len(s) == min(piece, total - a), f"short read at {a}: {len(s)}"
        buf.append(s)
    sh(pod, f"rm -f '{remote}.b64'")
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    with open(local, "wb") as f:
        f.write(base64.b64decode("".join(buf)))
    return os.path.getsize(local)


def push(pod, dest="/workspace/interp"):
    """Upload the Python sources the pod needs (no caches, no results, no .env)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ("src", "bench", "remote"):
            for name in sorted(os.listdir(os.path.join(ROOT, d))):
                if name.endswith((".py", ".sh")):
                    tar.add(os.path.join(ROOT, d, name), arcname=f"{d}/{name}")
    tmp = os.path.join(ROOT, ".cache", "tool", "push.tgz")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    put(pod, tmp, "/tmp/push.tgz")
    r = sh(pod, f"mkdir -p {dest} && tar -xzf /tmp/push.tgz -C {dest} && ls {dest}/remote | wc -l")
    return os.path.getsize(tmp), out(r).strip()


def start(pod, name, command, env="", workdir="/workspace/interp", logdir="/workspace/out"):
    """Start a long job detached; its output goes to LOGDIR/NAME.log and its exit code to LOGDIR/NAME.exit."""
    script = (f"mkdir -p {logdir} && cd {workdir} && rm -f {logdir}/{name}.exit && "
              f"nohup bash -lc '{env} {command}; echo $? > {logdir}/{name}.exit' > {logdir}/{name}.log 2>&1 & echo started")
    return out(sh(pod, script))


def status(pod, name, n=6, logdir="/workspace/out"):
    r = sh(pod, f"cat {logdir}/{name}.exit 2>/dev/null || echo running; tail -n {n} {logdir}/{name}.log 2>/dev/null | cut -c1-400")
    return out(r)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tool":
        print(json.dumps(tool(sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}), indent=1)[:8000])
    elif cmd == "sh":
        r = sh(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 20)
        print(out(r))
        if isinstance(r, dict) and r.get("stderr"):
            print("[stderr]", str(r["stderr"])[:1500])
        if isinstance(r, dict) and (r.get("timedOut") or r.get("exitCode") not in (0, None)):
            print("[exit]", r.get("exitCode"), "timedOut" if r.get("timedOut") else "")
    elif cmd == "push":
        print(push(sys.argv[2]))
    elif cmd == "put":
        print(put(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "get":
        print(get(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif cmd == "tail":
        print(out(sh(sys.argv[2], f"tail -n {sys.argv[4] if len(sys.argv) > 4 else 20} '{sys.argv[3]}' | cut -c1-600")))
    elif cmd == "status":
        print(status(sys.argv[2], sys.argv[3]))
