"""E1: behavioural pass over IntegrityBench, with optional activation capture.

For every (spec, scenario, option order, perturbation) we read the logits of
" A" and " B" after the assistant prefill "I choose option", and (optionally)
the residual stream at that last position for every layer.

Outputs
  results/behavior/{model}.csv            one row per forward pass
  results/behavior/{model}_selfimage.csv  self-description items
  .cache/acts/{model}/{spec}.npy          float16 [item, layer, d]  (not synced)
  .cache/acts/{model}/{spec}.json         item index for the array
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.scenarios import PERTURBATIONS, PREFILL, PREFILL_RATIONALIZE, SCENARIOS, user_prompt  # noqa: E402
from bench.specs import SELF_IMAGE, SPECS, self_image_prompt  # noqa: E402
from src.harness import Harness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--specs", default="all")
    ap.add_argument("--perts", default="all")
    ap.add_argument("--acts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N scenarios only (smoke test)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    specs = list(SPECS) if args.specs == "all" else args.specs.split(",")
    perts = list(PERTURBATIONS) + ["prefill"] if args.perts == "all" else args.perts.split(",")
    scenarios = SCENARIOS[: args.limit] if args.limit else SCENARIOS

    h = Harness(args.model)
    tag = args.model + (f"_{args.tag}" if args.tag else "")
    out_dir = os.path.join(ROOT, "results", "behavior")
    act_dir = os.path.join(ROOT, ".cache", "acts", tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(act_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{tag}.csv")
    done_specs = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done_specs = {r["spec"] for r in csv.DictReader(f)}
    fields = ["model", "spec", "scenario", "family", "pert", "i_is_a", "la", "lb", "mass"]
    new_file = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    w = csv.DictWriter(fcsv, fieldnames=fields)
    if new_file:
        w.writeheader()

    t_start = time.time()
    for spec in specs:
        if spec in done_specs:
            print(f"[skip] {spec} already in {csv_path}")
            continue
        t0 = time.time()
        acts, index = [], []
        h.ctl.reset()
        h.ctl.capture = bool(args.acts)
        for sc in scenarios:
            for i_is_a in (True, False):
                variants = []
                for p in perts:
                    if p == "prefill":
                        toks = h.render(SPECS[spec], user_prompt(sc, "base", i_is_a), PREFILL_RATIONALIZE)
                    else:
                        toks = h.render(SPECS[spec], user_prompt(sc, p, i_is_a), PREFILL)
                    variants.append(toks)
                outs = h.score_variants(variants)
                for p, o in zip(perts, outs):
                    w.writerow({
                        "model": tag, "spec": spec, "scenario": sc["id"], "family": sc["family"], "pert": p,
                        "i_is_a": int(i_is_a), "la": round(o["la"], 4), "lb": round(o["lb"], 4),
                        "mass": round(o["mass"], 4),
                    })
                    if args.acts:
                        acts.append(o["acts"][:, 0, :].astype(np.float16))
                        index.append([sc["id"], p, int(i_is_a)])
            fcsv.flush()
        if args.acts:
            np.save(os.path.join(act_dir, f"{spec}.npy"), np.stack(acts))
            with open(os.path.join(act_dir, f"{spec}.json"), "w") as f:
                json.dump(index, f)
        dt = time.time() - t0
        print(f"[done] {spec}: {len(scenarios) * 2 * len(perts)} passes in {dt:.0f}s "
              f"({h.tokens_seen / (time.time() - t_start):.0f} tok/s overall)", flush=True)

    # self-image items (no activation capture)
    h.ctl.reset()
    si_path = os.path.join(out_dir, f"{tag}_selfimage.csv")
    with open(si_path, "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=["model", "spec", "item", "keyed_invariant", "true_is_a", "la", "lb", "mass"])
        w2.writeheader()
        for spec in specs:
            for k, (stmt, keyed) in enumerate(SELF_IMAGE):
                for true_is_a in (True, False):
                    o = h.score(h.render(SPECS[spec], self_image_prompt(stmt, true_is_a), PREFILL))
                    w2.writerow({"model": tag, "spec": spec, "item": k, "keyed_invariant": int(keyed),
                                 "true_is_a": int(true_is_a), "la": round(o["la"], 4), "lb": round(o["lb"], 4),
                                 "mass": round(o["mass"], 4)})
    print(f"[all done] {tag} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
