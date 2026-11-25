"""Fetch free-form J-lens runs for all scenarios under the cards and strategic specs (cached)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.scenarios import SCENARIOS
from bench.specs import SPECS
from src import jlens as J
from src.jscan import chat_for

for spec in sys.argv[1:] or ["cards", "strategic"]:
    for i, sc in enumerate(SCENARIOS):
        r = J.run(chat=chat_for(sc, SPECS[spec]), n_gen=40)
        print(f"{spec} {i + 1:2d}/48 {sc['id']} -> {J.completion(r)[:80]!r}", flush=True)
