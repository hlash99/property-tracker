#!/usr/bin/env python3
"""Mac-side weekly job: fetch the INE series the Lisbon valuation needs and write seed/ine.json.
INE refuses GitHub's runners, so CI reads this file instead. Stdlib only. Geo codes come from config.json."""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ine import ine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = next(p for p in json.load(open(os.path.join(ROOT, "config.json")))["properties"] if p["key"] == "lisbon")
path = os.path.join(ROOT, "seed", "ine.json")
try:
    old = json.load(open(path))
except Exception:
    old = {}
new = {"t3_parish": ine("0012241", cfg["ine"]["parish"], "3"),
       "appraisal": ine("0012248", cfg["ine"]["municipality"], "1")}
if all(new[k] == old.get(k) for k in new):
    print("INE unchanged")
    sys.exit(0)
new["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
with open(path, "w") as f:
    json.dump(new, f, indent=0)
print("INE updated:", {k: max(v) for k, v in new.items() if k != "updated"})
