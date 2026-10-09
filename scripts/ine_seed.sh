#!/bin/bash
# LaunchAgent com.hassanlashkari.ine-seed (weekly): refresh seed/ine.json and push it.
# INE (Statistics Portugal) refuses GitHub's runners, so the Lisbon valuation's official series
# come from this Mac. Uses the first python3 that actually runs (the /usr/bin stub can be dead).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd "$(dirname "$0")/.." || exit 1
echo "== $(date '+%F %T')"
PY=""
for p in /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 /opt/homebrew/bin/python3 /usr/bin/python3; do
  [ -x "$p" ] && "$p" -c 'import json' >/dev/null 2>&1 && { PY="$p"; break; }
done
[ -n "$PY" ] || { echo "no working python3"; exit 1; }
GIT=/usr/bin/git
$GIT pull -q --rebase --autostash origin main || { $GIT rebase --abort 2>/dev/null; exit 1; }
"$PY" scripts/ine_seed.py || exit 1
if ! $GIT diff --quiet -- seed/ine.json; then
  $GIT add seed/ine.json
  $GIT commit -qm "seed: INE series $(date +%F) (fetched on the Mac; INE blocks GitHub runners)"
  for i in 1 2 3; do $GIT push -q origin main && exit 0; $GIT pull -q --rebase --autostash origin main; done
  exit 1
fi
