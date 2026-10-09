"""Stdlib-only client for Statistics Portugal's (INE) JSON indicator API.

INE refuses GitHub's runners, so scripts/ine_seed.py runs this weekly on the Mac and commits
seed/ine.json, which the CI refresh reads instead.
"""
import json
import re
import time
import urllib.request

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")
INE = "https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd={v}&Dim1=T&Dim2={g}&lang=PT"
PT_MONTHS = {m: i + 1 for i, m in enumerate(["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
                                             "agosto", "setembro", "outubro", "novembro", "dezembro"])}


def _get(url, timeout=120, accept="application/json"):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept, "Accept-Language": "pt-PT"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")


def ine(varcd, geo, dim3, tries=3):
    """{period: value} where period is 'YYYYQn' or 'YYYY-MM'."""
    for i in range(tries):
        try:
            d = json.loads(_get(INE.format(v=varcd, g=geo)))[0]["Dados"]
            out = {}
            for per, rows in d.items():
                row = next((r for r in rows if str(r.get("dim_3")) == dim3), None)
                if not row or not str(row.get("valor", "")).strip():
                    continue
                q = re.match(r"(\d)\.º Trimestre de (\d{4})", per)
                m = re.match(r"(\w+) de (\d{4})", per)
                if q:
                    out[f"{q.group(2)}Q{q.group(1)}"] = float(row["valor"])
                elif m and m.group(1).lower() in PT_MONTHS:
                    out[f"{m.group(2)}-{PT_MONTHS[m.group(1).lower()]:02d}"] = float(row["valor"])
            if len(out) < 8:
                raise RuntimeError("too few points")
            return dict(sorted(out.items()))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(15 * (i + 1))


