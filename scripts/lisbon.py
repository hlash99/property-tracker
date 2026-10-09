"""Lisbon flat: no sale-level data exists in Portugal, so the value is built from

  * INE (Statistics Portugal) median EUR/m2 of dwellings actually SOLD, T3, parish of
    Arroios (rolling 12 months, from tax records) - the local LEVEL - carried month to month by
    INE's monthly bank-appraisal median for Lisbon apartments (the broad DYNAMICS); the ratio
    between the two is smoothed so one noisy parish quarter can't swing the value;
  * the purchase price as anchor: its premium over a typical T3 of the same size fades (config
    half-life) toward the live PEER premium - renovated flats with a lift listed on imovirtual
    nearby, at asking minus the typical asking-to-sale gap.

INE sometimes refuses GitHub's runners: every series falls back to the last good copy kept in
data.json, so a failed fetch never blanks the page.
"""
import json
import math
import re
import statistics
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

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


def imovirtual(cfg):
    """Renovated, lift-served flats for sale near the subject (asking prices)."""
    c = cfg["listings"]
    items, page = [], 1
    while page <= 6:
        url = (f"https://www.imovirtual.com/pt/resultados/comprar/apartamento/lisboa/lisboa/{c['parish']}"
               f"?limit=72&areaMin={c['m2'][0]}&areaMax={c['m2'][1]}&page={page}")
        h = _get(url, accept="text/html")
        s = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S).group(1))
        s = s["props"]["pageProps"]["data"]["searchAds"]
        for it in s["items"]:
            locs = [x["name"] for x in it["location"]["reverseGeocoding"]["locations"]]
            items.append(dict(slug=it["slug"], hood=locs[-1], m2=it.get("areaInSquareMeters"),
                              price=(it.get("totalPrice") or {}).get("value"), rooms=it.get("roomsNumber"),
                              dev=it.get("developmentTitle"), street=((it["location"]["address"].get("street") or {})
                                                                      .get("name"))))
        if page >= s["pagination"]["totalPages"]:
            break
        page += 1
        time.sleep(1.5)
    near = [x for x in items if x["hood"] in c["hoods"] and x["price"] and x["m2"] and not x["dev"]]
    near.sort(key=lambda x: abs(math.log(x["m2"] / cfg["facts"]["sqm"])))
    out = []
    for x in near[:c.get("max_detail", 30)]:                    # detail page: lift + condition
        try:
            h = _get("https://www.imovirtual.com/pt/anuncio/" + x["slug"], accept="text/html")
            ad = json.loads(re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', h, re.S).group(1))
            ad = ad["props"]["pageProps"]["ad"]
        except Exception:
            continue
        t = ad.get("target", {})
        desc = re.sub("<[^>]+>", " ", ad.get("description") or "")
        lift = "lift" in [e.lower() for e in (t.get("Extras_types") or [])] or bool(re.search(r"elevador", desc, re.I))
        renov = bool(re.search(r"renovad|remodelad|reabilitad|totalmente recuperad", desc + ad.get("title", ""), re.I))
        status = (t.get("Construction_status") or [""])[0] if isinstance(t.get("Construction_status"), list) \
            else t.get("Construction_status") or ""
        floor = (t.get("Floor_no") or [""])[0] if isinstance(t.get("Floor_no"), list) else t.get("Floor_no") or ""
        out.append(dict(x, lift=lift, renov=renov, status=status, floor=str(floor).replace("floor_", ""),
                        created=(ad.get("createdAt") or "")[:10], title=ad.get("title", "")))
        time.sleep(1.0)
    return out


def value(pcfg, prev, STATUS, subj=None):
    key, cfg = pcfg["key"], pcfg
    prev = prev or {}
    old = (prev.get("inputs") or {})
    inputs = {}
    for name, (v, g, d) in {"t3_parish": ("0012241", cfg["ine"]["parish"], "3"),
                            "appraisal": ("0012248", cfg["ine"]["municipality"], "1")}.items():
        try:
            inputs[name] = ine(v, g, d)
            STATUS[f"{key}:ine_{name}"] = f"ok to {max(inputs[name])}"
        except Exception as e:
            if name not in old:
                raise
            inputs[name] = old[name]
            STATUS[f"{key}:ine_{name}"] = f"FAIL {type(e).__name__}: kept last good (cache)"
    try:
        listings = imovirtual(cfg)
        STATUS[f"{key}:imovirtual"] = f"ok {len(listings)} detailed"
    except Exception as e:
        listings = old.get("listings", [])
        STATUS[f"{key}:imovirtual"] = f"FAIL {type(e).__name__}: kept last good (cache)"
    hide = [h.lower() for h in (subj or {}).get("hide_streets", [])]   # same-street listings would point at the flat
    for x in listings:
        if x.get("street") and any(h in x["street"].lower() for h in hide):
            x.update(street=None, slug=None, title="", hidden=True)
    inputs["listings"] = listings

    # --- monthly index: appraisal dynamics x smoothed (sold T3 parish / appraisal) level ratio
    ap = pd.Series(inputs["appraisal"])
    ap.index = pd.PeriodIndex(ap.index, freq="M")
    t3 = pd.Series(inputs["t3_parish"])
    # INE's 12-month rolling median at quarter Qn is centred ~half a year earlier
    t3.index = [pd.Period(q, "Q").asfreq("M", "e") - 6 for q in t3.index]
    ratio = pd.Series({m: t3[m] / ap[m] for m in t3.index if m in ap.index})
    ratio = np.exp(np.log(ratio).rolling(3, center=True, min_periods=1).mean())
    now_m = pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), "M")
    start = pd.Period(cfg["purchase"]["date"][:7], "M")
    months = pd.period_range(start, now_m, freq="M")
    apm = ap.reindex(pd.period_range(ap.index.min(), now_m, freq="M")).ffill()
    ro = np.array([m.ordinal for m in ratio.index])
    rat_m = np.interp([m.ordinal for m in months], ro, ratio.to_numpy())
    idx = np.array([apm[m] for m in months]) * rat_m                # EUR/m2, typical sold T3 in the parish
    sqm = cfg["facts"]["sqm"]
    typical = idx * sqm

    # --- peers: renovated + lift listings, asking less the typical gap, scaled to this flat's size
    gap = cfg["listings"]["ask_to_sale_gap"]
    el = cfg["listings"].get("size_elasticity", 0.9)
    peers = [x for x in listings if x["lift"] and x["renov"] and "renovation" not in x["status"]]
    eq = [x["price"] * (sqm / x["m2"]) ** el * (1 - gap) for x in peers]
    if len(eq) >= 3:
        lr = [math.log(e / typical[-1]) for e in eq]
        r_peer = statistics.median(lr) * len(lr) / (len(lr) + 1)    # shrink a thin set toward typical
        sig_id = max(0.06, 1.4826 * statistics.median([abs(v - statistics.median(lr)) for v in lr]))
    else:
        r_peer, sig_id = cfg["listings"]["fallback_premium"], 0.12
    # --- anchor (purchase) premium fading toward the peer premium
    hl = cfg.get("half_life_years", 4)
    lam = math.log(2) / (hl * 12)
    a = cfg["anchors"][0]
    i0 = list(months).index(pd.Period(a["date"][:7], "M"))
    r0 = math.log(a["value"] / typical[i0])
    a["model_typical"] = round(typical[i0])
    k = np.arange(len(months)) - i0
    rho = np.exp(-lam * np.clip(k, 0, None))
    resid = r_peer + (r0 - r_peer) * rho
    val = typical * np.exp(resid)
    sig = np.sqrt(0.03 ** 2 + sig_id ** 2 * (1 - rho ** 2))
    z = 1.2816

    def comp_row(x):
        return {"addr": (x.get("street") or (x["hood"] + (" (street withheld)" if x.get("hidden") else ""))), "typology": (re.search(r"\bT\d\b", x.get("title", "")) or [""])[0], "sqm": x["m2"], "price": x["price"],
                "ppsm": x["price"] / x["m2"], "floor": x.get("floor"), "lift": x["lift"],
                "status": ("renovated" if x["renov"] else "") + (" · " + x["status"].replace("_", " ") if x["status"] else ""),
                "eq_now": round(x["price"] * (sqm / x["m2"]) ** el * (1 - gap), -3),
                "url": "https://www.imovirtual.com/pt/anuncio/" + x["slug"] if x.get("slug") else None,
                "date": x.get("created")}

    comps = sorted((comp_row(x) for x in peers), key=lambda c: abs(math.log(c["sqm"] / sqm)))[:15]
    others = [x for x in listings if x not in peers]
    scatter = [[now_m.strftime("%Y-%m") + "-01", c["eq_now"]] for c in comps]
    t3_disp = pd.Series(inputs["t3_parish"])
    t3_disp = t3_disp[[q >= f"{start.year}Q{(start.month - 1) // 3 + 1}" for q in t3_disp.index]]
    apd = ap[ap.index >= start]
    eurm = lambda v: f"€{round(v):,}"                                                    # noqa: E731
    html = (f"<p>Level: INE median €/m² of T3 flats actually sold in the parish ({max(inputs['t3_parish'])}: "
            f"{eurm(inputs['t3_parish'][max(inputs['t3_parish'])])}/m², a 12-month rolling median, so it lags ~6 months). "
            f"Month-to-month movement: INE's bank-appraisal median for Lisbon apartments (to {max(inputs['appraisal'])}). "
            f"Typical {sqm} m² T3 today ≈ {eurm(typical[-1])}.</p>"
            f"<p>Purchase {eurm(a['value'])} vs typical {eurm(typical[i0])} then ({(math.exp(r0) - 1) * 100:+.0f}%): "
            f"part renovation + lift + micro-location, part the 2021 Golden-Visa buying rush (Lisbon's residential "
            f"route closed Jan 2022). That premium fades with a {hl}-yr half-life toward the live peer premium: "
            f"{len(peers)} renovated lift-served listings of {len(listings)} checked, at asking −{gap * 100:.0f}% → "
            f"{(math.exp(r_peer) - 1) * 100:+.0f}% vs typical. Premium now {(math.exp(resid[-1]) - 1) * 100:+.0f}%.</p>"
            f"<p>*“As this flat” = asking × ({sqm} m² / its m²)^{el} × (1 − {gap:.0%} asking-to-sale gap).</p>")
    return {
        "series": {"m": [str(m) for m in months], "v": [round(v) for v in val],
                   "lo": [round(v * math.exp(-z * s)) for v, s in zip(val, sig)],
                   "hi": [round(v * math.exp(z * s)) for v, s in zip(val, sig)],
                   "typical": [round(t) for t in typical]},
        "anchors": cfg["anchors"],
        "local_index": {"q": list(t3_disp.index), "v": [round(v / t3_disp.iloc[0] * 100, 2) for v in t3_disp]},
        "ref_index": {"label": "INE bank appraisal, Lisbon apartments", "m": [str(m) for m in apd.index],
                      "v": [round(v / apd.iloc[0] * 100, 2) for v in apd]},
        "comps": comps, "scatter": scatter, "realtor_estimate": None,
        "inputs": inputs,
        "model": {"html": html, "n_listings": len(listings), "n_peers": len(peers), "peer_premium": round(r_peer, 3),
                  "resid_now": round(float(resid[-1]), 3), "half_life_years": hl, "n_other": len(others)},
    }
