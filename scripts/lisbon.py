"""Lisbon flat: no sale-level data exists in Portugal, so the value is built from

  * INE (Statistics Portugal) median EUR/m2 of dwellings actually SOLD, T3, parish of
    Arroios (rolling 12 months, from tax records) - the local LEVEL - carried month to month by
    INE's monthly bank-appraisal median for Lisbon apartments (the broad DYNAMICS); the ratio
    between the two is smoothed so one noisy parish quarter can't swing the value;
  * the purchase price as anchor: its premium over a typical T3 of the same size fades (config
    half-life) toward the live PEER premium - renovated flats with a lift listed on imovirtual
    nearby, at asking minus the typical asking-to-sale gap. Listings are de-duplicated (the same
    flat is often advertised by several agents) and pooled over the last `pool_days` (default 90)
    so that one advert appearing or vanishing cannot move the headline by itself.

The 80% range adds up, in log points: the idiosyncratic spread between peers that the purchase
price no longer pins down, the standard error of the peer median, the uncertainty of the assumed
asking-to-sale gap, and the error of carrying the parish level forward from INE's last (lagged)
reading - the last one measured by a walk-forward test on the INE series itself (`index_check`).

INE sometimes refuses GitHub's runners: every series falls back to the last good copy kept in
data.json, so a failed fetch never blanks the page.
"""
import json
import math
import os
import re
import statistics
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ine import UA, _get, ine  # noqa: E402,F401  (stdlib-only; shared with the Mac seed job)

SEED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seed", "ine.json")
SERIES = {"t3_parish": ("0012241", "parish", "3"), "appraisal": ("0012248", "municipality", "1")}


def _now():
    return datetime.now(timezone.utc)


def lkey(x):
    """Identity of a listing for de-duplication: same neighbourhood + street + size is one flat, however
    many agents advertise it. Without a usable street the price has to stand in for it."""
    st = (x.get("street") or "").strip().lower()
    m2 = round(float(x["m2"]))
    if x.get("hidden"):                                        # street withheld: rooms stand in for it
        return f"{x.get('hood')}|~|{m2}|{x.get('rooms')}"
    if not st or st in ("lisboa", str(x.get("hood", "")).lower()):
        return f"{x.get('hood')}|~|{m2}|{round(float(x['price']))}"
    return f"{x.get('hood')}|{st}|{m2}"


def dedupe(items):
    seen, out = set(), []
    for x in items:
        k = lkey(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def index_check(t3, ap, max_h=4, first=7):
    """Walk-forward test of the level nowcast on INE's own history: at each past quarter, hold the
    smoothed sold/appraisal ratio known then and carry it h quarters ahead with the appraisal index;
    compare with what INE later published, and with simply holding the last level flat.
    Returns {h: {n, rmse, naive_rmse}} in log points."""
    qs = sorted(t3)
    cen = [pd.Period(q, "Q").asfreq("M", "e") - 6 for q in qs]
    out = {}
    for h in range(1, max_h + 1):
        e, e0 = [], []
        for i in range(first, len(qs) - h):
            if cen[i + h] not in ap.index or any(c not in ap.index for c in cen[max(0, i - 1):i + 1]):
                continue
            r = np.mean([math.log(t3[qs[j]] / ap[cen[j]]) for j in (i - 1, i)])   # = the smoothed ratio's last point
            e.append(math.log(t3[qs[i + h]]) - (r + math.log(ap[cen[i + h]])))
            e0.append(math.log(t3[qs[i + h]] / t3[qs[i]]))
        if len(e) >= 6:
            out[h] = {"n": len(e), "rmse": round(float(np.sqrt(np.mean(np.square(e)))), 4),
                      "naive_rmse": round(float(np.sqrt(np.mean(np.square(e0)))), 4)}
    return out


def imovirtual(cfg, hide=()):
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
    for x in near:                                              # same-street listings would point at the flat
        if x.get("street") and any(h in x["street"].lower() for h in hide):
            x.update(street=None, slug_hidden=x["slug"], hidden=True)
    near = dedupe(near)                                         # don't spend detail fetches on re-posts
    out = []
    for x in near[:c.get("max_detail", 30)]:                    # detail page: lift + condition
        try:
            h = _get("https://www.imovirtual.com/pt/anuncio/" + (x.get("slug_hidden") or x["slug"]), accept="text/html")
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
        row = dict(x, lift=lift, renov=renov, status=status, floor=str(floor).replace("floor_", ""),
                   created=(ad.get("createdAt") or "")[:10], title=ad.get("title", ""))
        if row.pop("slug_hidden", None):
            row.update(slug=None, title="")
        out.append(row)
        time.sleep(1.0)
    return out


def value(pcfg, prev, STATUS, subj=None):
    key, cfg = pcfg["key"], pcfg
    prev = prev or {}
    old = (prev.get("inputs") or {})
    inputs = {}
    seed = {}
    try:
        with open(SEED) as f:
            seed = json.load(f)
    except Exception:
        pass
    on_ci = bool(os.environ.get("GITHUB_ACTIONS"))
    for name, (v, geo, d) in SERIES.items():
        if on_ci and seed.get(name):                               # INE blocks runners: use the Mac's weekly copy
            inputs[name] = seed[name]
            STATUS[f"{key}:ine_{name}"] = f"seed {seed.get('updated', '?')[:10]}, to {max(seed[name])}"
            continue
        try:
            inputs[name] = ine(v, cfg["ine"][geo], d)
            STATUS[f"{key}:ine_{name}"] = f"ok to {max(inputs[name])}"
        except Exception as e:
            fallback = seed.get(name) or old.get(name)
            if not fallback:
                raise
            inputs[name] = fallback
            STATUS[f"{key}:ine_{name}"] = f"FAIL {type(e).__name__}: kept last good (cache)"
    hide = [h.lower() for h in (subj or {}).get("hide_streets", [])]   # same-street listings would point at the flat
    today = _now().strftime("%Y-%m-%d")
    fresh = True
    try:
        listings = imovirtual(cfg, hide)
        STATUS[f"{key}:imovirtual"] = f"ok {len(listings)} detailed"
    except Exception as e:
        listings, fresh = old.get("listings", []), False
        STATUS[f"{key}:imovirtual"] = f"FAIL {type(e).__name__}: kept last good (cache)"
    for x in listings:
        if x.get("street") and any(h in x["street"].lower() for h in hide):
            x.update(street=None, slug=None, title="", hidden=True)
    listings = dedupe(listings)
    inputs["listings"] = listings
    # pool: every distinct listing seen in the last pool_days, at its latest asking price
    pool_days = cfg["listings"].get("pool_days", 90)
    pool = {k: v for k, v in (old.get("listing_pool") or {}).items()}
    if not pool:                                                   # first run with a pool: start from the last run's
        pool = {lkey(x): dict(x, seen=(prev.get("runs") or [{"d": today}])[-1]["d"]) for x in old.get("listings", [])}
    if fresh:
        for x in listings:
            if x.get("slug"):                                      # the same advert can be geocoded to another street
                pool = {k: v for k, v in pool.items() if v.get("slug") != x["slug"]}
            pool[lkey(x)] = dict(x, seen=today)
    cut = (pd.Timestamp(today) - pd.Timedelta(days=pool_days)).strftime("%Y-%m-%d")
    pool = {k: v for k, v in sorted(pool.items()) if v.get("seen", today) >= cut}
    inputs["listing_pool"] = pool

    # --- monthly index: appraisal dynamics x smoothed (sold T3 parish / appraisal) level ratio
    ap = pd.Series(inputs["appraisal"])
    ap.index = pd.PeriodIndex(ap.index, freq="M")
    t3 = pd.Series(inputs["t3_parish"])
    # INE's 12-month rolling median at quarter Qn is centred ~half a year earlier
    t3.index = [pd.Period(q, "Q").asfreq("M", "e") - 6 for q in t3.index]
    ratio = pd.Series({m: t3[m] / ap[m] for m in t3.index if m in ap.index})
    ratio = np.exp(np.log(ratio).rolling(3, center=True, min_periods=1).mean())
    now_m = pd.Period(_now().strftime("%Y-%m"), "M")
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
    peers = [x for x in pool.values() if x["lift"] and x["renov"] and "renovation" not in x["status"]]
    eq = [x["price"] * (sqm / x["m2"]) ** el * (1 - gap) for x in peers]
    if len(eq) >= 3:
        lr = [math.log(e / typical[-1]) for e in eq]
        shrink = len(lr) / (len(lr) + 1)
        r_peer = statistics.median(lr) * shrink                     # shrink a thin set toward typical
        mad = 1.4826 * statistics.median([abs(v - statistics.median(lr)) for v in lr])
        # a spread measured on a handful of adverts is itself shaky (it flipped between 0.09 and 0.28 from one
        # refresh to the next): pull it toward the 0.12 used when there are no peers, with the weight of 4 adverts
        sig_id = max(0.06, math.sqrt((len(lr) * mad ** 2 + 4 * 0.12 ** 2) / (len(lr) + 4)))
        se_peer = 1.2533 * sig_id / math.sqrt(len(lr))              # standard error of a median
    else:
        r_peer, sig_id, shrink, se_peer = cfg["listings"]["fallback_premium"], 0.12, 0.0, 0.12
    gap_sd = cfg["listings"].get("ask_to_sale_gap_sd", 0.03)
    # --- how well does carrying the parish level forward with the appraisal index work? (walk-forward)
    chk = index_check(inputs["t3_parish"], ap)
    last_c = max(t3.index)                                          # centre month of INE's latest parish reading
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
    # index error: 3% where INE has a parish reading; beyond it, the walk-forward error at that horizon
    g = np.array([max(0, (m - last_c).n) for m in months])          # months past the last parish reading
    hq = np.clip(np.ceil(g / 3), 0, 4).astype(int)
    s_idx = np.array([max(0.03, chk[h]["rmse"]) if h and h in chk else 0.03 for h in hq])
    w_idx = 1 - shrink * (1 - rho)                                  # the peer part is index-free except its shrinkage
    sig = np.sqrt(0.03 ** 2 + w_idx ** 2 * (s_idx ** 2 - 0.03 ** 2) + (1 - rho) ** 2 * (se_peer ** 2 + gap_sd ** 2)
                  + sig_id ** 2 * (1 - rho ** 2))
    z = 1.2816

    def comp_row(x):
        return {"addr": (x.get("street") or (x["hood"] + (" (street withheld)" if x.get("hidden") else ""))), "typology": (re.search(r"\bT\d\b", x.get("title", "")) or [""])[0], "sqm": x["m2"], "price": x["price"],
                "ppsm": x["price"] / x["m2"], "floor": x.get("floor"), "lift": x["lift"],
                "status": ("renovated" if x["renov"] else "") + (" · " + x["status"].replace("_", " ") if x["status"] else "")
                + (" · last seen " + x["seen"][5:] if x.get("seen", today) != today else ""), "seen": x.get("seen", today),
                "eq_now": round(x["price"] * (sqm / x["m2"]) ** el * (1 - gap), -3),
                "url": "https://www.imovirtual.com/pt/anuncio/" + x["slug"] if x.get("slug") else None,
                "date": x.get("created")}

    comps = sorted((comp_row(x) for x in peers), key=lambda c: abs(math.log(c["sqm"] / sqm)))[:15]
    others = [x for x in pool.values() if x not in peers]
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
            f"{len(peers)} distinct renovated lift-served listings of {len(pool)} seen in the last {pool_days} days "
            f"(re-posts of the same flat counted once), at asking −{gap * 100:.0f}% → "
            f"{(math.exp(r_peer) - 1) * 100:+.0f}% vs typical. Premium now {(math.exp(resid[-1]) - 1) * 100:+.0f}%.</p>"
            + (f"<p>Honesty check on the index: INE's parish figure is centred on {last_c.strftime('%b %Y')}, "
               f"{int(g[-1])} months back; the gap is bridged with the appraisal index. Replayed on INE's own history "
               f"({chk[int(hq[-1])]['n']} past quarters), that bridge missed the parish level INE later published by "
               f"±{chk[int(hq[-1])]['rmse'] * 100:.0f}% (RMSE) – no better than holding the level flat "
               f"(±{chk[int(hq[-1])]['naive_rmse'] * 100:.0f}%), though less biased while prices rise. "
               f"So it is used for direction, and that error is in the range.</p>"
               if int(hq[-1]) in chk and chk[int(hq[-1])]["rmse"] >= 0.9 * chk[int(hq[-1])]["naive_rmse"] else
               f"<p>Index check: bridging the {int(g[-1])} months since INE's latest parish reading with the appraisal "
               f"index missed by ±{chk[int(hq[-1])]['rmse'] * 100:.0f}% (RMSE, {chk[int(hq[-1])]['n']} past quarters) "
               f"vs ±{chk[int(hq[-1])]['naive_rmse'] * 100:.0f}% for holding the level flat; that error is in the range.</p>"
               if int(hq[-1]) in chk else "")
            + f"<p>80% range, in log points: peer-to-peer spread {sig_id:.2f} (the part the purchase price no longer "
              f"pins), peer-median error {se_peer:.2f}, asking-to-sale gap ±{gap_sd:.2f}, index {s_idx[-1]:.2f} "
              f"→ total ±{sig[-1] * 100:.0f}% (1 sd).</p>"
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
                  "resid_now": round(float(resid[-1]), 3), "half_life_years": hl, "n_other": len(others),
                  "n_pool": len(pool), "pool_days": pool_days, "index_check": {str(h): v for h, v in chk.items()},
                  "sigma_parts": {"peer_spread": round(sig_id, 3), "peer_median_se": round(se_peer, 3),
                                  "gap_sd": gap_sd, "index": round(float(s_idx[-1]), 3), "total": round(float(sig[-1]), 3)}},
    }
