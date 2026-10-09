#!/usr/bin/env python3
"""Rebuild data.json: comp-based value history for each property.

US homes: pull ~10 years of SOLD comps around each home (realtor.com via HomeHarvest,
plus Redfin's sold export for the last year), fit the local hedonic model in
hedonic.py, and predict the home's value every month since purchase. The home's
own known prices (purchase, appraisal) pin the line: the gap between what the model
says a home like it was worth and what it actually traded for is carried forward,
fading with a half-life (config `half_life_years`) because part of any one sale's
premium or discount is noise.

Lisbon: see lisbon.py (INE official transaction prices per m2 for the parish).

Street addresses and coordinates come from the SUBJECTS_JSON secret and are never
written to data.json or printed (this repo and its Actions logs are public).
"""
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hedonic as H  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data.json")
CACHE = os.path.join(ROOT, "cache")          # Actions cache only (gitignored): holds coordinates
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36")
ZBASE = "https://files.zillowstatic.com/research/public_csvs/zhvi/"
STATUS = {}


def log(*a):
    print(*a, flush=True)


def load_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def subjects():
    raw = os.environ.get("SUBJECTS_JSON") or (open(os.path.join(ROOT, ".subjects.json")).read()
                                              if os.path.exists(os.path.join(ROOT, ".subjects.json")) else "")
    if not raw:
        sys.exit("SUBJECTS_JSON not set")
    return json.loads(raw)


def get(url, tries=3, **kw):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=kw.pop("timeout", 90), **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            if i == tries - 1:
                raise
            log("  retry", url[:70], type(e).__name__)
            time.sleep(4 * (i + 1))


# ---------------------------------------------------------------- reference series
def fred(sid):
    """FRED CSV, no key. Default UA on purpose (a browser UA makes FRED hang)."""
    r = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df.v, errors="coerce")
    df = df.dropna()
    return pd.Series(df.v.values, index=pd.to_datetime(df.date).dt.strftime("%Y-%m"))


def zillow(file, col, rid):
    """Stream a ZHVI CSV and keep the one region (files are 20-120 MB)."""
    import csv
    with requests.get(ZBASE + file, stream=True, timeout=300) as r:
        r.raise_for_status()
        rows = csv.reader(ln.decode("utf-8") for ln in r.iter_lines())
        header = next(rows)
        ci, si = header.index(col), header.index("State") if "State" in header else None
        for parts in rows:
            if parts[ci] == rid and (si is None or parts[si] in ("CA", "")):
                return pd.Series({h[:7]: float(p) for h, p in zip(header, parts)
                                  if h[:2] in ("19", "20") and p}).sort_index()
    raise RuntimeError(f"Zillow region {rid} not in {file}")


def benchmarks():
    """Total-return S&P 500 + AAPL monthly, from car-value-tracker (one source of truth),
    with AAPL 2015-01..2016-09 from a static seed (CVT's Nasdaq history starts 2016-10)."""
    j = get("https://hlash99.github.io/car-value-tracker/data.json").json()["bench"]["series"]
    out = {"S&P 500": dict(j["S&P 500"]["m"]), "AAPL": dict(j["AAPL"]["m"])}
    seed = load_json(os.path.join(ROOT, "seed", "aapl_2015_2016.json"), {}).get("m", {})
    for k, v in seed.items():
        out["AAPL"].setdefault(k, v)
    return {k: dict(sorted(v.items())) for k, v in out.items()}


# ---------------------------------------------------------------- sold comps
COLS = ["src", "pid", "addr", "unit", "city", "date", "price", "sqft", "lot", "beds", "baths",
        "yb", "hoa", "park", "text", "lat", "lon", "url", "est"]


def num(s):
    return pd.to_numeric(s, errors="coerce").astype("float64")


def sales_homeharvest(subj, pc, date_from):
    from homeharvest import scrape_property
    d = scrape_property(location=subj["search"], listing_type="sold", radius=pc["radius_mi"],
                        date_from=date_from, date_to=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        property_type=pc["types"], extra_property_data=False, parallel=False, limit=5000)
    if d is None or not len(d):
        raise RuntimeError("no rows")
    o = pd.DataFrame({
        "src": "realtor", "pid": d.property_id.astype(str), "addr": d.formatted_address,
        "unit": d.get("unit"), "city": d.city, "date": pd.to_datetime(d.last_sold_date, errors="coerce"),
        "price": num(d.sold_price), "sqft": num(d.sqft), "lot": num(d.lot_sqft), "beds": num(d.beds),
        "baths": num(d.full_baths).fillna(1) + 0.5 * num(d.half_baths).fillna(0), "yb": num(d.year_built),
        "hoa": num(d.hoa_fee), "park": num(d.parking_garage), "text": d.text, "lat": num(d.latitude),
        "lon": num(d.longitude), "url": d.property_url, "est": num(d.estimated_value)})
    return o


def sales_redfin(subj, pc, days=365):
    lat, lon, rad = subj["lat"], subj["lon"], pc["radius_mi"]
    dlat, dlon = rad / 69.05, rad / (69.05 * math.cos(math.radians(lat)))
    pts = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat), (lon + dlon, lat + dlat), (lon - dlon, lat + dlat),
           (lon - dlon, lat - dlat)]
    params = {"al": 1, "num_homes": 350, "sold_within_days": days, "status": 9, "uipt": pc["redfin_uipt"],
              "poly": ",".join(f"{x:.6f} {y:.6f}" for x, y in pts), "v": 8}
    r = get("https://www.redfin.com/stingray/api/gis-csv", params=params,
            headers={"User-Agent": UA, "Referer": "https://www.redfin.com/"})
    if not r.text.startswith("SALE TYPE"):
        raise RuntimeError("unexpected body")
    d = pd.read_csv(io.StringIO(r.text))
    d = d[d["SALE TYPE"].eq("PAST SALE")]
    url = d.filter(like="URL").iloc[:, 0] if len(d.filter(like="URL").columns) else None
    return pd.DataFrame({
        "src": "redfin", "pid": None, "addr": d.ADDRESS + ", " + d.CITY + ", " + d["STATE OR PROVINCE"] + " "
        + d["ZIP OR POSTAL CODE"].astype(str), "unit": None, "city": d.CITY,
        "date": pd.to_datetime(d["SOLD DATE"], format="%B-%d-%Y", errors="coerce"), "price": num(d.PRICE),
        "sqft": num(d["SQUARE FEET"]), "lot": num(d["LOT SIZE"]), "beds": num(d.BEDS), "baths": num(d.BATHS),
        "yb": num(d["YEAR BUILT"]), "hoa": num(d["HOA/MONTH"]), "park": np.nan, "text": np.nan,
        "lat": num(d.LATITUDE), "lon": num(d.LONGITUDE), "url": url, "est": np.nan})


def addr_key(a):
    a = str(a).lower().replace("#", " unit ").replace(",", " ")
    for k, v in ((" apt ", " unit "), (" ste ", " unit "), (" street", " st"), (" avenue", " ave")):
        a = a.replace(k, v)
    t = a.split()
    return " ".join(t[:2] + ([t[t.index("unit") + 1]] if "unit" in t and t.index("unit") + 1 < len(t) else []))


def all_sales(key, subj, pc, date_from):
    frames = []
    cpath = os.path.join(CACHE, f"sales_{key}.pkl")

    def realtor():
        for i in range(3):
            try:
                return sales_homeharvest(subj, pc, date_from)
            except Exception:
                if i == 2:
                    if os.path.exists(cpath):                  # keep the 10-yr history from the last good run
                        c = pd.read_pickle(cpath)
                        STATUS[f"{key}:realtor_cache"] = "realtor failed: used cached history"
                        return c[c.src == "realtor"]
                    raise
                time.sleep(10 * (i + 1))
    for name, fn in (("realtor", realtor),
                     ("redfin", lambda: sales_redfin(subj, pc))):
        try:
            f = fn()
            frames.append(f)
            STATUS[f"{key}:{name}"] = f"ok {len(f)} sales"
        except Exception as e:
            STATUS[f"{key}:{name}"] = f"FAIL {type(e).__name__}"
        log(f"  {key} {name}: {STATUS[f'{key}:{name}']}")
    if not frames:
        if os.path.exists(cpath):
            STATUS[f"{key}:sales"] = "using cached sales"
            return pd.read_pickle(cpath)
        raise RuntimeError("no sales source worked and no cache")
    d = pd.concat(frames, ignore_index=True)
    d = d.dropna(subset=["date", "price", "lat", "lon"])
    d["k"] = d.addr.map(addr_key)
    d["ym"] = d.date.dt.to_period("M")
    d = d.sort_values("src")                                   # 'realtor' before 'redfin': keep realtor's remarks
    d = d.drop_duplicates(["k", "ym"]).drop(columns=["k", "ym"])
    # cache only when realtor (the 10-year history) succeeded, so a redfin-only run can't shrink it
    if f"{key}:realtor_cache" not in STATUS and any(f.src.iloc[0] == "realtor" for f in frames if len(f)):
        os.makedirs(CACHE, exist_ok=True)
        d.to_pickle(cpath)
    return d


# ---------------------------------------------------------------- model features
def model_frame(d, subj, pc, kind):
    d = d.copy()
    d = d[d.city.isin(pc["cities"]) & d.sqft.between(*pc["sqft"]) & (d.price > 150_000)]
    if subj.get("realtor_id"):
        d = d[d.pid.astype(str) != str(subj["realtor_id"])]   # never the home itself
    dy = (d.lat - subj["lat"]) * 69.05
    dx = (d.lon - subj["lon"]) * 69.05 * math.cos(math.radians(subj["lat"]))
    d["dx"], d["dy"], d["dist"] = dx, dy, np.hypot(dx, dy)
    d = d[d.dist <= pc["radius_mi"]]
    fl = pd.DataFrame([H.text_flags(t) for t in d.text], index=d.index).astype(float)
    known = d.text.notna()
    for c in fl:                                               # Redfin rows have no remarks: impute the mean
        fl.loc[~known, c] = fl.loc[known, c].mean() if known.any() else 0
    d = d.join(fl)
    d["lsq"] = np.log(d.sqft)
    d["beds"] = d.beds.fillna(d.beds.median())
    d["baths"] = d.baths.fillna(d.baths.median())
    d["yb"] = d.yb.fillna(d.yb.median())
    d["new"] = (d.yb >= 2010).astype(float)
    if kind == "sfr":
        d["llot"] = np.log(d.lot.fillna(d.lot.median()).clip(2000, 20000))
        d["rc"] = (d.city == "Redwood City").astype(float)
        feats = ["lsq", "llot", "beds", "baths", "new", "remodel", "fixer", "rc", "dx", "dy"]
    else:
        d["pre40"] = (d.yb < 1940).astype(float)
        d["hoa"] = d.hoa.fillna(d.hoa.median()).clip(0, 2500) / 100
        d["park"] = (d.park.fillna(0) > 0).astype(float)
        # Building size: distinct units seen selling at the same street address. Big 1990s
        # complexes (no view, amenity HOAs) trade far below small pre-war buildings.
        street = d.addr.astype(str).str.split(",").str[0].str.replace(r"\s+(apt|unit|#|ste).*$", "", case=False,
                                                                       regex=True).str.lower()
        units = d.assign(s=street).groupby("s").addr.nunique()
        d["lbld"] = np.log(street.map(units).clip(1, 200))
        feats = ["lsq", "beds", "baths", "pre40", "new", "remodel", "fixer", "view", "tic", "park", "hoa", "lbld",
                 "dx", "dy"]
    d["q"] = d.date.dt.to_period("Q")
    return d, feats


def subject_x(pcfg, month, kind):
    x = {}
    for f in pcfg["features"]:
        if f["from"] <= month:
            x.update({k: v for k, v in f.items() if k != "from"})
    x["lsq"] = math.log(x["sqft"])
    if kind == "sfr":
        x["llot"] = math.log(min(max(x["lot"], 2000), 20000))
    else:
        x["lbld"] = math.log(max(x.get("bldg_sales", 1), 1))
    x["dx"] = x["dy"] = 0.0
    return x


# ---------------------------------------------------------------- valuation
def month_range(a, b):
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def months_between(a, b):
    pa, pb = pd.Period(a, "M"), pd.Period(b, "M")
    return (pb - pa).n


def value_us(pcfg, subj, half_life, zl):
    key, kind = pcfg["key"], pcfg["kind"]
    start = pcfg["purchase"]["date"][:7]
    date_from = f"{int(start[:4]) - 2}-01-01"
    raw = all_sales(key, subj, pcfg["comps"], date_from)
    est = raw[raw.pid.astype(str) == str(subj.get("realtor_id"))].est.dropna()
    if est.empty and subj.get("realtor_id"):                       # not in the sold list: look the parcel up
        try:
            from homeharvest import scrape_property
            sd = scrape_property(location=subj["search"], extra_property_data=False, parallel=False, limit=50)
            est = num(sd[sd.property_id.astype(str) == str(subj["realtor_id"])].estimated_value).dropna()
        except Exception as e:
            log(f"  {key}: realtor estimate lookup failed {type(e).__name__}")
    d, feats = model_frame(raw, subj, pcfg["comps"], kind)
    zq = zl.groupby(pd.PeriodIndex(zl.index, freq="M").asfreq("Q")).mean()
    zq = zq[zq.index >= min(d.q.min(), pd.Period(start, "M").asfreq("Q") - 2)]
    last_q = max(zq.index[-1], pd.Period(datetime.now(timezone.utc).strftime("%Y-%m"), "M").asfreq("Q"))
    zq = zq.reindex(pd.period_range(zq.index[0], last_q, freq="Q")).ffill()
    d = d[d.q.isin(zq.index)]
    w = np.exp(-d.dist.to_numpy() / pcfg["comps"]["decay_mi"])
    m = H.fit(d, feats, zq, w)
    log(f"  {key}: model n={m['n']} sigma={m['sigma']:.3f}")
    d["res"] = m["res"]

    # Peer premium: how far above/below the model's "typical" home the closest true peers
    # trade - hand-picked same-view units (SF) or the immediate blocks (Menlo). The home's
    # own anchor premium fades toward THIS, not toward zero, so a lasting edge (a view) stays.
    pc = pcfg["comps"]
    keys = d.addr.map(addr_key)
    if pc.get("peer_units"):
        pk = [u.lower() for u in pc["peer_units"]]
        is_peer = keys.map(lambda k: any(k == u or (len(u.split()) == 2 and k.startswith(u + " ")) or k == u
                                         for u in pk))
        peer_label = f"{len(pk)} hand-picked same-view units/buildings"
    else:
        is_peer = d.dist <= pc.get("peer_radius_mi", 0.15)
        peer_label = f"sales within {pc.get('peer_radius_mi', 0.15)} mi"
    pe = d[is_peer]
    age = (pd.Timestamp(datetime.now(timezone.utc).date()) - pe.date).dt.days / 365.25
    pw = 0.5 ** (age / 3.0)                                        # recent peer sales count most
    r_peer = float((pw * pe.res).sum() / (pw.sum() + 1.0)) if len(pe) else 0.0   # +1 shrinks a thin set to 0
    peer = {"label": peer_label, "n": int(len(pe)), "n_eff": round(float(pw.sum()), 1), "premium": round(r_peer, 3),
            # radius peers would pinpoint the home's block, so only hand-picked peers are listed
            "sales": [] if not pc.get("peer_units") else
            [{"addr": str(r.addr).split(",")[0], "date": r.date.strftime("%Y-%m-%d"), "price": int(r.price),
              "res": round(float(r.res), 3)} for _, r in pe.sort_values("date").tail(12).iterrows()]}
    log(f"  {key}: peers n={peer['n']} premium={r_peer:+.3f}")

    # monthly time effect: quarterly tau interpolated at quarter mid-months
    now_m = datetime.now(timezone.utc).strftime("%Y-%m")
    months = month_range(start, now_m)
    qmid = np.array([(q.asfreq("M", "s") + 1).ordinal for q in m["tau"].index])
    mo = np.array([pd.Period(x, "M").ordinal for x in months])
    tau_m = np.interp(mo, qmid, m["tau"].to_numpy())

    base, se = [], []
    for x_m, t in zip(months, tau_m):
        x = subject_x(pcfg, x_m, kind)
        q = pd.Period(x_m, "M").asfreq("Q")
        lp, s = H.predict(m, x, q)
        base.append(lp - m["tau"][q] + t)
        se.append(s)
    base, se = np.array(base), np.array(se)

    # anchor residuals: actual known prices vs what the model says a home like it was worth
    lam = math.log(2) / (half_life * 12)
    idx = {x: i for i, x in enumerate(months)}
    anchors = sorted(pcfg["anchors"], key=lambda a: a["date"])
    resid = np.zeros(len(months))
    pts = []                                                       # (month index, residual, strength)
    for a in anchors:
        i = idx[a["date"][:7]]
        r_a = math.log(a["value"]) - base[i]
        wgt = a.get("weight", 1.0)
        if pts:                                                    # blend a soft anchor with what we'd carry
            j, r0, _ = pts[-1]
            r_a = wgt * r_a + (1 - wgt) * (r_peer + (r0 - r_peer) * math.exp(-lam * (i - j)))
        pts.append((i, r_a, wgt))
        a["model_typical"] = round(math.exp(base[i]))
    rho = np.zeros(len(months))
    for i in range(len(months)):
        prev = [p for p in pts if p[0] <= i]
        nxt = [p for p in pts if p[0] > i]
        if prev and nxt:
            (i0, r0, _), (i1, r1, _) = prev[-1], nxt[0]
            resid[i] = r0 + (r1 - r0) * (i - i0) / (i1 - i0)
        elif prev:
            i0, r0, _ = prev[-1]
            resid[i] = r_peer + (r0 - r_peer) * math.exp(-lam * (i - i0))
        rho[i] = max(p[2] * math.exp(-lam * abs(i - p[0])) for p in pts)
    sig = np.sqrt(se ** 2 + (m["sigma"] ** 2) * (1 - rho ** 2))
    val = np.exp(base + resid)
    z80 = 1.2816

    # comps: the last 12 months' sales, each translated into "this home, today"
    x_now = subject_x(pcfg, now_m, kind)
    q_now = pd.Period(now_m, "M").asfreq("Q")
    lp_now = H.predict(m, x_now, q_now)[0]
    beta_x = lambda row: sum(H.coef(m, f) * float(row[f]) for f in feats)   # noqa: E731
    bx_now = sum(H.coef(m, f) * float(x_now[f]) for f in feats)
    d["eq_now"] = [math.exp(math.log(p) + bx_now - beta_x(r) + m["tau"][q_now] - m["tau"][q] + resid[-1])
                   for p, (_, r), q in zip(d.price, d.iterrows(), d.q)]
    recent = d[d.date >= pd.Timestamp(datetime.now(timezone.utc).date()) - pd.Timedelta(days=365)].copy()
    recent["sim"] = np.exp(-recent.dist / pcfg["comps"]["decay_mi"]) * np.exp(
        -abs(np.log(recent.sqft / x_now["sqft"])) / 0.25)
    hide = [h.lower() for h in subj.get("hide_streets", [])]          # same-street comps would point at the home
    recent = recent[~recent.addr.astype(str).str.lower().map(lambda a: any(h in a for h in hide))]
    recent = recent.sort_values("sim", ascending=False).head(15)
    band = lambda mi: "<¼ mi" if mi < .25 else "¼–½ mi" if mi < .5 else "½–¾ mi" if mi < .75 else "¾–1 mi"  # noqa
    comps = [{
        "addr": str(r.addr).split(",")[0], "city": r.city, "date": r.date.strftime("%Y-%m-%d"),
        "price": int(r.price), "sqft": int(r.sqft), "lot": None if pd.isna(r.lot) else int(r.lot),
        "beds": None if pd.isna(r.beds) else float(r.beds), "baths": float(r.baths),
        "yb": int(r.yb), "ppsf": round(r.price / r.sqft), "dist": band(r.dist),
        "flags": [f for f in ("remodel", "fixer", "view", "tic") if f in r and r[f] == 1],
        "eq_now": round(r.eq_now, -3), "url": r.url if isinstance(r.url, str) else None, "src": r.src,
    } for _, r in recent.iterrows()]

    # scatter: every nearby sale since purchase as "this home, at that date"
    near = d[(d.date >= pd.Timestamp(start + "-01")) & (np.exp(-d.dist / pcfg["comps"]["decay_mi"]) >= 0.3)]
    scatter = []
    for (_, r), q in zip(near.iterrows(), near.q):
        ym = r.date.strftime("%Y-%m")
        if ym not in idx:
            continue
        x_t = subject_x(pcfg, ym, kind)
        bx_t = sum(H.coef(m, f) * float(x_t[f]) for f in feats)
        scatter.append([r.date.strftime("%Y-%m-%d"), round(math.exp(math.log(r.price) + bx_t - beta_x(r)
                                                                         + resid[idx[ym]]), -3)])

    ztail = zl[zl.index >= start]
    tau_s = m["tau"][m["tau"].index >= pd.Period(start, "M").asfreq("Q")]
    nice = {"lsq": "Size (log sq ft)", "llot": "Lot (log)", "beds": "Per bedroom", "baths": "Per bath",
            "remodel": "Remodeled (remarks)", "fixer": "Fixer (remarks)", "view": "View (remarks)",
            "tic": "TIC (remarks)", "park": "Garage parking", "lbld": "Building size (log units sold)", "rc": "Redwood City address", "new": "Built 2010+"}
    return {
        "series": {"m": months, "v": [round(v) for v in val],
                   "lo": [round(v * math.exp(-z80 * s)) for v, s in zip(val, sig)],
                   "hi": [round(v * math.exp(z80 * s)) for v, s in zip(val, sig)],
                   "typical": [round(math.exp(b)) for b in base]},
        "anchors": anchors,
        "local_index": {"q": [str(q) for q in tau_s.index], "v": [round(math.exp(t - tau_s.iloc[0]) * 100, 2)
                                                                 for t in tau_s],
                        "n": [int(m["n_q"][q]) for q in tau_s.index]},
        "ref_index": {"label": pcfg["zillow"]["label"], "m": list(ztail.index),
                   "v": [round(v / ztail.iloc[0] * 100, 2) for v in ztail]},
        "comps": comps, "scatter": scatter,
        "realtor_estimate": None if est.empty else int(est.iloc[0]),
        "model": {"n": int(m["n"]), "sigma": round(m["sigma"], 3), "half_life_years": half_life,
                  "since": str(d.date.min().date()), "resid_now": round(float(resid[-1]), 3), "peer": peer,
                  "effects": {nice.get(f, f): round(H.coef(m, f), 4) for f in feats if f in nice},
                  "size_elasticity": round(H.coef(m, "lsq"), 3)},
    }


# ---------------------------------------------------------------- main
def main():
    cfg = load_json(os.path.join(ROOT, "config.json"))
    subs = subjects()
    prev = load_json(DATA, {}) or {}
    prev_props = {p["key"]: p for p in prev.get("properties", [])}
    out = {"updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "properties": []}

    ref = prev.get("ref", {})
    for name, fn in (("cpi_us", lambda: fred("CPIAUCSL")), ("hicp_pt", lambda: fred("CP0000PTM086NEST")),
                     ("eurusd", lambda: fred("EXUSEU"))):
        try:
            ref[name] = {k: round(float(v), 4) for k, v in fn().items() if k >= "2014-01"}
            STATUS[name] = f"ok to {max(ref[name])}"
        except Exception as e:
            STATUS[name] = f"FAIL {type(e).__name__} (kept last)"
        log(" ", name, STATUS[name])
    try:
        ref["eurusd_now"] = float(fred("DEXUSEU").iloc[-1])
    except Exception:
        pass
    try:
        ref["bench"] = benchmarks()
        STATUS["bench"] = "ok"
    except Exception as e:
        STATUS["bench"] = f"FAIL {type(e).__name__} (kept last)"
    out["ref"] = ref

    for pcfg in cfg["properties"]:
        key = pcfg["key"]
        log(f"[{key}]")
        p = {k: pcfg[k] for k in ("key", "name", "icon", "ccy", "kind", "area", "facts", "purchase", "checks")}
        try:
            if pcfg["kind"] in ("sfr", "condo"):
                z = pcfg["zillow"]
                zl = zillow(z["file"], z["col"], z["id"])
                STATUS[f"{key}:zillow"] = f"ok to {zl.index[-1]}"
                p.update(value_us(pcfg, subs[key], cfg["half_life_years"], zl))
            else:
                import lisbon
                p.update(lisbon.value(pcfg, prev_props.get(key), STATUS, subs.get(key)))
            STATUS[key] = "ok"
        except Exception as e:
            import traceback
            traceback.print_exc()
            STATUS[key] = f"FAIL {type(e).__name__}: kept last good"
            if key in prev_props:
                p = prev_props[key]
            else:
                continue
        # one point per day: the headline as published, to show how the estimate itself moves
        runs = [r for r in (prev_props.get(key, {}).get("runs") or []) if r["d"] != out["updated"][:10]]
        if p.get("series"):
            runs.append({"d": out["updated"][:10], "v": p["series"]["v"][-1]})
        p["runs"] = runs[-400:]
        out["properties"].append(p)

    out["status"] = STATUS
    with open(DATA, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    log("status:", json.dumps(STATUS, indent=1))
    if not out["properties"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
