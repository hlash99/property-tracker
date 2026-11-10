"""Honesty checks for the US hedonic model, run on the sales already pulled for the refresh.

Two questions, both answered out of sample and published as aggregates only (no addresses):

1. holdout()  - does the model price sales it has not seen? For each of the last few quarters, refit
   on EARLIER sales only and predict that quarter's sales. Compared with the plain comps rule of
   thumb (median $/sq ft of the previous year's sales x the home's size), and used to check that the
   "typical error" the page quotes holds up on unseen sales.
2. repeat_sales() - is pinning the line to the home's own purchase price justified, and for how long?
   Homes around the subject that sold twice are the direct test: the first sale's premium over the
   model is carried to the second sale (a) not at all, (b) fully, (c) fading with the configured
   half-life, and the three are scored against what the home actually fetched.
"""
import math

import numpy as np

import hedonic as H

Z80 = 1.2816


def wquant(v, w, q):
    v, w = np.asarray(v, float), np.asarray(w, float)
    if not len(v):
        return float("nan")
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


def holdout(d, feats, zq, w, n_q=8, min_train=150, min_test=3, score_w=None, **fitkw):
    qs = sorted(d.q.unique())
    err, base, wt, sig, nq = [], [], [], [], 0
    for q in qs[-n_q:]:
        tr, te = (d.q < q).to_numpy(), (d.q == q).to_numpy()
        if tr.sum() < min_train or te.sum() < min_test:
            continue
        dtr, dte = d[tr], d[te]
        m = H.fit(dtr, feats, zq[zq.index <= q], w[tr], **fitkw)            # tau for q comes from Zillow's change only
        X = (dte[feats].to_numpy(float) - m["mu"]) / m["sd"]
        pred = X @ m["beta"] + m["tau"][q]
        err += list(np.log(dte.price.to_numpy(float)) - pred)
        yr = dtr[dtr.q >= q - 4]                                    # rule of thumb: last year's median $/sq ft
        ppsf = float(np.median(yr.price / yr.sqft)) if len(yr) else float(np.median(dtr.price / dtr.sqft))
        base += list(np.log(dte.price.to_numpy(float)) - np.log(ppsf * dte.sqft.to_numpy(float)))
        wt += list((w if score_w is None else score_w)[te])
        sig += [m["sigma"]] * int(te.sum())
        nq += 1
    if len(err) < 20:
        return None
    err, base, wt, sig = map(np.array, (err, base, wt, sig))
    a = np.abs(err)
    return {"n": int(len(err)), "quarters": nq,
            "bias": round(wquant(err, wt, 0.5), 4),               # median: one distressed sale shouldn't count as bias
            "mdae": round(wquant(a, wt, 0.5), 4), "mdae_ppsf": round(wquant(np.abs(base), wt, 0.5), 4),
            "q80": round(wquant(a, wt, 0.8), 4),
            "cover80": round(float(np.sum(wt * (a <= Z80 * sig)) / wt.sum()), 3),
            "sigma_oos": round(wquant(a, wt, 0.8) / Z80, 4)}


def repeat_sales(d, keys, sigma, half_life, min_gap_m=12):
    """d needs 'res' (model residual), 'date'. keys: address key per row."""
    t = d.assign(k=keys.values).sort_values("date")
    r1, r2, gap = [], [], []
    for _, g in t.groupby("k"):
        if len(g) < 2:
            continue
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            mths = (g.date[i + 1] - g.date[i]).days / 30.44
            same = not (g.sqft[i] > 0 and g.sqft[i + 1] > 0) or abs(math.log(g.sqft[i + 1] / g.sqft[i])) < 0.03
            if mths >= min_gap_m and same:                          # same size = same unit (condo addresses can lack one)
                r1.append(g.res[i]); r2.append(g.res[i + 1]); gap.append(mths)   # noqa: E702
    if len(r1) < 15:
        return {"n": len(r1)}
    r1, r2, gap = map(np.array, (r1, r2, gap))
    lam = math.log(2) / (half_life * 12)
    rho = np.exp(-lam * gap)
    md = lambda e: round(float(np.median(np.abs(e))), 4)            # noqa: E731
    out = {"n": int(len(r1)), "gap_years_median": round(float(np.median(gap)) / 12, 1),
           "corr": round(float(np.corrcoef(r1, r2)[0, 1]), 3),
           "mdae_model_only": md(r2), "mdae_full_carry": md(r2 - r1), "mdae_fade": md(r2 - rho * r1),
           "cover80_fade": round(float(np.mean(np.abs(r2 - rho * r1) <= Z80 * sigma * np.sqrt(1 - rho ** 2))), 3)}
    # what the pairs themselves say: r2 ~ c * 0.5^(gap/HL) * r1 (c < 1 = part of one sale's premium is noise)
    best = None
    for c in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for hl in (2, 3, 5, 7, 10, 15, 20, 30, 1000):
            sse = float(np.sum(np.clip(np.abs(r2 - c * 0.5 ** (gap / (hl * 12)) * r1), None, 0.5) ** 2))  # winsorised
            if best is None or sse < best[0] - 1e-12:
                best = (sse, c, hl)
    cur = float(np.sum(np.clip(np.abs(r2 - rho * r1), None, 0.5) ** 2))
    c, hl = best[1], best[2]
    out.update(best_c=c, best_half_life=hl, mdae_best=md(r2 - c * 0.5 ** (gap / (hl * 12)) * r1),
               sse_ratio_best_vs_config=round(best[0] / cur, 3))
    return out
