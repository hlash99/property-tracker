"""Local hedonic price model on real sold comps.

    log(price_i) = x_i . beta + tau_q(i) + e_i

x = home attributes (size, lot, beds, baths, age, remodel/fixer/view flags from the
listing remarks, city, position), tau = one effect per calendar quarter. The quarter
effects are a LOCAL sales index: their quarter-to-quarter changes are shrunk toward
Zillow's ZHVI changes for the same area (penalty `lam`), so a quarter with few sales
follows Zillow and a quarter with many sales follows the actual sales. Comps are
weighted by distance and the fit is made robust with a few Huber IRLS passes.
"""
import math
import re

import numpy as np
import pandas as pd

FLAGS = {
    "remodel": r"remodel|renovat|rebuilt|newly (?:updated|built)|brand[- ]new|down to the studs|studs?[- ]up|fully updated|\bgut",
    "fixer": r"fixer|\btlc\b|as[- ]is|contractor|investor|tear[- ]?down|needs work|bring your|developer|potential to",
    "view": r"bay views?|views? of the bay|golden gate|bridge views?|coit|panoramic|city views?|water views?",
    "tic": r"tenancy[- ]in[- ]common|\bTIC\b",
}


def text_flags(text):
    t = "" if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    return {k: int(bool(re.search(p, t, re.I))) for k, p in FLAGS.items()}


def fit(df, feats, zq, w, lam=60.0, ridge=1.0, iters=4):
    """df needs columns feats + 'q' (pd.Period) + 'price'. zq: pd.Series of Zillow by quarter."""
    qs = list(zq.index)
    qi = {q: i for i, q in enumerate(qs)}
    X = df[feats].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    n, k = Xs.shape
    Q = len(qs)
    Qm = np.zeros((n, Q))
    Qm[np.arange(n), [qi[q] for q in df.q]] = 1
    A = np.hstack([Xs, Qm])
    y = np.log(df.price.to_numpy(float))
    rows, tgt = [], []
    dz = np.diff(np.log(zq.to_numpy(float)))
    for j in range(1, Q):                      # random walk around Zillow's changes
        r = np.zeros(k + Q)
        r[k + j], r[k + j - 1] = 1, -1
        rows.append(r * math.sqrt(lam))
        tgt.append(dz[j - 1] * math.sqrt(lam))
    for j in range(k):                         # light ridge on standardized betas
        r = np.zeros(k + Q)
        r[j] = math.sqrt(ridge)
        rows.append(r)
        tgt.append(0.0)
    P, t = np.array(rows), np.array(tgt)
    wr = w.copy()
    for _ in range(iters):
        sw = np.sqrt(wr)
        M = np.vstack([A * sw[:, None], P])
        b = np.concatenate([y * sw, t])
        theta, *_ = np.linalg.lstsq(M, b, rcond=None)
        res = y - A @ theta
        s = 1.4826 * np.median(np.abs(res - np.median(res))) or 0.1
        c = 1.5 * s
        wr = w * np.where(np.abs(res) <= c, 1.0, c / np.maximum(np.abs(res), 1e-9))
    sigma = math.sqrt(float(np.sum(wr * res ** 2) / np.sum(wr)))
    cov = sigma ** 2 * np.linalg.pinv(M.T @ M)
    n_q = pd.Series(Qm.sum(0), index=qs)
    return dict(beta=theta[:k], tau=pd.Series(theta[k:], index=qs), mu=mu, sd=sd, feats=feats,
                sigma=sigma, cov=cov, k=k, n=n, n_q=n_q, res=res, wr=wr)


def xs(m, x):
    return (np.array([float(x[f]) for f in m["feats"]]) - m["mu"]) / m["sd"]


def predict(m, x, q):
    """(log value, standard error of the fitted mean) for attributes x in quarter q."""
    v = xs(m, x)
    row = np.concatenate([v, np.zeros(len(m["tau"]))])
    row[m["k"] + list(m["tau"].index).index(q)] = 1
    return float(v @ m["beta"] + m["tau"][q]), math.sqrt(max(float(row @ m["cov"] @ row), 0.0))


def coef(m, f):
    """Effect of +1 unit of raw feature f on log price."""
    i = m["feats"].index(f)
    return float(m["beta"][i] / m["sd"][i])
