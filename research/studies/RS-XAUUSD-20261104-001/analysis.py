"""RS-XAUUSD-20261104-001 — gold on S&P 500 crash days (pre-registered 2026-11-04, with a design amendment before any outcome was computed).

Reads the frozen inputs listed in reproduction_inputs.json (each verified by sha256), runs the pre-run checks
with the registered thresholds and fixed actions (checks 2a and 2c as amended before any outcome was computed), then computes P1, P2, the secondary family and the pre-declared S-SRC source sensitivity.
Outputs aggregates and date lists only (no per-day price or return values, no per-day group labels).

usage (needs numpy, pandas, scipy; the repo research venv has them):
  ~/.venvs/trading-research/bin/python scripts/research/build_xauusd_equity_crash.py [--prechecks-only] [--dry]
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "RS-XAUUSD-20261104-001"
OUT = Path("reproduced")
OUT.mkdir(parents=True, exist_ok=True)
TAIPEI = timezone(timedelta(hours=8))

SAMPLE_START = "2008-01-02"
SAMPLE_END = "2026-10-30"
CRASH_PRIMARY = -0.02
CRASH_SECONDARY = -0.03
EPS = 1e-12
GAP5, GAP20 = 5, 20
VIX_JUMP = 0.20
BOOT_DRAWS, BOOT_SEED = 10000, 20261104
Z_PRIMARY = 2.2414  # z at 0.9875 (Bonferroni over 2 primaries)
Z_SECONDARY = 1.96
MIN_DAYS, MIN_EPISODES = 20, 10
LOCAL_WINDOW = 60


# ----------------------------------------------------------------------------- inputs

def sha_info(path: Path) -> dict:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def load_manifest() -> dict:
    # Public form: the inputs are the files named in reproduction_inputs.json, staged by the
    # reader under local-inputs/. Each is still verified by sha256 in verified_path().
    m = json.loads((Path(__file__).resolve().parent / "reproduction_inputs.json").read_text())
    return {s["role"]: {"locator": path, "sha256": s["sha256"], "size_bytes": s["size_bytes"]}
            for path, s in m["inputs"].items()}


def verified_path(sources: dict, role: str) -> Path:
    s = sources[role]
    p = ROOT / s["locator"]
    got = sha_info(p)
    if got["sha256"] != s["sha256"] or got["size_bytes"] != s["size_bytes"]:
        raise SystemExit(f"hash mismatch for {role}: {s['locator']} (manifest {s['sha256'][:12]}, file {got['sha256'][:12]})")
    return p


def read_yahoo(path: Path) -> pd.Series:
    d = pd.read_csv(path, usecols=["date", "close"])
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna().drop_duplicates("date", keep="last").set_index("date")["close"].astype(float).sort_index()


def read_gold(path: Path) -> pd.Series:
    d = pd.read_csv(path, usecols=["time", "close"])
    d["date"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None).dt.normalize()
    return d.drop_duplicates("date", keep="last").set_index("date")["close"].astype(float).sort_index()


def read_fred(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"])
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna().set_index("date")["v"].sort_index()


def read_cboe(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["DATE"], format="%m/%d/%Y")
    return d.dropna(subset=["CLOSE"]).drop_duplicates("date", keep="last").set_index("date")["CLOSE"].astype(float).sort_index()


# ----------------------------------------------------------------------------- statistics

def crit(level: str, g: int | None) -> float:
    q = 0.9875 if level == "primary" else 0.975
    z = Z_PRIMARY if level == "primary" else Z_SECONDARY
    if g is not None and g < 30 and g >= 2:
        return float(stats.t.ppf(q, g - 1))
    return z


def pval(stat: float, g: int | None) -> float:
    if g is not None and g < 30 and g >= 2:
        return float(2 * stats.t.sf(abs(stat), g - 1))
    return float(2 * stats.norm.sf(abs(stat)))


def episodes(idx: np.ndarray, gap: int) -> np.ndarray:
    """idx sorted ascending; consecutive crash days share an episode iff idx[j]-idx[i] <= gap."""
    if len(idx) == 0:
        return np.array([], dtype=int)
    new = np.concatenate([[0], (np.diff(idx) > gap).astype(int)])
    return np.cumsum(new)


def cr1_ols(y: np.ndarray, X: np.ndarray, cl: np.ndarray):
    n, k = X.shape
    cls = np.unique(cl)
    g = len(cls)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros((k, k))
    for c in cls:
        m = cl == c
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    c1 = g / (g - 1) * (n - 1) / (n - k) if g > 1 and n > k else float("nan")
    V = c1 * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V)), g


def mean_cr1(y: np.ndarray, cl: np.ndarray) -> dict:
    if len(y) < 2:
        return {"n": int(len(y)), "g": int(len(np.unique(cl))), "mean": float(y.mean()) if len(y) else None, "se": None}
    beta, se, g = cr1_ols(y, np.ones((len(y), 1)), cl)
    return {"n": int(len(y)), "g": int(g), "mean": float(beta[0]), "se": float(se[0])}


def mean_iid(y: np.ndarray) -> dict:
    n = len(y)
    if n < 2:
        return {"n": n, "g": None, "mean": float(y.mean()) if n else None, "se": None}
    return {"n": n, "g": None, "mean": float(y.mean()), "se": float(y.std(ddof=1) / math.sqrt(n))}


def mean_nw(y_full: np.ndarray, mask: np.ndarray, lag: int) -> dict:
    """Newey-West (Bartlett, lag=horizon) standard error of the mean over the days where mask is True,
    on the full session axis (days outside the mask contribute zero moments)."""
    n = int(mask.sum())
    mu = float(y_full[mask].mean())
    gt = np.where(mask, y_full - mu, 0.0)
    gt = np.nan_to_num(gt)
    s = float((gt * gt).sum())
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        s += 2 * w * float((gt[l:] * gt[:-l]).sum())
    var = s / (n * n)
    return {"n": n, "g": None, "mean": mu, "se": math.sqrt(max(var, 0.0)), "nw_lag": lag}


def compare(a: dict, b: dict, level: str, cluster_side_g: int | None, label: str = "") -> dict:
    """difference a - b of two independent estimates; df from the clustered side."""
    if a["se"] is None or b["se"] is None:
        return {"label": label, "judged": False, "reason": "se unavailable"}
    d = a["mean"] - b["mean"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    cv = crit(level, cluster_side_g)
    return {"label": label, "n_a": a["n"], "g_a": a.get("g"), "mean_a": a["mean"], "n_b": b["n"], "g_b": b.get("g"),
            "mean_b": b["mean"], "diff": d, "se": se, "crit": cv, "bound": cv * se, "bound_196": 1.96 * se,
            "resolved": bool(abs(d) > cv * se), "p": pval(d / se, cluster_side_g), "df": (cluster_side_g - 1) if cluster_side_g and cluster_side_g < 30 else None}


def size_ok(n_days: int, n_ep: int) -> bool:
    return n_days >= MIN_DAYS and n_ep >= MIN_EPISODES


def wilson(k: int, n: int):
    if n == 0:
        return None
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [(c - h) * 100, (c + h) * 100]


# ----------------------------------------------------------------------------- build the session table

def build_sessions(spx: pd.Series, gold: pd.Series, vix: pd.Series, dxy: pd.Series):
    full = spx.index
    start = full.searchsorted(pd.Timestamp(SAMPLE_START))
    end = full.searchsorted(pd.Timestamp(SAMPLE_END), side="right")
    cal = full[start:end]
    prev = full[start - 1:end - 1]
    S = pd.DataFrame({"date": cal, "prev_date": prev})
    S["idx"] = np.arange(len(S))
    S["spx_ret"] = spx.reindex(cal).values / spx.reindex(prev).values - 1
    S["gold_c"] = gold.reindex(cal).values
    S["gold_prev"] = gold.reindex(prev).values
    S["H0"] = S["gold_c"] / S["gold_prev"] - 1
    for k in (1, 5, 20):
        S[f"H{k}"] = S["gold_c"].shift(-k) / S["gold_c"] - 1
    S["vix_c"] = vix.reindex(cal).values
    S["vix_prev"] = vix.reindex(prev).values
    S["vix_ret"] = S["vix_c"] / S["vix_prev"] - 1
    S["dxy_c"] = dxy.reindex(cal).values
    S["dxy_prev"] = dxy.reindex(prev).values
    ret = S["dxy_c"] / S["dxy_prev"] - 1
    S["dxy_stale"] = (S["dxy_c"] == S["dxy_prev"]) & S["dxy_c"].notna()
    S["dxy_ret"] = ret.where(~S["dxy_stale"])
    S["crash2"] = S["spx_ret"] <= CRASH_PRIMARY + EPS
    S["crash3"] = S["spx_ret"] <= CRASH_SECONDARY + EPS
    return S


def read_fx(path: Path) -> pd.Series:
    d = pd.read_csv(path, usecols=["time", "close"])
    d["date"] = pd.to_datetime(d["time"])
    return d.dropna().drop_duplicates("date", keep="last").set_index("date")["close"].astype(float).sort_index()


def excluded_sessions(S: pd.DataFrame) -> set:
    """Crash day +/- 1 NYSE session (calendar of S). Used by the amended checks 2a and 2c; no gold value is involved."""
    c = S["crash2"].values
    nxt = np.concatenate([[False], c[:-1]])
    prv = np.concatenate([c[1:], [False]])
    return set(S.loc[c | nxt | prv, "date"])


def sess_close_ret(ser: pd.Series, S: pd.DataFrame):
    """Close on session t and the return between consecutive NYSE sessions (t-1 = previous session of the calendar), indexed by session date.
    Never pct_change after dropping rows (that would span the dropped sessions)."""
    c = pd.Series(ser.reindex(S["date"]).values, index=S["date"])
    p = pd.Series(ser.reindex(S["prev_date"]).values, index=S["date"])
    return c, c / p - 1


def pair_sess(a: pd.Series, b: pd.Series, S: pd.DataFrame, keep: pd.Series) -> dict:
    ca, ra = sess_close_ret(a, S)
    cb, rb = sess_close_ret(b, S)
    ok = ra.notna() & rb.notna() & keep
    okc = ca.notna() & cb.notna() & keep
    return {"n_sessions": int(ok.sum()), "corr_returns": float(np.corrcoef(ra[ok], rb[ok])[0, 1]),
            "share_return_diff_gt_1pp_pct": float((np.abs(ra[ok] - rb[ok]) > 0.01).mean() * 100),
            "median_abs_diff_pct_of_close": float((np.abs(ca[okc] - cb[okc]) / ca[okc]).median() * 100)}


def us_dst(d: pd.Timestamp) -> bool:
    return bool(pd.Timestamp(f"{d:%Y-%m-%d} 12:00", tz="America/New_York").dst())


def lead_coef(spot: pd.Series, gc: pd.Series, S: pd.DataFrame, keep: pd.Series) -> dict:
    """OLS spot r(t) = a + b0 * GC r(t) + b1 * GC r(t+1), consecutive-session returns, by US daylight / standard time."""
    _, rs = sess_close_ret(spot, S)
    _, rg = sess_close_ret(gc, S)
    rg1 = rg.shift(-1)
    dst = pd.Series([us_dst(d) for d in S["date"]], index=S["date"])
    out = {}
    for name, m in (("daylight", dst), ("standard", ~dst)):
        ok = rs.notna() & rg.notna() & rg1.notna() & keep & m
        X = np.column_stack([np.ones(int(ok.sum())), rg[ok].values, rg1[ok].values])
        beta = np.linalg.lstsq(X, rs[ok].values, rcond=None)[0]
        out[name] = {"n_sessions": int(ok.sum()), "b0": float(beta[1]), "b1": float(beta[2])}
    return out


# ----------------------------------------------------------------------------- pre-run checks

def prechecks(sources: dict, S: pd.DataFrame, spx: pd.Series, gold: pd.Series, vix_cboe: pd.Series) -> dict:
    out = {}
    crash_dates = set(S.loc[S["crash2"], "date"])
    prev_of_crash = set(S.loc[S["crash2"], "prev_date"])
    excl = crash_dates | prev_of_crash

    # check 1: gold day boundary
    h1 = pd.read_csv(verified_path(sources, "price_1h_frozen"), usecols=["time", "close"])
    h1["t"] = pd.to_datetime(h1["time"], utc=True).dt.tz_localize(None)
    h1 = h1.set_index("t")["close"].astype(float)
    days = [d for d in gold.index if h1.index.min().normalize() <= d <= h1.index.max().normalize() and d not in excl]
    med = {}
    for hh in (19, 20, 21, 22, 23):
        diffs = []
        for d in days:
            ts = d + pd.Timedelta(hours=hh)
            if ts in h1.index:
                diffs.append(abs(gold[d] - h1[ts]))
        med[f"stamp_{hh}Z"] = {"median_abs_diff_usd": float(np.median(diffs)) if diffs else None, "n_days": len(diffs)}
    best2021 = min((med["stamp_20Z"]["median_abs_diff_usd"], med["stamp_21Z"]["median_abs_diff_usd"]))
    m23 = med["stamp_23Z"]["median_abs_diff_usd"]
    c1_pass = bool(best2021 <= 1.00 and best2021 <= m23 / 3)
    best_hour = min(med, key=lambda k: med[k]["median_abs_diff_usd"])
    out["check1_gold_day_boundary"] = {
        "rule": "better of 20Z/21Z median <= 1.00 USD and <= 1/3 of the 23Z median; days = 1h-overlap days except S&P crash days and the session before",
        "overlap_days_used": len(days), "by_stamp": med, "best_stamp": best_hour, "pass": c1_pass,
        "action": ("none: bar stamped D ends about 1-2 h after the 16:00 ET S&P close (1h stamp convention not verified; before the restatement the daily close equalled the 20Z-stamped 1h close)" if c1_pass else "FAIL: stop (registered fixed action)"),
        "scope_note": "US daylight time only (2026-04-06 to 2026-10-30)."}

    # check 2a (amended 2026-11-04): stability vs post-restatement fetches + adjudication against FX_IDC
    fx = read_fx(verified_path(sources, "price_1d_fxidc_frozen"))
    excl_set = excluded_sessions(S)
    sess = S["date"]
    keep_all = pd.Series(~sess.isin(excl_set).values, index=sess)
    stab = {}
    for c in ("06d4fd0", "811f9af"):
        v = read_gold(verified_path(sources, f"price_1d_version_{c}"))
        sh = v.index.intersection(gold.index)
        nd = int((np.abs(v[sh] - gold[sh]) > 0.01).sum())
        stab[c] = {"shared_dates": int(len(sh)), "dates_differing_gt_0p01": nd, "pass": bool(nd == 0)}
    pre_v = read_gold(verified_path(sources, "price_1d_version_826439b"))
    win = pd.Series(((sess >= pd.Timestamp("2022-08-15")) & (sess <= pd.Timestamp(SAMPLE_END))).values, index=sess) & keep_all
    r_fro = pair_sess(gold, fx, S, win)
    # same-session comparison with the pre-restatement version (its coverage ends 2026-10-09)
    _, rp = sess_close_ret(pre_v, S)
    common = win & rp.notna()
    r_fro_c = pair_sess(gold, fx, S, common)
    r_pre_c = pair_sess(pre_v, fx, S, common)
    adj_pass = bool(r_fro["corr_returns"] >= 0.99 and r_fro["median_abs_diff_pct_of_close"] <= 0.10
                    and r_fro_c["share_return_diff_gt_1pp_pct"] <= r_pre_c["share_return_diff_gt_1pp_pct"])
    # disclosed counts (not gates)
    snap = read_gold(verified_path(sources, "price_1d_snapshot_crosscheck"))
    sh = snap.index.intersection(gold.index)
    same = (np.abs(snap[sh] - gold[sh]) <= 0.01)
    a_ = pd.DataFrame({"s": snap[sh], "g": gold[sh]})
    legacy_shift_gt_0p10 = int(((a_["g"].pct_change() - a_["s"].pct_change()).abs() * 100 > 0.10).sum())
    _, rsn = sess_close_ret(snap, S)
    _, rfr = sess_close_ret(gold, S)
    shift = ((rsn - rfr).abs() * 100 > 0.10) & rsn.notna() & rfr.notna()
    crash_sess = pd.Series(S["crash2"].values, index=sess)
    dif = (np.abs(pre_v.reindex(S["date"]).values - gold.reindex(S["date"]).values) > 0.01)
    dif_prev = (np.abs(pre_v.reindex(S["prev_date"]).values - gold.reindex(S["prev_date"]).values) > 0.01)
    endpoint = pd.Series((dif | dif_prev), index=sess) & crash_sess
    by_year = {str(y): int(endpoint[endpoint.index.year == y].sum()) for y in sorted(set(endpoint[endpoint].index.year))}
    snap_last = snap.index.max()
    rev = {
        "amended": "threshold amended 2026-11-04, before any outcome was computed",
        "stability_vs_post_restatement_fetches": stab,
        "adjudication_vs_fxidc": {
            "sessions": "NYSE sessions 2022-08-15 to 2026-10-30, crash day +/- 1 session excluded; consecutive-session returns",
            "frozen_vs_fxidc": r_fro,
            "same_sessions_as_pre_restatement_version": {"n_sessions": r_fro_c["n_sessions"], "frozen_share_gt_1pp_pct": r_fro_c["share_return_diff_gt_1pp_pct"],
                                                         "pre_restatement_826439b_share_gt_1pp_pct": r_pre_c["share_return_diff_gt_1pp_pct"],
                                                         "pre_restatement_corr_returns": r_pre_c["corr_returns"]}},
        "disclosed_not_gates": {
            "vs_2026_08_24_snapshot": {"shared_dates": int(len(sh)), "revised_dates_gt_0p01": int((~same).sum()),
                                       "returns_shifted_gt_0p10pp_shared_rows": legacy_shift_gt_0p10,
                                       "returns_shifted_gt_0p10pp_sessions_not_excluded": int((shift & keep_all).sum()),
                                       "returns_shifted_gt_0p10pp_sessions_crash_pm1": int((shift & ~keep_all).sum()),
                                       "snapshot_last_bar": snap_last.strftime("%Y-%m-%d"), "snapshot_last_bar_incomplete": True,
                                       "note": "snapshot taken 2026-08-24 15:00Z; its last bar was not closed"},
            "crash_days_with_revised_H0_endpoint_vs_826439b": {"count": int(endpoint.sum()), "by_year": by_year, "excluded_from_sample": False}},
    }
    rev["pass"] = bool(all(v["pass"] for v in stab.values()) and adj_pass)
    rev["action"] = ("none: frozen file equals the post-restatement fetches and agrees with FX_IDC" if rev["pass"]
                     else "FAIL: stop; no further amendment of 2a")
    out["check2a_stability_and_fxidc_adjudication"] = rev

    # check 2b: FX_IDC
    # (fx was read above from the frozen byte copy of the FX_IDC export)

    def pair_stats(other: pd.Series, dates: pd.DatetimeIndex, offset: int = 0):
        sess = dates
        a = pd.Series(gold.reindex(sess).values, index=sess)
        o = other.copy()
        if offset:
            o.index = o.index + pd.Timedelta(days=offset)
        b = pd.Series(o.reindex(sess).values, index=sess)
        ok = a.notna() & b.notna()
        ra, rb = a[ok].pct_change(), b[ok].pct_change()
        r = pd.concat([ra, rb], axis=1).dropna()
        corr = float(np.corrcoef(r.iloc[:, 0], r.iloc[:, 1])[0, 1])
        rel = float((np.abs(a[ok] - b[ok]) / a[ok]).median() * 100)
        big = float((np.abs(r.iloc[:, 0] - r.iloc[:, 1]) > 0.01).mean() * 100)
        return {"n_sessions": int(ok.sum()), "corr_returns": corr, "median_abs_diff_pct_of_close": rel, "share_return_diff_gt_1pp_pct": big}

    d2b = S.loc[S["date"] >= pd.Timestamp("2013-07-16"), "date"]
    r2b = pair_stats(fx, pd.DatetimeIndex(d2b))
    r2b["pass"] = bool(r2b["corr_returns"] >= 0.99 and r2b["median_abs_diff_pct_of_close"] <= 0.10)
    r2b["action"] = "none" if r2b["pass"] else "test +/-1-day offset"
    if not r2b["pass"]:
        r2b["offset_tests"] = {str(o): pair_stats(fx, pd.DatetimeIndex(d2b), o) for o in (-1, 1)}
    out["check2b_fxidc_2013_07_16_on"] = r2b

    # check 2c (amended 2026-11-04): relative check, Twelve Data vs GC=F (2008-2013) against the FX_IDC vs GC=F reference pair (2013-07-16 on)
    gc = read_yahoo(verified_path(sources, "price_1d_independent_crosscheck_2008_2013"))
    k_ref = pd.Series(((sess >= pd.Timestamp("2013-07-16")) & (sess <= pd.Timestamp(SAMPLE_END))).values, index=sess) & keep_all
    k_tst = pd.Series(((sess >= pd.Timestamp(SAMPLE_START)) & (sess <= pd.Timestamp("2013-07-15"))).values, index=sess) & keep_all
    ref = pair_sess(fx, gc, S, k_ref)
    tst = pair_sess(gold, gc, S, k_tst)
    ref["lead"] = lead_coef(fx, gc, S, k_ref)
    tst["lead"] = lead_coef(gold, gc, S, k_tst)
    cond_a = bool(tst["corr_returns"] >= ref["corr_returns"] - 0.03)
    cond_b = bool(tst["share_return_diff_gt_1pp_pct"] <= ref["share_return_diff_gt_1pp_pct"] + 1.0)
    cond_c = {k: bool(abs(tst["lead"][k]["b1"] - ref["lead"][k]["b1"]) <= 0.05) for k in ("daylight", "standard")}
    # reference-pair context (not gates): Twelve Data vs FX_IDC on the reference sessions
    r2c = {"source": "Yahoo GC=F (registered fallback; Stooq returned a JavaScript bot-challenge page on 2026-11-04)",
           "sessions": "NYSE sessions, crash day +/- 1 session excluded; consecutive-session returns",
           "reference_pair_fxidc_vs_gc_2013_07_16_on": ref, "test_pair_twelvedata_vs_gc_2008_01_02_to_2013_07_15": tst,
           "context_twelvedata_vs_fxidc_reference_sessions": pair_sess(gold, fx, S, k_ref),
           "conditions": {"a_corr_ge_ref_minus_0p03": cond_a, "b_share_le_ref_plus_1pp": cond_b, "c_lead_b1_within_0p05": cond_c}}
    r2c["pass"] = bool(cond_a and cond_b and all(cond_c.values()))
    r2c["action"] = ("none" if r2c["pass"] else "FAIL: stop; no further amendment of 2c")
    out["check2c_relative_2008_2013"] = r2c

    # check 3: VIX Cboe vs FRED VIXCLS
    fred_vix = read_fred(verified_path(sources, "vix_1d_crosscheck"))
    sh = vix_cboe.index.intersection(fred_vix.index)
    df = np.abs(vix_cboe[sh] - fred_vix[sh])
    miss = float(S["date"].map(lambda d: d not in vix_cboe.index).mean() * 100)
    c3 = {"shared_dates": int(len(sh)), "share_within_0p01": float((df <= 0.01).mean()), "share_gt_0p05": float((df > 0.05).mean()),
          "cboe_missing_sample_sessions_pct": miss}
    c3["pass"] = bool(c3["share_within_0p01"] >= 0.995 and miss <= 0.5)
    if c3["pass"]:
        c3["action"] = "none: Cboe used"
    elif c3["share_gt_0p05"] <= 0.005:
        c3["action"] = "coverage failure only: FRED VIXCLS refresh is used as the VIX series"
    else:
        c3["action"] = "STOP (value failure)"
    out["check3_vix"] = c3

    # check 4: S&P 500
    old = read_yahoo(ROOT / sources["spx_1d_existing"]["locator"])
    verified_path(sources, "spx_1d_existing")
    sh = old.index.intersection(spx.index)
    ident = float((np.abs(old[sh] - spx[sh]) <= 0.01).mean())
    old_ret = old.pct_change()
    new_ret = spx.pct_change().reindex(old.index)
    win = (old.index >= pd.Timestamp(SAMPLE_START)) & (old.index <= pd.Timestamp(SAMPLE_END))
    same_list = bool(((old_ret[win] <= CRASH_PRIMARY + EPS) == (new_ret[win] <= CRASH_PRIMARY + EPS)).all())
    c4 = {"shared_dates": int(len(sh)), "identical_within_0p01_share": ident, "same_crash_list_on_overlap": same_list}
    fred = read_fred(verified_path(sources, "spx_1d_crosscheck"))
    fr = fred.pct_change()
    ys = spx.pct_change().reindex(fr.index)
    cmp_ok = fr.notna() & ys.notna() & (fr.index <= pd.Timestamp(SAMPLE_END))
    ndiff = int(((fr[cmp_ok] <= CRASH_PRIMARY + EPS) != (ys[cmp_ok] <= CRASH_PRIMARY + EPS)).sum())
    c4["fred_sp500_dates_compared"] = int(cmp_ok.sum())
    c4["fred_classification_differences"] = ndiff
    c4["pass"] = bool(ident >= 0.999 and same_list and ndiff <= 2)
    c4["action"] = "none" if c4["pass"] else "list dates; stop if the crash list differs"
    out["check4_spx"] = c4

    # check 5: DXY coverage
    miss_mask = S["dxy_ret"].isna()
    c5 = {"missing_share_sample_sessions_pct": float(miss_mask.mean() * 100), "stale_sessions": int(S["dxy_stale"].sum()),
          "crash_days_missing_dxy": int((miss_mask & S["crash2"]).sum()), "crash_days": int(S["crash2"].sum())}
    c5["pass"] = bool(c5["missing_share_sample_sessions_pct"] <= 3.0 and c5["crash_days_missing_dxy"] <= 5)
    c5["action"] = "none" if c5["pass"] else "S-USD1, S-USD2 and the DXY-only split reported descriptive only"
    out["check5_dxy_coverage"] = c5
    return out


# ----------------------------------------------------------------------------- bootstrap

def boot_ratio_diff(ep_sum_a, ep_n_a, ep_sum_b, ep_n_b, base: np.ndarray | None, seed=BOOT_SEED, draws=BOOT_DRAWS):
    """Episode bootstrap of (mean_a - mean_b). Group b is either another group over the same episodes
    (ep_sum_b/ep_n_b arrays of the same length: kept together) or, if base is given, a fixed baseline sample
    resampled iid. Returns draw array (NaN where a group is empty)."""
    rng = np.random.default_rng(seed)
    G = len(ep_sum_a)
    idx = rng.integers(0, G, size=(draws, G))
    na = ep_n_a[idx].sum(axis=1)
    ma = np.where(na > 0, ep_sum_a[idx].sum(axis=1) / np.maximum(na, 1), np.nan)
    if base is None:
        nb = ep_n_b[idx].sum(axis=1)
        mb = np.where(nb > 0, ep_sum_b[idx].sum(axis=1) / np.maximum(nb, 1), np.nan)
    else:
        mb = np.empty(draws)
        nbase = len(base)
        for s in range(0, draws, 500):
            e = min(draws, s + 500)
            mb[s:e] = base[rng.integers(0, nbase, size=(e - s, nbase))].mean(axis=1)
    return ma - mb


def pct_interval(d: np.ndarray, lo: float, hi: float):
    d = d[~np.isnan(d)]
    return [float(np.percentile(d, lo)), float(np.percentile(d, hi))], int(len(d))


def boot_p(d: np.ndarray) -> float:
    d = d[~np.isnan(d)]
    n = len(d)
    p = 2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1))
    return float(min(p, 1.0))


# ----------------------------------------------------------------------------- analysis

def fmt_dates(S: pd.DataFrame, mask) -> list:
    return [d.strftime("%Y-%m-%d") for d in S.loc[mask, "date"]]


def episode_table(S: pd.DataFrame, mask_col: str, gap: int):
    idx = S.loc[S[mask_col], "idx"].values
    ep = episodes(idx, gap)
    dates = S.loc[S[mask_col], "date"].values
    rows = []
    for e in np.unique(ep):
        m = ep == e
        rows.append({"episode": int(e), "first": str(dates[m][0])[:10], "last": str(dates[m][-1])[:10], "days": int(m.sum())})
    return ep, rows


def run_analysis(S: pd.DataFrame, flags: dict) -> dict:
    res = {}
    N = len(S)
    crash = S["crash2"].values
    crash3 = S["crash3"].values
    cidx = S.loc[S["crash2"], "idx"].values
    ep5_c = episodes(cidx, GAP5)
    ep20_c = episodes(cidx, GAP20)
    S["ep5"] = -1
    S.loc[S["crash2"], "ep5"] = ep5_c
    S["ep20"] = -1
    S.loc[S["crash2"], "ep20"] = ep20_c
    cidx3 = S.loc[S["crash3"], "idx"].values
    S["ep5_3"] = -1
    S.loc[S["crash3"], "ep5_3"] = episodes(cidx3, GAP5)
    S["ep20_3"] = -1
    S.loc[S["crash3"], "ep20_3"] = episodes(cidx3, GAP20)

    ep5_tab = episode_table(S, "crash2", GAP5)[1]
    ep20_tab = episode_table(S, "crash2", GAP20)[1]

    # ---- coverage / counts
    res["counts"] = {
        "sessions": int(N), "first_session": S["date"].iloc[0].strftime("%Y-%m-%d"), "last_session": S["date"].iloc[-1].strftime("%Y-%m-%d"),
        "crash_days_le_minus2": int(crash.sum()), "episodes_5_session": len(ep5_tab), "episodes_20_session": len(ep20_tab),
        "single_day_episodes_5_session": int(sum(1 for e in ep5_tab if e["days"] == 1)),
        "crash_days_le_minus3": int(crash3.sum()),
        "episodes_5_session_le_minus3": int(S.loc[S["crash3"], "ep5_3"].nunique()),
        "episodes_20_session_le_minus3": int(S.loc[S["crash3"], "ep20_3"].nunique()),
        "crash_days_before_2013_07_16": int((crash & (S["date"] < pd.Timestamp("2013-07-16"))).sum()),
        "crash_days_gold_H0_missing": int((crash & S["H0"].isna()).sum()),
        "crash_days_vix_missing": int((crash & S["vix_ret"].isna()).sum()),
        "crash_days_dxy_missing": int((crash & S["dxy_ret"].isna()).sum()),
        "gold_H0_missing_sessions": int(S["H0"].isna().sum()),
        "gold_H1_missing_or_beyond_end_sessions": int(S["H1"].isna().sum()),
    }

    base_mask = (~S["crash2"]) & S["H0"].notna()
    base0 = S.loc[base_mask, "H0"].values * 100
    cr = S[S["crash2"] & S["H0"].notna()]
    y_c = cr["H0"].values * 100
    cl_c = cr["ep5"].values
    allmean = S["H0"].dropna().mean() * 100

    # ---- C1 baseline description
    base_stat = mean_iid(base0)
    crash_stat = mean_cr1(y_c, cl_c)
    res["baseline_C1"] = {"n_days": int(len(base0)), "mean_H0_pct": base_stat["mean"], "sd_pct": float(base0.std(ddof=1)),
                          "down_share_pct": float((base0 < 0).mean() * 100), "up_share_pct": float((base0 > 0).mean() * 100)}
    res["all_days_descriptive"] = {"n_days": int(S["H0"].notna().sum()), "mean_H0_pct": float(allmean)}
    res["crash_days_descriptive"] = {"n_days": int(len(y_c)), "episodes": int(crash_stat["g"]), "mean_H0_pct": crash_stat["mean"],
                                     "median_H0_pct": float(np.median(y_c)), "sd_pct": float(y_c.std(ddof=1)),
                                     "gold_up_days": int((y_c > 0).sum()), "gold_down_days": int((y_c < 0).sum()),
                                     "gold_up_share_pct": float((y_c > 0).mean() * 100)}

    # ---- P1
    ok1 = size_ok(len(y_c), crash_stat["g"])
    p1 = compare(crash_stat, base_stat, "primary", crash_stat["g"], "P1 crash-day H0 minus C1 H0 (percentage points)")
    p1["judged"] = ok1
    p1["verdict"] = ("分得出" if p1["resolved"] else "分不出") if ok1 else "樣本不足，不判定"
    # bootstrap
    ueps = np.unique(cl_c)
    ep_sum = np.array([y_c[cl_c == e].sum() for e in ueps])
    ep_n = np.array([(cl_c == e).sum() for e in ueps], dtype=float)
    d = boot_ratio_diff(ep_sum, ep_n, None, None, base0)
    p1["bootstrap_98p75"], p1["bootstrap_valid_draws"] = pct_interval(d, 1.25, 98.75)
    p1["bootstrap_95"], _ = pct_interval(d, 2.5, 97.5)
    p1["bootstrap_excludes_zero_primary"] = bool(not (p1["bootstrap_98p75"][0] <= 0 <= p1["bootstrap_98p75"][1]))
    # episode-equal-weight sensitivity
    epm = ep_sum / ep_n
    ew_se = math.sqrt(epm.var(ddof=1) / len(epm) + base_stat["se"] ** 2)
    p1["episode_equal_weight"] = {"episodes": int(len(epm)), "mean_of_episode_means": float(epm.mean()), "diff": float(epm.mean() - base_stat["mean"]),
                                  "se": ew_se, "bound": crit("primary", len(epm)) * ew_se,
                                  "resolved": bool(abs(epm.mean() - base_stat["mean"]) > crit("primary", len(epm)) * ew_se)}
    # leave-one-episode-out
    flips, loo_rows = [], []
    for e in ueps:
        m = cl_c != e
        cs = mean_cr1(y_c[m], cl_c[m])
        t = compare(cs, base_stat, "primary", cs["g"])
        loo_rows.append(bool(t["resolved"]))
        if bool(t["resolved"]) != bool(p1["resolved"]):
            tab = [r for r in ep5_tab if r["episode"] == int(np.unique(S.loc[(S["crash2"]) & S["H0"].notna() & (S["ep5"] == e), "ep5"])[0])][0]
            flips.append({"first": tab["first"], "last": tab["last"], "days": tab["days"], "diff_without": t["diff"], "bound_without": t["bound"]})
    p1["fragility"] = {"removals": int(len(ueps)), "verdict_flips": len(flips), "flipping_episodes": flips}
    res["P1"] = p1

    # ---- P2
    vj = (S["vix_ret"] >= VIX_JUMP - EPS)
    c2d = S[S["crash2"] & S["H0"].notna() & S["vix_ret"].notna()].copy()
    c2d["vj"] = (c2d["vix_ret"] >= VIX_JUMP - EPS).astype(float)
    res["P2_counts"] = {
        "crash_days_in_regression": int(len(c2d)), "dropped_gold_missing": int((S["crash2"] & S["H0"].isna()).sum()),
        "dropped_vix_missing": int((S["crash2"] & S["H0"].notna() & S["vix_ret"].isna()).sum()),
        "vix_jump_days": int(c2d["vj"].sum()), "vix_jump_episodes": int(c2d.loc[c2d["vj"] == 1, "ep5"].nunique()),
        "other_days": int((c2d["vj"] == 0).sum()), "other_episodes": int(c2d.loc[c2d["vj"] == 0, "ep5"].nunique()),
        "episodes_in_regression": int(c2d["ep5"].nunique())}
    pc = res["P2_counts"]
    ok2 = size_ok(pc["vix_jump_days"], pc["vix_jump_episodes"]) and size_ok(pc["other_days"], pc["other_episodes"])
    p2 = reg_dummy(c2d, "vj", "primary")
    p2["label"] = "P2 mean H0 of VIX-jump crash days minus other crash days (percentage points)"
    p2["judged"] = ok2
    p2["verdict"] = (("分得出" if p2["resolved"] else "分不出") if ok2 else "樣本不足，不判定")
    p2["group_size"] = {"vix_jump": [pc["vix_jump_days"], pc["vix_jump_episodes"]], "other": [pc["other_days"], pc["other_episodes"]], "min_days": MIN_DAYS, "min_episodes": MIN_EPISODES}
    # bootstrap
    ue = np.unique(c2d["ep5"].values)
    yy = c2d["H0"].values * 100
    vv = c2d["vj"].values
    ee = c2d["ep5"].values
    sa = np.array([yy[(ee == e) & (vv == 1)].sum() for e in ue])
    na = np.array([((ee == e) & (vv == 1)).sum() for e in ue], dtype=float)
    sb = np.array([yy[(ee == e) & (vv == 0)].sum() for e in ue])
    nb = np.array([((ee == e) & (vv == 0)).sum() for e in ue], dtype=float)
    d = boot_ratio_diff(sa, na, sb, nb, None)
    p2["bootstrap_98p75"], p2["bootstrap_valid_draws"] = pct_interval(d, 1.25, 98.75)
    p2["bootstrap_95"], _ = pct_interval(d, 2.5, 97.5)
    p2["bootstrap_excludes_zero_primary"] = bool(not (p2["bootstrap_98p75"][0] <= 0 <= p2["bootstrap_98p75"][1]))
    # equal-weight sensitivity
    ma = (sa / np.where(na > 0, na, np.nan))
    mb = (sb / np.where(nb > 0, nb, np.nan))
    ma, mb = ma[~np.isnan(ma)], mb[~np.isnan(mb)]
    if len(ma) > 1 and len(mb) > 1:
        sec = math.sqrt(ma.var(ddof=1) / len(ma) + mb.var(ddof=1) / len(mb))
        dd = ma.mean() - mb.mean()
        gmin = min(len(ma), len(mb))
        p2["episode_equal_weight"] = {"episodes_vix_jump": int(len(ma)), "episodes_other": int(len(mb)), "diff": float(dd), "se": sec,
                                      "bound": crit("primary", gmin) * sec, "resolved": bool(abs(dd) > crit("primary", gmin) * sec),
                                      "note": "SE ignores the covariance of groups sharing an episode; sensitivity only"}
    # leave-one-episode-out
    flips = []
    for e in ue:
        m = (c2d["ep5"].values != e)
        t = reg_dummy(c2d[m], "vj", "primary")
        if t.get("resolved") is None or not t.get("judged", True):
            continue
        if bool(t["resolved"]) != bool(p2["resolved"]):
            tab = [r for r in ep5_tab if r["episode"] == int(e)][0]
            flips.append({"first": tab["first"], "last": tab["last"], "days": tab["days"], "diff_without": t["diff"], "bound_without": t["bound"]})
    p2["fragility"] = {"removals": int(len(ue)), "verdict_flips": len(flips), "flipping_episodes": flips}
    p2["dxy_up_share_by_group"] = {}
    for name, gm in (("vix_jump", c2d["vj"] == 1), ("other", c2d["vj"] == 0)):
        sub = c2d[gm & c2d["dxy_ret"].notna()]
        p2["dxy_up_share_by_group"][name] = {"n_with_dxy": int(len(sub)), "dxy_up": int((sub["dxy_ret"] > 0).sum()),
                                             "share_pct": float((sub["dxy_ret"] > 0).mean() * 100) if len(sub) else None}
    res["P2"] = p2

    # ---- secondary family
    sec = []

    def add(name, test, note=""):
        t = dict(test)
        t["name"] = name
        t["note"] = note
        sec.append(t)

    # H1 (5-episode clustering), H5/H20 (20-episode first days)
    def horizon_test(hk, mask_col, ep5col, ep20col, label):
        b = S[(~S["crash2"]) & S[hk].notna()]
        bmask = ((~S["crash2"]) & S[hk].notna()).values
        yfull = np.nan_to_num(S[hk].values * 100)
        if hk == "H1":
            c = S[S[mask_col] & S[hk].notna()]
            cs = mean_cr1(c[hk].values * 100, c[ep5col].values)
            bs = mean_iid(b[hk].values * 100)
            ok = size_ok(cs["n"], cs["g"])
            t = compare(cs, bs, "secondary", cs["g"], label)
            t["groups"] = {"crash_days": cs["n"], "episodes": cs["g"], "baseline_days": bs["n"]}
        else:
            lag = int(hk[1:])
            first = S[S[mask_col] & (S.groupby(ep20col)["idx"].transform("min") == S["idx"])]
            first = first[first[hk].notna()]
            cs = mean_iid(first[hk].values * 100)
            cs["g"] = cs["n"]
            bs = mean_nw(yfull, bmask, lag)
            ok = size_ok(cs["n"], cs["n"])
            t = compare(cs, bs, "secondary", cs["g"], label)
            t["groups"] = {"crash_first_days": cs["n"], "episodes_20_session": cs["n"], "baseline_days": bs["n"], "baseline_se": f"Newey-West lag {lag}"}
        t["judged"] = bool(ok and "diff" in t)
        t["group_size_ok"] = bool(ok)
        return t

    add("H1 crash(<=-2%) vs non-crash", horizon_test("H1", "crash2", "ep5", "ep20", "H1 close(t)->close(t+1), crash days vs C1"))
    add("H5 crash(<=-2%) vs non-crash", horizon_test("H5", "crash2", "ep5", "ep20", "H5 first day of each 20-session episode vs C1"))
    add("H20 crash(<=-2%) vs non-crash", horizon_test("H20", "crash2", "ep5", "ep20", "H20 first day of each 20-session episode vs C1"))

    # -3% threshold
    for hk in ("H0", "H1", "H5", "H20"):
        if hk == "H0":
            c = S[S["crash3"] & S["H0"].notna()]
            cs = mean_cr1(c["H0"].values * 100, c["ep5_3"].values)
            t = compare(cs, base_stat, "secondary", cs["g"], "H0, <=-3% crash days vs C1")
            t["groups"] = {"crash_days": cs["n"], "episodes": cs["g"], "baseline_days": base_stat["n"]}
            t["group_size_ok"] = bool(size_ok(cs["n"], cs["g"]))
            t["judged"] = t["group_size_ok"]
        else:
            t = horizon_test(hk, "crash3", "ep5_3", "ep20_3", f"{hk}, <=-3% days vs C1")
        add(f"{hk} crash(<=-3%) vs non-crash", t)

    # C2 local baseline
    near = np.zeros(N, dtype=bool)
    for i in cidx:
        near[max(0, i - LOCAL_WINDOW): i + LOCAL_WINDOW + 1] = True
    c2mask = near & base_mask.values
    c2s = mean_iid(S.loc[c2mask, "H0"].values * 100)
    t = compare(crash_stat, c2s, "secondary", crash_stat["g"], "H0 crash days vs local non-crash days (+-60 sessions, each day once)")
    t["groups"] = {"crash_days": crash_stat["n"], "episodes": crash_stat["g"], "baseline_days": c2s["n"]}
    t["group_size_ok"] = True
    t["judged"] = True
    add("C2 local baseline H0", t)

    # C3 VIX-stratified
    t = c3_test(S, base_mask.values)
    add("C3 VIX-stratified H0", t)

    # down-day share
    k = int((y_c < 0).sum())
    nshare = len(y_c)
    base_share = float((base0 < 0).mean())
    sa_ = np.array([(y_c[cl_c == e] < 0).sum() for e in ueps], dtype=float)
    d = boot_ratio_diff(sa_, ep_n, None, None, (base0 < 0).astype(float))
    ci, nvalid = pct_interval(d + base_share, 2.5, 97.5)  # crash-share interval = diff + baseline point share
    # crash-share bootstrap interval proper (baseline held at its point for the registered verdict)
    ci_crash = [ci[0] * 100, ci[1] * 100]
    dpct = d * 100
    t = {"label": "gold down-day share (H0<0), crash days vs C1", "n_crash": nshare, "episodes": int(len(ueps)), "crash_share_pct": k / nshare * 100,
         "baseline_share_pct": base_share * 100, "baseline_n": int(len(base0)), "wilson_95_pct_reference_only": wilson(k, nshare),
         "bootstrap_crash_share_95_pct": ci_crash, "diff_pp": k / nshare * 100 - base_share * 100,
         "resolved": bool(not (ci_crash[0] <= base_share * 100 <= ci_crash[1])), "p": boot_p(d), "judged": bool(size_ok(nshare, len(ueps))), "group_size_ok": bool(size_ok(nshare, len(ueps)))}
    add("down-day share", t, "verdict: episode-bootstrap 95% interval of the crash-day share excludes the C1 share; Wilson assumes independent days")

    # subperiods
    for name, a, b in (("2008-2012", "2008-01-01", "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", "2026-12-31")):
        pm = (S["date"] >= pd.Timestamp(a)) & (S["date"] <= pd.Timestamp(b))
        c = S[pm & S["crash2"] & S["H0"].notna()]
        bb = S[pm & base_mask]
        if len(c) >= 2:
            cs = mean_cr1(c["H0"].values * 100, c["ep5"].values)
            bs = mean_iid(bb["H0"].values * 100)
            t = compare(cs, bs, "secondary", cs["g"], f"H0 crash vs non-crash, {name}")
            ok = size_ok(cs["n"], cs["g"])
            t["groups"] = {"crash_days": cs["n"], "episodes": cs["g"], "baseline_days": bs["n"]}
            t["group_size_ok"] = bool(ok)
            t["judged"] = bool(ok and "diff" in t)
        else:
            t = {"label": f"H0 crash vs non-crash, {name}", "judged": False, "group_size_ok": False, "groups": {"crash_days": int(len(c))}}
        add(f"subperiod {name}", t)

    # DXY-only split
    dd = S[S["crash2"] & S["H0"].notna() & S["dxy_ret"].notna()].copy()
    dd["x"] = (dd["dxy_ret"] > 0).astype(float)
    if flags["dxy_ok"]:
        t = reg_dummy(dd, "x", "secondary")
        t["label"] = "crash days: DXY up minus DXY down, mean H0 (pp)"
    else:
        t = {"label": "DXY-only split", "judged": False, "reason": "check 5 failed: descriptive only"}
    t["groups"] = {"dxy_up": [int(dd["x"].sum()), int(dd.loc[dd["x"] == 1, "ep5"].nunique())], "dxy_down": [int((dd["x"] == 0).sum()), int(dd.loc[dd["x"] == 0, "ep5"].nunique())]}
    add("DXY-only split", t)

    # VIX level split (prior-day VIX >= 30)
    vl = S[S["crash2"] & S["H0"].notna() & S["vix_prev"].notna()].copy()
    vl["x"] = (vl["vix_prev"] >= 30).astype(float)
    t = reg_dummy(vl, "x", "secondary")
    t["label"] = "crash days: prior-day VIX >= 30 minus < 30, mean H0 (pp)"
    t["groups"] = {"vix_prev_ge30": [int(vl["x"].sum()), int(vl.loc[vl["x"] == 1, "ep5"].nunique())], "vix_prev_lt30": [int((vl["x"] == 0).sum()), int(vl.loc[vl["x"] == 0, "ep5"].nunique())]}
    add("VIX-level split", t)

    # S-USD1 and S-USD2
    ud = S[S["crash2"] & S["H0"].notna() & S["vix_ret"].notna() & S["dxy_ret"].notna()].copy()
    ud["x"] = ((ud["dxy_ret"] > 0) & (ud["vix_ret"] >= VIX_JUMP - EPS)).astype(float)
    # dollar baseline on non-crash C1 days
    bd = S[base_mask & S["dxy_ret"].notna() & (S["dxy_ret"] != 0)]
    up, dn = bd[bd["dxy_ret"] > 0]["H0"].values * 100, bd[bd["dxy_ret"] < 0]["H0"].values * 100
    dbase = {"dxy_up_days": int(len(up)), "dxy_down_days": int(len(dn)), "mean_up_pct": float(up.mean()), "mean_down_pct": float(dn.mean()),
             "diff": float(up.mean() - dn.mean()), "se": math.sqrt(up.var(ddof=1) / len(up) + dn.var(ddof=1) / len(dn))}
    dbase["bound_196"] = 1.96 * dbase["se"]
    res["dollar_baseline_non_crash"] = dbase
    if flags["dxy_ok"]:
        t = reg_dummy(ud, "x", "secondary")
        t["label"] = "S-USD1 usd_up_vix_jump minus other crash days, mean H0 (pp)"
        if t.get("diff") is not None:
            net = t["diff"] - dbase["diff"]
            nse = math.sqrt(t["se"] ** 2 + dbase["se"] ** 2)
            t["net_of_dollar_baseline"] = {"net_diff": net, "se": nse, "bound_196_or_t": crit("secondary", t.get("g_a_total")) * nse,
                                           "clears": bool(abs(net) > crit("secondary", t.get("g_a_total")) * nse)}
        beta, se, g = cr1_ols(ud["H0"].values * 100, np.column_stack([np.ones(len(ud)), ud["x"].values, ud["dxy_ret"].values * 100]), ud["ep5"].values)
        t2 = {"label": "S-USD2 gold_H0 = a + b*usd_up_vix_jump + c*DXY_return(%), CR1 by episode", "n": int(len(ud)), "g": int(g),
              "b": float(beta[1]), "se_b": float(se[1]), "c": float(beta[2]), "se_c": float(se[2])}
        cv = crit("secondary", int(g))
        t2["bound_b"], t2["bound_c"] = cv * float(se[1]), cv * float(se[2])
        t2["resolved_b"] = bool(abs(beta[1]) > cv * se[1])
        t2["resolved_c"] = bool(abs(beta[2]) > cv * se[2])
        t2["p_b"], t2["p_c"] = pval(beta[1] / se[1], int(g)), pval(beta[2] / se[2], int(g))
    else:
        t = {"label": "S-USD1", "judged": False, "reason": "check 5 failed: descriptive only"}
        t2 = {"label": "S-USD2", "judged": False}
    t["groups"] = {"usd_up_vix_jump": [int(ud["x"].sum()), int(ud.loc[ud["x"] == 1, "ep5"].nunique())], "other": [int((ud["x"] == 0).sum()), int(ud.loc[ud["x"] == 0, "ep5"].nunique())]}
    add("S-USD1", t)
    if flags["dxy_ok"]:
        s2 = [dict(name="S-USD2 b (usd_up_vix_jump)", label=t2["label"], judged=bool(size_ok(int(ud["x"].sum()), int(ud.loc[ud["x"] == 1, "ep5"].nunique()))),
                   n=t2["n"], g=t2["g"], coef=t2["b"], se=t2["se_b"], bound=t2["bound_b"], resolved=t2["resolved_b"], p=t2["p_b"], note=""),
              dict(name="S-USD2 c (DXY return, per 1 pp)", label=t2["label"], judged=True, n=t2["n"], g=t2["g"], coef=t2["c"], se=t2["se_c"],
                   bound=t2["bound_c"], resolved=t2["resolved_c"], p=t2["p_c"], note="")]
        sec.extend(s2)

    # BH q=0.10 over judged tests
    judged = [t for t in sec if t.get("judged") and t.get("p") is not None]
    order = sorted(range(len(judged)), key=lambda i: judged[i]["p"])
    m = len(judged)
    kmax = 0
    for rank, i in enumerate(order, 1):
        if judged[i]["p"] <= rank * 0.10 / m:
            kmax = rank
    for rank, i in enumerate(order, 1):
        judged[i]["bh_pass_q10"] = bool(rank <= kmax)
    for t in sec:
        t.setdefault("bh_pass_q10", False)
        if t.get("judged"):
            t["verdict_secondary"] = "分得出（次要、未校正）" if t.get("resolved") else "分不出"
        else:
            t["verdict_secondary"] = "樣本不足，不判定"
    res["secondary"] = sec
    res["secondary_summary"] = {"tests_judged": m, "cleared_uncorrected": int(sum(1 for t in judged if t.get("resolved"))),
                                "bh_q10_pass": int(sum(1 for t in judged if t["bh_pass_q10"])),
                                "chance_expectation_pct": 5, "descriptive_only": int(sum(1 for t in sec if not t.get("judged")))}

    # ---- claim rows (mechanical)
    res["claim_rows"] = claim_rows(res)

    # ---- lists
    res["crash_dates_le_minus2"] = fmt_dates(S, S["crash2"])
    res["crash_dates_le_minus3"] = fmt_dates(S, S["crash3"])
    res["episodes_5_session"] = ep5_tab
    res["episodes_20_session"] = ep20_tab
    return res


def reg_dummy(df: pd.DataFrame, col: str, level: str) -> dict:
    y = df["H0"].values * 100
    x = df[col].values
    cl = df["ep5"].values
    n1, n0 = int((x == 1).sum()), int((x == 0).sum())
    g1, g0 = int(df.loc[df[col] == 1, "ep5"].nunique()), int(df.loc[df[col] == 0, "ep5"].nunique())
    if n1 < 2 or n0 < 2:
        return {"judged": False, "n_1": n1, "n_0": n0, "g_1": g1, "g_0": g0, "reason": "a group is empty or has fewer than 2 days"}
    beta, se, g = cr1_ols(y, np.column_stack([np.ones(len(y)), x]), cl)
    d, s = float(beta[1]), float(se[1])
    cv = crit(level, int(g))
    return {"n_1": n1, "g_1": g1, "mean_1": float(y[x == 1].mean()), "n_0": n0, "g_0": g0, "mean_0": float(y[x == 0].mean()),
            "diff": d, "se": s, "g_total": int(g), "g_a_total": int(g), "crit": cv, "bound": cv * s, "bound_196": 1.96 * s,
            "resolved": bool(abs(d) > cv * s), "p": pval(d / s, int(g)), "df": int(g) - 1 if g < 30 else None,
            "judged": bool(size_ok(n1, g1) and size_ok(n0, g0))}


def c3_test(S: pd.DataFrame, base_mask: np.ndarray) -> dict:
    c = S[S["crash2"] & S["H0"].notna() & S["vix_prev"].notna()].copy()
    b = S[base_mask & S["vix_prev"].notna()].copy()

    def bucket(v):
        return np.where(v < 20, 0, np.where(v < 30, 1, 2))
    c["k"], b["k"] = bucket(c["vix_prev"].values), bucket(b["vix_prev"].values)
    mapping = {0: 0, 1: 1, 2: 2}
    notes = []
    for k in (0, 2):
        cc = c[c["k"] == k]
        if not size_ok(len(cc), cc["ep5"].nunique()):
            mapping[k] = 1
            notes.append(f"bucket {k} merged into 20-30 (crash days {len(cc)}, episodes {cc['ep5'].nunique()})")
    c["m"] = c["k"].map(mapping)
    b["m"] = b["k"].map(mapping)
    mid = c[c["m"] == 1]
    if not size_ok(len(mid), mid["ep5"].nunique()) and len(set(mapping.values())) == 1:
        pass
    var, diff, w_sum = 0.0, 0.0, 0
    parts = []
    for k in sorted(c["m"].unique()):
        ck, bk = c[c["m"] == k], b[b["m"] == k]
        if len(ck) < 2 or len(bk) < 2:
            continue
        cs = mean_cr1(ck["H0"].values * 100, ck["ep5"].values)
        bs = mean_iid(bk["H0"].values * 100)
        if cs["se"] is None:
            continue
        w = len(ck) / len(c)
        diff += w * (cs["mean"] - bs["mean"])
        var += w * w * (cs["se"] ** 2 + bs["se"] ** 2)
        parts.append({"bucket": int(k), "crash_days": int(len(ck)), "episodes": int(ck["ep5"].nunique()), "baseline_days": int(len(bk)), "weight": w,
                      "crash_mean": cs["mean"], "baseline_mean": bs["mean"]})
    se = math.sqrt(var) if var > 0 else None
    gtot = int(c["ep5"].nunique())
    ok = bool(size_ok(len(c), gtot) and se is not None)
    t = {"label": "C3 crash vs non-crash within prior-day VIX buckets (<20 / 20-30 / >=30), weighted by crash days", "diff": diff, "se": se, "n_crash": int(len(c)),
         "g": gtot, "buckets": parts, "merge_notes": notes, "judged": ok, "group_size_ok": ok}
    if se:
        cv = crit("secondary", gtot)
        t.update({"crit": cv, "bound": cv * se, "resolved": bool(abs(diff) > cv * se), "p": pval(diff / se, gtot)})
    return t


def claim_rows(res: dict) -> dict:
    p1, p2 = res["P1"], res["P2"]
    cm = res["crash_days_descriptive"]["mean_H0_pct"]
    if not p1["judged"]:
        r1 = "P1_descriptive_only"
    elif p1["resolved"] and p1["diff"] < 0 and cm < 0:
        r1 = "P1_neg_clears_crash_mean_neg"
    elif p1["resolved"] and p1["diff"] < 0:
        r1 = "P1_neg_clears_crash_mean_nonneg"
    elif p1["resolved"] and p1["diff"] > 0:
        r1 = "P1_pos_clears"
    else:
        r1 = "P1_not_cleared"
    if not p2["judged"]:
        r2 = "P2_descriptive_only"
    elif p2["resolved"] and p2["diff"] < 0:
        r2 = "P2_neg_clears"
    elif p2["resolved"] and p2["diff"] > 0:
        r2 = "P2_pos_clears"
    else:
        r2 = "P2_not_cleared"
    title = ("美股大跌時，黃金為什麼也跟著跌？" if r1 == "P1_neg_clears_crash_mean_neg" else
             f"美股大跌時，黃金反而漲？2008 年以來的 {res['counts']['crash_days_le_minus2']} 天" if r1 == "P1_pos_clears" else
             f"美股大跌時，黃金會跟著跌嗎？2008 年以來的 {res['counts']['crash_days_le_minus2']} 天")
    return {"P1": r1, "P2": r2, "title": title,
            "P1_fragile": bool(p1["resolved"] and p1["fragility"]["verdict_flips"] > 0) if p1["judged"] else False,
            "P2_fragile": bool(p2.get("resolved") and p2["fragility"]["verdict_flips"] > 0) if p2["judged"] else False}


# ----------------------------------------------------------------------------- S-SRC source sensitivity (pre-declared; amendment 2026-11-04)

FX_START = pd.Timestamp("2013-07-16")


def s_src(spx, gold, fx, vix, dxy, flags, main_res) -> dict:
    """P1 and P2 recomputed with FX_IDC replacing Twelve Data from 2013-07-16 on (Twelve Data before). Reported only; never the verdict;
    not in the secondary family or BH. The single session 2013-07-16 has its H0 spanning the junction (TD close 2013-07-15 -> FX_IDC close)."""
    alt = pd.concat([gold[gold.index < FX_START], fx[fx.index >= FX_START]]).sort_index()
    S2 = build_sessions(spx, alt, vix, dxy)
    r = run_analysis(S2, flags)
    out = {"definition": "gold = Twelve Data XAU/USD before 2013-07-16, FX_IDC:XAUUSD from 2013-07-16; all else as the primary run; H0 of the 2013-07-16 session spans the junction"}
    for k in ("P1", "P2"):
        a, m = r[k], main_res[k]
        d = {"judged": a["judged"], "verdict": a["verdict"], "diff": a.get("diff"), "bound": a.get("bound"), "bound_196": a.get("bound_196"),
             "resolved": a.get("resolved"), "fragility_flips": a["fragility"]["verdict_flips"],
             "flipping_episodes": a["fragility"]["flipping_episodes"],
             "primary_diff": m.get("diff"), "primary_verdict": m["verdict"]}
        if k == "P1":
            d["crash_days"] = r["crash_days_descriptive"]["n_days"]
            d["episodes"] = r["crash_days_descriptive"]["episodes"]
            d["mean_crash_H0_pct"] = r["crash_days_descriptive"]["mean_H0_pct"]
            d["mean_C1_H0_pct"] = r["baseline_C1"]["mean_H0_pct"]
            d["bootstrap_98p75"] = a["bootstrap_98p75"]
        else:
            d.update({"n_vix_jump": a.get("n_1"), "g_vix_jump": a.get("g_1"), "mean_vix_jump": a.get("mean_1"),
                      "n_other": a.get("n_0"), "g_other": a.get("g_0"), "mean_other": a.get("mean_0"), "bootstrap_98p75": a["bootstrap_98p75"]})
        d["flips_primary_verdict"] = bool(a["verdict"] != m["verdict"] or (a.get("resolved") and m.get("resolved") and (a["diff"] > 0) != (m["diff"] > 0)))
        out[k] = d
    out["claim_rows"] = r["claim_rows"]
    out["any_primary_verdict_flipped"] = bool(out["P1"]["flips_primary_verdict"] or out["P2"]["flips_primary_verdict"])
    out["sentence_if_flipped"] = "這個結論對黃金資料來源敏感：換成另一家報價，結果變成 ⟨…⟩"
    return out


# ----------------------------------------------------------------------------- README

def pp(x, d=3):
    return "n/a" if x is None else f"{x:+.{d}f}"


def readme(payload: dict) -> str:
    r = payload
    c, P1, P2 = r["coverage"], r["P1"], r["P2"]
    cd, b = r["crash_days_descriptive"], r["baseline_C1"]
    pc = r["P2_counts"]
    L = []
    L.append(f"# {STUDY_ID} — gold on S&P 500 crash days (results)\n")
    L.append(f"Status: `progress`. Generated by `scripts/research/build_xauusd_equity_crash.py` ({payload['generated_at']}). "
             "Design pre-registered 2026-11-04 (the research record: original section, design amendment and the precheck amendment for checks 2a/2c). Every difference below is in percentage points of gold's simple return; "
             "bounds are the pre-registered critical value times the clustered standard error. 「分得出」 means the difference exceeds its bound; 「分不出」 means it does not "
             "(it does not mean no relationship).\n")
    L.append("## Sample\n")
    L.append(f"- NYSE sessions {c['first_session']} to {c['last_session']}: {c['sessions']} sessions. S&P 500 = Yahoo `^GSPC` close-to-close.")
    L.append(f"- Crash days (S&P 500 ≤ −2%): {c['crash_days_le_minus2']} days in {c['episodes_5_session']} episodes (5-session gap; {c['single_day_episodes_5_session']} single-day); "
             f"{c['episodes_20_session']} episodes at the 20-session gap. ≤ −3%: {c['crash_days_le_minus3']} days, {c['episodes_5_session_le_minus3']} / {c['episodes_20_session_le_minus3']} episodes (5 / 20-session).")
    L.append(f"- {c['crash_days_before_2013_07_16']} of the {c['crash_days_le_minus2']} crash days fall before 2013-07-16 (no FX_IDC overlap; only the GC=F check covers them).")
    L.append(f"- Crash days dropped from the H0 tests: gold missing {c['crash_days_gold_H0_missing']}; VIX missing (P2 only) {c['crash_days_vix_missing']}; DXY missing {c['crash_days_dxy_missing']}.")
    L.append("- Gold H0 = close(t−1)→close(t) on NYSE sessions (Twelve Data daily bar stamped t; closes about 1–2 h after the 16:00 ET S&P close). After an NYSE holiday H0 spans two gold bars; ordinary days are measured the same way.\n")
    L.append("## Pre-run checks (before any outcome)\n")
    L.append("| check | result | action |\n|---|---|---|")
    for k, v in r["prechecks"].items():
        L.append(f"| {k} | {'pass' if v.get('pass') else 'FAIL'} | {v.get('action')} |")
    L.append("")
    L.append("## Primary results (Bonferroni over 2; bound = 2.2414 × SE, or t(0.9875, G−1) if G < 30)\n")
    L.append("| | n (days / episodes) | mean H0 | comparison | difference | bound ±(2.24×SE) | 1.96×SE (reference) | verdict |\n|---|---|---|---|---|---|---|---|")
    L.append(f"| P1 crash days (≤ −2%) | {cd['n_days']} / {cd['episodes']} | {pp(cd['mean_H0_pct'])}% | C1 non-crash days n={b['n_days']}, mean {pp(b['mean_H0_pct'])}% | "
             f"{pp(P1['diff'])} pp | ±{P1['bound']:.3f} | ±{P1['bound_196']:.3f} | **{P1['verdict']}** |")
    if P2.get("diff") is not None:
        L.append(f"| P2 VIX-jump crash days | {P2['n_1']} / {P2['g_1']} | {pp(P2['mean_1'])}% | other crash days n={P2['n_0']} / {P2['g_0']} episodes, mean {pp(P2['mean_0'])}% | "
                 f"{pp(P2['diff'])} pp | ±{P2['bound']:.3f} | ±{P2['bound_196']:.3f} | **{P2['verdict']}** |")
    L.append("")
    L.append(f"- P1 sensitivity: bootstrap (10,000 episode draws, seed 20261104) 98.75% interval {pp(P1['bootstrap_98p75'][0])} to {pp(P1['bootstrap_98p75'][1])} pp; "
             f"episode-equal-weight mean difference {pp(P1['episode_equal_weight']['diff'])} pp (bound ±{P1['episode_equal_weight']['bound']:.3f}, {P1['episode_equal_weight']['episodes']} episodes). "
             f"Leave-one-episode-out: {P1['fragility']['verdict_flips']} of {P1['fragility']['removals']} removals change the verdict.")
    if P1["fragility"]["flipping_episodes"]:
        L.append("  - Episodes whose removal flips P1: " + "; ".join(f"{e['first']}–{e['last']} ({e['days']} d)" for e in P1["fragility"]["flipping_episodes"]) + ".")
    L.append(f"- Gold up on {cd['gold_up_days']} of the {cd['n_days']} crash days ({cd['gold_up_share_pct']:.1f}%); C1 up share {b['up_share_pct']:.1f}% (n={b['n_days']}).")
    if P2.get("diff") is not None:
        L.append(f"- P2 sensitivity: bootstrap 98.75% interval {pp(P2['bootstrap_98p75'][0])} to {pp(P2['bootstrap_98p75'][1])} pp; "
                 f"leave-one-episode-out: {P2['fragility']['verdict_flips']} of {P2['fragility']['removals']} removals change the verdict.")
        if P2["fragility"]["flipping_episodes"]:
            L.append("  - Episodes whose removal flips P2: " + "; ".join(f"{e['first']}–{e['last']} ({e['days']} d)" for e in P2["fragility"]["flipping_episodes"]) + ".")
        if "episode_equal_weight" in P2:
            L.append(f"  - Episode-equal-weight difference {pp(P2['episode_equal_weight']['diff'])} pp (bound ±{P2['episode_equal_weight']['bound']:.3f}; SE ignores shared-episode covariance).")
        L.append(f"- P2 group sizes: VIX-jump {pc['vix_jump_days']} days / {pc['vix_jump_episodes']} episodes; other {pc['other_days']} / {pc['other_episodes']}; "
                 f"DXY-up share within VIX-jump {r['P2']['dxy_up_share_by_group']['vix_jump']['dxy_up']}/{r['P2']['dxy_up_share_by_group']['vix_jump']['n_with_dxy']}, within other "
                 f"{r['P2']['dxy_up_share_by_group']['other']['dxy_up']}/{r['P2']['dxy_up_share_by_group']['other']['n_with_dxy']}.")
    L.append(f"- Claim rows selected mechanically: P1 `{r['claim_rows']['P1']}`, P2 `{r['claim_rows']['P2']}`; title: 「{r['claim_rows']['title']}」.\n")
    ss_ = r["sensitivity_S_SRC"]
    L.append("## Source sensitivity S-SRC (pre-declared; reported only, never the verdict; not in the secondary family or BH)\n")
    L.append(f"{ss_['definition']}.\n")
    p1s, p2s = ss_["P1"], ss_["P2"]
    L.append("| | primary (Twelve Data) | S-SRC (FX_IDC from 2013-07-16) | verdict flips? |\n|---|---|---|---|")
    L.append(f"| P1 difference (pp) / verdict | {pp(p1s['primary_diff'])} / {p1s['primary_verdict']} | {pp(p1s['diff'])} (bound ±{p1s['bound']:.3f}; crash mean {pp(p1s['mean_crash_H0_pct'])}%, C1 {pp(p1s['mean_C1_H0_pct'])}%, {p1s['crash_days']} days / {p1s['episodes']} episodes) / {p1s['verdict']} | {'yes' if p1s['flips_primary_verdict'] else 'no'} |")
    L.append(f"| P2 difference (pp) / verdict | {pp(p2s['primary_diff'])} / {p2s['primary_verdict']} | {pp(p2s['diff'])} (bound ±{p2s['bound']:.3f}; VIX-jump {p2s['n_vix_jump']}/{p2s['g_vix_jump']}, other {p2s['n_other']}/{p2s['g_other']}) / {p2s['verdict']} | {'yes' if p2s['flips_primary_verdict'] else 'no'} |")
    L.append(f"\nS-SRC claim rows: P1 `{ss_['claim_rows']['P1']}`, P2 `{ss_['claim_rows']['P2']}`. Leave-one-episode-out flips: P1 {p1s['fragility_flips']}, P2 {p2s['fragility_flips']}. "
             + ("If a primary verdict flips, the statement must add: " + ss_["sentence_if_flipped"] + "." if ss_["any_primary_verdict_flipped"] else "No primary verdict flips, so the source sentence is not required.") + "\n")
    L.append("## Secondary results (uncorrected; 1.96 × SE or t(0.975, G−1); Benjamini–Hochberg q = 0.10 column)\n")
    ss = r["secondary_summary"]
    L.append(f"{ss['tests_judged']} secondary tests were judged; {ss['cleared_uncorrected']} cleared their uncorrected bound (chance alone gives about 5%); {ss['bh_q10_pass']} pass BH q = 0.10; "
             f"{ss['descriptive_only']} were descriptive only (group too small). Only BH-passing results may be mentioned, never promoted.\n")
    L.append("| test | groups | estimate (pp) | bound | verdict | BH q=0.10 |\n|---|---|---|---|---|---|")
    for t in r["secondary"]:
        est = t.get("diff", t.get("coef", t.get("diff_pp")))
        bnd = t.get("bound")
        grp = t.get("groups") or {"n": t.get("n"), "g": t.get("g")}
        bnd_s = "n/a" if bnd is None else f"±{bnd:.3f}"
        if t["name"] == "down-day share":
            bnd_s = f"crash-share 95% bootstrap {t['bootstrap_crash_share_95_pct'][0]:.1f}–{t['bootstrap_crash_share_95_pct'][1]:.1f}% vs C1 {t['baseline_share_pct']:.1f}%"
            grp = {"crash_days": t["n_crash"], "episodes": t["episodes"], "C1_days": t["baseline_n"]}
        L.append(f"| {t['name']} | {json.dumps(grp, ensure_ascii=False)} | {pp(est)} | {bnd_s} | {t['verdict_secondary']} | {'yes' if t.get('bh_pass_q10') else 'no'} |")
    db = r["dollar_baseline_non_crash"]
    L.append(f"\nDollar baseline (non-crash C1 days): gold H0 on DXY-up days {pp(db['mean_up_pct'])}% (n={db['dxy_up_days']}) minus DXY-down days {pp(db['mean_down_pct'])}% (n={db['dxy_down_days']}) = {pp(db['diff'])} pp (1.96×SE ±{db['bound_196']:.3f}).\n")
    L.append("## Limitations\n")
    for s in payload["limitations"]:
        L.append(f"- {s}")
    L.append("\n## Reproduce\n\n`~/.venvs/trading-research/bin/python scripts/research/build_xauusd_equity_crash.py` reads the frozen inputs in `reproduction_inputs.json` (each verified by sha256) and rewrites `results.json` and this file. "
             "Public scope: aggregates and the crash-date lists only; no per-day prices, returns or group labels.\n")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prechecks-only", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    sources = load_manifest()
    spx = read_yahoo(verified_path(sources, "spx_1d"))
    gold = read_gold(verified_path(sources, "price_1d_frozen"))
    vix_cboe = read_cboe(verified_path(sources, "vix_1d"))
    vix_fred = read_fred(verified_path(sources, "vix_1d_crosscheck"))
    dxy = read_yahoo(verified_path(sources, "dxy_1d"))
    S0 = build_sessions(spx, gold, vix_cboe, dxy)
    pre = prechecks(sources, S0, spx, gold, vix_cboe)
    for k, v in pre.items():
        print(k, json.dumps(v, ensure_ascii=False, default=float))
    if a.prechecks_only:
        return
    # fixed actions
    hard_stop = []
    if not pre["check1_gold_day_boundary"]["pass"]:
        hard_stop.append("check1")
    if not pre["check2a_stability_and_fxidc_adjudication"]["pass"]:
        hard_stop.append("check2a")
    if not pre["check2b_fxidc_2013_07_16_on"]["pass"]:
        hard_stop.append("check2b")
    if not pre["check2c_relative_2008_2013"]["pass"]:
        hard_stop.append("check2c")
    if pre["check3_vix"]["action"].startswith("STOP"):
        hard_stop.append("check3")
    if not pre["check4_spx"]["pass"]:
        hard_stop.append("check4")
    if hard_stop:
        stopped = {"study_id": STUDY_ID, "schema_version": 1, "revision": 2,
                   "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
                   "status": "stopped_at_prechecks",
                   "note": "A registered pre-run check hit its fixed 'stop' action before any outcome was computed (registered stop rule). No P1/P2/secondary result exists.",
                   "stopped_checks": hard_stop, "prechecks": pre,
                   "method_parameters": None, "coverage": None, "baseline": None, "subgroups": None, "confidence": None, "limitations": []}
        txt = json.dumps(stopped, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (bool(o) if isinstance(o, np.bool_) else str(o)))
        if not a.dry:
            (OUT / "results.json").write_text(txt + "\n")
        print("STOPPED at pre-run checks:", hard_stop)
        sys.exit(2)
    vix_series = vix_cboe if pre["check3_vix"]["pass"] else vix_fred
    S = build_sessions(spx, gold, vix_series, dxy)
    flags = {"dxy_ok": bool(pre["check5_dxy_coverage"]["pass"]), "vix_source": "cboe" if pre["check3_vix"]["pass"] else "fred_vixcls"}
    res = run_analysis(S, flags)
    fx_frozen = read_fx(verified_path(sources, "price_1d_fxidc_frozen"))
    ssrc = s_src(spx, gold, fx_frozen, vix_series, dxy, flags, res)
    # out-of-sample crash days after the fixed sample end (dates only; not used in any test)
    after = spx[spx.index > pd.Timestamp(SAMPLE_END)]
    after_ret = spx.pct_change().reindex(after.index)
    oos = [d.strftime("%Y-%m-%d") for d in after_ret.index[after_ret <= CRASH_PRIMARY + EPS]]
    payload = {
        "study_id": STUDY_ID, "schema_version": 1, "revision": 2,
        "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "status": "computed",
        "method_parameters": {
            "sample": f"NYSE sessions {SAMPLE_START} to {SAMPLE_END} (fixed)", "crash_primary_pct": -2.0, "crash_secondary_pct": -3.0,
            "vix_jump": "VIX close(t)/close(t-1)-1 >= +20%, t-1 previous NYSE session", "episode_gap_sessions": [GAP5, GAP20],
            "clustered_se": "CR1 G/(G-1)*(N-1)/(N-K)", "primary_critical": "2.2414 or t(0.9875,G-1) when G<30",
            "secondary_critical": "1.96 or t(0.975,G-1) when G<30", "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED, "unit": "episode", "baseline": "C1 days resampled iid"},
            "group_size_rule": f">= {MIN_DAYS} days and >= {MIN_EPISODES} episodes", "local_window_sessions": LOCAL_WINDOW,
            "h5_h20": "first crash day of each 20-session episode (iid SE) vs all non-crash days (Newey-West, lag = horizon)",
            "implementation_choices_not_fixed_in_registration": [
                "-3% secondary tests use the primary C1 non-crash baseline (S&P > -2%)",
                "VIX-level split uses prior-day VIX close(t-1) >= 30",
                "bootstrap resamples C1 baseline days iid (episodes for the crash side)",
                "floating-point tolerance 1e-12 on the -2%/-3%/+20% thresholds",
                "down-day share enters BH with an episode-bootstrap two-sided p-value",
                "S-USD2 b and c each count as one secondary test",
                "down-day-share interval = bootstrap of the crash share minus an iid-resampled C1 share, plus the C1 point share"],
            "flags": flags,
            "prechecks_amended": "checks 2a and 2c amended 2026-11-04 before any outcome was computed; thresholds unchanged otherwise",
            "S_SRC": "pre-declared source sensitivity, reported only"},
        "coverage": res["counts"],
        "prechecks": pre,
        "baseline_C1": res["baseline_C1"], "all_days_descriptive": res["all_days_descriptive"], "crash_days_descriptive": res["crash_days_descriptive"],
        "P1": res["P1"], "P2_counts": res["P2_counts"], "P2": res["P2"], "dollar_baseline_non_crash": res["dollar_baseline_non_crash"],
        "sensitivity_S_SRC": ssrc,
        "secondary": res["secondary"], "secondary_summary": res["secondary_summary"], "claim_rows": res["claim_rows"],
        "crash_dates_le_minus2": res["crash_dates_le_minus2"], "crash_dates_le_minus3": res["crash_dates_le_minus3"],
        "episodes_5_session": res["episodes_5_session"], "episodes_20_session": res["episodes_20_session"],
        "out_of_sample_crash_days_after_sample_end": {"dates": oos, "rows_after_sample_end": int(len(after)), "used_in_any_test": False,
                                                      "note": "the last row may be an incomplete session"},
        "limitations": [
            "Same-day co-movement, not causation. P2's VIX label uses no USD-priced input, but a VIX jump may coincide with dollar strength, so a P2 difference may still run partly through the dollar.",
            "Gold daily bar (Twelve Data) closes about 1-2 h after the 16:00 ET S&P 500 close; DXY about 1 h after. The 1h stamp convention is not verified, and before the restatement the daily close equalled the 20Z-stamped 1h close. The 1h overlap covers 2026-04 to 2026-10 (US daylight time) only; the boundary in US standard time and before 2026-04 is assumed (the check 2c lead-coefficient comparison supports it only partly).",
            "Twelve Data restated 2024-05 to 2026-10 (and 24 earlier dates, 646 dates in all) between the 2026-10-10 and 2026-10-17 fetches; the frozen post-restatement version was used and adjudicated against FX_IDC (check 2a). 12 crash days have a revised H0 endpoint; they are not excluded. S-SRC reports the FX_IDC sensitivity.",
            f"{res['counts']['crash_days_before_2013_07_16']} of {res['counts']['crash_days_le_minus2']} crash days are before 2013-07-16 (no FX_IDC overlap); the only independent check of 2008-2013 is Yahoo GC=F (COMEX futures, settlement about 13:30 ET), because Stooq was blocked; it is judged relative to the FX_IDC vs GC=F reference pair (amended check 2c), not against an absolute 0.95.",
            "2008-2026 only; 1987 and 2000-2002 not covered.",
            "Crash days cluster; the effective sample is the episode count.",
            "Checks 2a and 2c were amended after their original thresholds failed. The new thresholds were set after the check values were seen but before any outcome was computed, so passing them means the numbers reproduce; it is not an independent data-quality gate.",
            "Not fully blind: the S&P 500 crash-day counts and the gold 1h/1d close-time alignment had been looked at before the pre-run checks, and the researcher recalled some crash episodes from memory. No gold, VIX or DXY value on a crash day had been seen before the pre-run checks.",
            "Secondary results are uncorrected; none is a headline.",
            "The two secondary results that pass BH (DXY-only split -1.098 pp; S-USD2 c -0.652 per 1 pp of DXY return) are the mechanical effect of gold being priced in US dollars, not coincidence and not safe-haven behaviour: on ordinary days gold is already 0.725 pp lower when the dollar rises (+/-0.059, n=4,478). Read them only next to that baseline. Post-hoc: the DXY-only split net of the baseline is -0.373 pp (bound +/-0.593, not distinguishable); the simple slope of gold H0 on DXY return is -1.00 on ordinary days and -0.72 on crash days (difference not tested).",
            "The down-day-share interval (45.3-58.5%) bootstraps the crash share minus an iid-resampled ordinary-day share and adds the ordinary-day point share, so it includes baseline sampling error; a pure crash-share bootstrap gives 45.5-58.5% (post-hoc recomputation); same verdict (not distinguishable)."],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else (bool(o) if isinstance(o, np.bool_) else str(o)))
    if a.dry:
        print(text[:6000])
        return
    (OUT / "results.json").write_text(text + "\n")
    (OUT / "README.md").write_text(readme(json.loads(text)) + "\n")
    print("written", OUT)


if __name__ == "__main__":
    main()
