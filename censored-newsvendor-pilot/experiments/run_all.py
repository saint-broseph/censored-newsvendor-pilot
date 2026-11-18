"""Run every experiment and save the numbers to results/*.json.

    python experiments/run_all.py            # all experiments (~5 minutes)
    python experiments/run_all.py e2 e4      # a subset

E1  the robust LP reproduces the Hssaine-Sinclair closed form
E2  value of the signal against forecast noise and number of buckets
E3  value of the signal against the severity of censoring
E4  a wrong error model: when the guarantee fails, and how delta repairs it
E5  finite samples: everything estimated from n simulated periods
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from pilot import (AmbiguitySet, EmptyAmbiguitySet, critical_quantile, hssaine_sinclair,  # noqa: E402
                   regret, sales_pmf, sales_quantile, simulate_history, solve_robust)

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def pct(x: float) -> float:
    """Regret as a percentage of the optimal expected cost."""
    return 100.0 * x / C.C_STAR


def naive_q(lam: int) -> int:
    return critical_quantile(sales_pmf(C.P_TRUE, C.order_levels(lam)), C.RHO)


def save(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------- E1
def e1():
    rows = []
    for lam in (22, 26, 30, 34, 38):
        for b, h in ((4, 1), (9, 1), (3, 2), (19, 1), (1, 1)):
            amb = AmbiguitySet.population(C.P_TRUE, lam)
            F0 = float(C.P_TRUE[:lam].sum())
            q_hs, d_hs = hssaine_sinclair(F0, lam, C.M, b, h)
            sol = solve_robust(amb, b, h)
            rows.append({"lam": lam, "b": b, "h": h, "F0": F0, "rho": b / (b + h),
                         "q_lp": sol.q, "delta_lp": sol.value,
                         "q_closed": q_hs, "delta_closed": d_hs,
                         "true_regret_lp": regret(sol.q, C.P_TRUE, b, h)})
    unid = [r for r in rows if r["q_closed"] is not None]
    summary = {
        "cases": len(rows), "unidentifiable": len(unid),
        "max_abs_q_error": max(abs(r["q_lp"] - r["q_closed"]) for r in unid),
        "max_abs_delta_error": max(abs(r["delta_lp"] - r["delta_closed"]) for r in rows),
    }
    save("e1_closed_form", {"summary": summary, "rows": rows})
    return summary


# ---------------------------------------------------------------- E2
def e2():
    base = solve_robust(AmbiguitySet.population(C.P_TRUE, C.LAM), C.B, C.H)
    base_true = regret(base.q, C.P_TRUE, C.B, C.H)
    qn = naive_q(C.LAM)
    sigmas = [0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    rows = []
    for L in C.CUTS:
        d = C.delta_allowance(L)
        for s in sigmas:
            amb = AmbiguitySet.population(C.P_TRUE, C.LAM, C.signal(L, s), delta=d)
            sol = solve_robust(amb, C.B, C.H)
            exact = solve_robust(amb.with_delta(0.0), C.B, C.H)      # infinite data
            tr = regret(sol.q, C.P_TRUE, C.B, C.H)
            rows.append({"L": L, "sigma": s, "delta": d, "q": sol.q, "certificate": sol.value,
                         "true_regret": tr, "true_pct": pct(tr), "cert_pct": pct(sol.value),
                         "share_recovered": 1 - tr / base_true,
                         "exact_q": exact.q, "exact_cert_pct": pct(exact.value),
                         "exact_true_pct": pct(regret(exact.q, C.P_TRUE, C.B, C.H))})
    out = {
        "n_eff": C.N_EFF,
        "q_star": C.Q_STAR, "c_star": C.C_STAR, "sd_demand": C.SD, "lam": C.LAM,
        "F0": float(C.P_TRUE[:C.LAM].sum()),
        "no_signal": {"q": base.q, "certificate": base.value, "true_regret": base_true,
                      "true_pct": pct(base_true), "cert_pct": pct(base.value)},
        "naive": {"q": qn, "true_regret": regret(qn, C.P_TRUE, C.B, C.H),
                  "true_pct": pct(regret(qn, C.P_TRUE, C.B, C.H))},
        "rows": rows,
    }
    save("e2_noise", out)
    main = next(r for r in rows if r["L"] == C.MAIN_L and r["sigma"] == C.MAIN_SIGMA)
    return {"no_signal_pct": pct(base_true), "main_pct": main["true_pct"],
            "main_share": main["share_recovered"], "naive_pct": out["naive"]["true_pct"]}


# ---------------------------------------------------------------- E3
def e3():
    rows = []
    for lam in range(20, 41, 2):
        F0 = float(C.P_TRUE[:lam].sum())
        nos = solve_robust(AmbiguitySet.population(C.P_TRUE, lam), C.B, C.H)
        sig = solve_robust(AmbiguitySet.population(C.P_TRUE, lam, C.signal(lam=lam),
                                                   delta=C.delta_allowance(C.MAIN_L, lam)), C.B, C.H)
        qn = naive_q(lam)
        rows.append({"lam": lam, "F0": F0,
                     "naive_q": qn, "naive_pct": pct(regret(qn, C.P_TRUE, C.B, C.H)),
                     "nosig_q": nos.q, "nosig_pct": pct(regret(nos.q, C.P_TRUE, C.B, C.H)),
                     "nosig_cert_pct": pct(nos.value),
                     "sig_q": sig.q, "sig_pct": pct(regret(sig.q, C.P_TRUE, C.B, C.H)),
                     "sig_cert_pct": pct(sig.value)})
    save("e3_boundary", {"rho": C.RHO, "rows": rows})
    return {"lams": [r["lam"] for r in rows]}


# ---------------------------------------------------------------- E4
def e4():
    out = {}
    for shrink in (0.3, 0.6):
        truth = C.signal(shrink=shrink)
        model = C.signal()
        amb0 = AmbiguitySet.population(C.P_TRUE, C.LAM, truth, model=model)
        delta_star = float(np.abs(amb0.K @ C.P_TRUE[C.LAM:] - amb0.beta).sum())
        d_min = amb0.min_delta()
        rows = []
        for d in np.unique(np.concatenate([np.linspace(0, 0.2, 41), [d_min, delta_star]])):
            if d < d_min - 1e-12:
                rows.append({"delta": float(d), "empty": True})
                continue
            sol = solve_robust(amb0.with_delta(max(d, d_min)), C.B, C.H)
            tr = regret(sol.q, C.P_TRUE, C.B, C.H)
            rows.append({"delta": float(d), "empty": False, "q": sol.q,
                         "cert_pct": pct(sol.value), "true_pct": pct(tr),
                         "covered": bool(sol.value >= tr - 1e-9)})
        out[str(shrink)] = {"delta_star": delta_star, "delta_min": d_min, "rows": rows}
    save("e4_misspecification", out)
    return {k: (v["delta_star"], v["delta_min"]) for k, v in out.items()}


# ---------------------------------------------------------------- E5
def e5(reps: int = 100, seed: int = 2027):
    rng = np.random.default_rng(seed)
    res = {}
    for sigma in (4.0, 16.0):
        sig = C.signal(sigma=sigma)
        for n in (250, 1000, 4000):
            rows = []
            for _ in range(reps):
                hist = simulate_history(C.P_TRUE, C.order_levels(), n, sig, rng)
                qn = sales_quantile(hist.sales, C.RHO)
                a_nos = AmbiguitySet.from_history(hist, C.M)
                a_sig = AmbiguitySet.from_history(hist, C.M, C.CUTS[C.MAIN_L], z=1.0)
                nos = solve_robust(a_nos, C.B, C.H)
                s = solve_robust(a_sig, C.B, C.H)
                r_n, r_nos, r_sig = (regret(q, C.P_TRUE, C.B, C.H) for q in (qn, nos.q, s.q))
                rows.append({"naive_pct": pct(r_n), "nosig_pct": pct(r_nos), "sig_pct": pct(r_sig),
                             "naive_q": qn, "nosig_q": nos.q, "sig_q": s.q,
                             "sig_cert_pct": pct(s.value), "nosig_cert_pct": pct(nos.value),
                             "sig_covered": bool(s.value >= r_sig - 1e-9),
                             "nosig_covered": bool(nos.value >= r_nos - 1e-9),
                             "delta": a_sig.delta, "sigma_hat": a_sig.info["sigma_hat"]})
            res[f"{sigma}|{n}"] = rows
    save("e5_finite_sample", {"reps": reps, "seed": seed, "results": res})
    summary = {}
    for k, rows in res.items():
        summary[k] = {m: float(np.median([r[m] for r in rows])) for m in ("naive_pct", "nosig_pct", "sig_pct")}
        summary[k]["sig_coverage"] = float(np.mean([r["sig_covered"] for r in rows]))
    return summary


EXPERIMENTS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    for name in chosen:
        t0 = time.time()
        summary = EXPERIMENTS[name]()
        print(f"{name} done in {time.time() - t0:.1f}s: {summary}")
