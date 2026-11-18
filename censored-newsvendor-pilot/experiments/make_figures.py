"""Draw every figure from results/*.json into figures/ (PDF for the note, PNG for the README)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import config as C  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e8e7e3", "#a9a8a2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # with signal, no signal, naive
BLUES = {2: "#86b6ef", 3: "#3987e5", 5: "#1c5cab", 8: "#0d366b"}   # ordinal: more buckets, darker
CRIT = "#d03b3b"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlelocation": "left",
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
    "lines.markersize": 4.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 220,
    "pdf.fonttype": 42,
})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def pct_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))


# ------------------------------------------------------------- figure 1: setup
def fig_setup():
    e2 = load("e2_noise")
    p, lam, M = C.P_TRUE, C.LAM, C.M
    q_sig = next(r["q"] for r in e2["rows"] if r["L"] == C.MAIN_L and r["sigma"] == C.MAIN_SIGMA)
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    y = np.arange(M + 1)
    ax.bar(y[:lam], p[:lam], width=0.8, color="#86b6ef", label="seen in sales (identified)")
    ax.bar(y[lam:], p[lam:], width=0.8, color="none", edgecolor=AXIS, linewidth=0.7, hatch="////",
           label=f"hidden by stockouts (only the total, {p[lam:].sum():.2f}, is known)")
    top = p.max() * 1.12
    for x, lab, col, ls in ((lam, f"largest past order λ = {lam}", INK2, (0, (3, 2))),
                            (C.Q_STAR, f"optimal order q* = {C.Q_STAR}", INK, "-"),
                            (q_sig, f"robust order, with forecast: {q_sig:.1f}", BLUE, "-"),
                            (e2["no_signal"]["q"], f"robust order, no forecast: {e2['no_signal']['q']:.1f}", ORANGE, "-")):
        ax.axvline(x, color=col, lw=1.3, ls=ls, label=lab)
    ax.set_ylim(0, top)
    ax.set_xlim(-1, M + 1)
    ax.set_xlabel("Demand in a period, y")
    ax.set_ylabel("Probability")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.6, columnspacing=1.2)
    save(fig, "fig1_setup")


# ------------------------------------------------------------- figure 2: noise
def fig_noise():
    e2 = load("e2_noise")
    rows = [r for r in e2["rows"] if r["sigma"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, key, title in ((axes[0], "cert_pct", "(a) Guaranteed worst case"),
                           (axes[1], "true_pct", "(b) Actual excess cost")):
        for L, col in BLUES.items():
            rr = [r for r in rows if r["L"] == L]
            x = [r["sigma"] / C.SD for r in rr]
            ax.plot(x, [r[key] for r in rr], "-o", color=col, label=f"{L}-level forecast", ms=3.2)
        ref = e2["no_signal"][key]
        ax.axhline(ref, color=ORANGE, ls=(0, (4, 2)), lw=1.3)
        ax.text(0.042, ref * (1.12 if key == "true_pct" else 0.97), f"no forecast: {ref:.0f}%",
                color=ORANGE, va="bottom" if key == "true_pct" else "top", fontsize=7.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Forecast noise σ / sd(demand)")
        ax.set_title(title)
        pct_axis(ax)
    axes[0].set_ylabel("Excess cost over the optimum")
    axes[0].set_ylim(0, 105)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.005, 60)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 0.80))
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_value_vs_noise")


# ------------------------------------------------------------- figure 3: boundary
def fig_boundary():
    e3 = load("e3_boundary")["rows"]
    x = [r["F0"] for r in e3]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    for key, col, lab in (("naive_pct", AQUA, "treat sales as demand"),
                          ("nosig_pct", ORANGE, "robust, no forecast"),
                          ("sig_pct", BLUE, "robust, with forecast")):
        ax.plot(x, [max(r[key], 1e-3) for r in e3], "-o", color=col, label=lab, ms=3.2)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 300)
    ax.set_title("(a) Actual excess cost")
    ax.set_ylabel("Excess cost over the optimum")
    ax.legend(loc="lower left")
    ax2 = axes[1]
    for key, col, lab in (("nosig_cert_pct", ORANGE, "robust, no forecast"),
                          ("sig_cert_pct", BLUE, "robust, with forecast")):
        ax2.plot(x, [r[key] for r in e3], "-o", color=col, label=lab, ms=3.2)
    ax2.set_title("(b) Guaranteed worst case")
    ax2.text(x[2], e3[2]["nosig_cert_pct"] + 12, "robust, no forecast", color=ORANGE, fontsize=7.5)
    ax2.text(x[2], e3[2]["sig_cert_pct"] + 10, "robust, with forecast", color=BLUE, fontsize=7.5)
    for a in axes:
        a.set_xlabel("Share of demand seen below λ, P(D < λ)")
        a.axvline(C.RHO, color=INK2, lw=1, ls=(0, (2, 2)))
        a.text(C.RHO - 0.01, a.get_ylim()[1] * (0.6 if a is ax else 0.95), "critical\nratio 0.8",
               ha="right", va="top", color=INK2, fontsize=7)
        pct_axis(a)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_value_vs_censoring")


# ------------------------------------------------------------- figure 4: misspecification
def fig_misspec():
    e4 = load("e4_misspecification")
    d_allow = C.delta_allowance(C.MAIN_L)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, (k, title) in zip(axes, (("0.3", "(a) Mild tail bias, κ = 0.3"),
                                     ("0.6", "(b) Strong tail bias, κ = 0.6"))):
        v = e4[k]
        rr = [r for r in v["rows"] if not r["empty"]]
        d = np.array([r["delta"] for r in rr])
        cert = np.array([r["cert_pct"] for r in rr])
        act = np.array([r["true_pct"] for r in rr])
        ax.fill_between(d, 0, 30, where=cert < act, color=CRIT, alpha=0.12, lw=0, step="mid")
        ax.plot(d, cert, color=INK2, ls=(0, (4, 2)), lw=1.4, label="guaranteed worst case")
        ax.plot(d, act, color=BLUE, lw=1.8, label="actual excess cost")
        right = k == "0.3"
        ax.axvline(v["delta_star"], color=INK, lw=1, ls=(0, (1, 2)))
        ax.text(v["delta_star"] + (0.003 if right else -0.003), 21, "δ*: true demand\nenters the set",
                fontsize=7, color=INK, va="top", ha="left" if right else "right")
        ax.axvline(d_allow, color=AXIS, lw=1)
        ax.text(d_allow - 0.003, 27, f"sampling allowance\nfor n = {C.N_EFF:,}", fontsize=7, color=INK2,
                va="top", ha="right")
        if cert[0] < act[0]:
            ax.annotate(f"at δ = 0 the guarantee is broken:\nclaims {cert[0]:.1f}%, actual {act[0]:.1f}%",
                        xy=(0.0015, act[0]), xytext=(0.085, 4.6), color=CRIT, fontsize=7, va="bottom",
                        arrowprops={"arrowstyle": "->", "color": CRIT, "lw": 0.9})
        ax.set_title(title)
        ax.set_xlabel("Relaxation δ (l1 budget for forecast error)")
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0, 28)
        pct_axis(ax)
    axes[0].set_ylabel("Excess cost over the optimum")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    save(fig, "fig4_misspecification")


# ------------------------------------------------------------- figure 5: finite samples
def fig_finite():
    e5 = load("e5_finite_sample")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    pols = (("naive_pct", AQUA, "treat sales as demand"), ("nosig_pct", ORANGE, "robust, no forecast"),
            ("sig_pct", BLUE, "robust, with forecast"))
    ns = (250, 1000, 4000)
    for ax, sigma, title in ((axes[0], "4.0", "(a) Sharp forecast, σ = 4"),
                             (axes[1], "16.0", "(b) Noisy forecast, σ = 16")):
        for gi, n in enumerate(ns):
            rows = e5[f"{sigma}|{n}"]
            for pi, (key, col, lab) in enumerate(pols):
                vals = np.clip([r[key] for r in rows], 0.003, None)
                pos = gi + (pi - 1) * 0.26
                bp = ax.boxplot(vals, positions=[pos], widths=0.2, whis=(5, 95), showfliers=False,
                                patch_artist=True, medianprops={"color": INK, "lw": 1.2})
                for part in ("boxes",):
                    for b in bp[part]:
                        b.set(facecolor=col, alpha=0.85, edgecolor=col, lw=0.8)
                for part in ("whiskers", "caps"):
                    for b in bp[part]:
                        b.set(color=col, lw=1)
                if np.subtract(*np.percentile(vals, [75, 25])) < 1e-9:   # (almost) constant: draw it as a bar
                    ax.plot([pos - 0.1, pos + 0.1], [np.median(vals)] * 2, color=col, lw=3.5, solid_capstyle="butt")
                if gi == 0 and ax is axes[0]:
                    ax.plot([], [], "s", color=col, label=lab, ms=6)
            cov = np.mean([r["sig_covered"] for r in rows])
            ax.text(gi + 0.26, 0.0042, f"held {cov:.0%}", ha="center", fontsize=6.5, color=BLUE)
        ax.set_xticks(range(len(ns)), [f"n = {n:,}" for n in ns])
        ax.set_yscale("log")
        ax.set_ylim(0.002, 200)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        pct_axis(ax)
        ax.set_xlabel("Periods of sales history")
    axes[0].set_ylabel("Excess cost over the optimum")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.5)
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    save(fig, "fig5_finite_sample")


if __name__ == "__main__":
    fig_setup()
    fig_noise()
    fig_boundary()
    fig_misspec()
    fig_finite()
    print("figures written to", FIG)
