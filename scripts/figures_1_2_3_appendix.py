"""v3 figure variants (post early-review iteration):
  1. fig_concept_frac      : Figure 1 + epistemic fractile bands (0.1-0.9, transparent fill)
  2. fig_normalized_hazard_v3 : Figure 2 with 8 IMs (adds SA 2, 5, 10 s); IL4 at k=1.5
  3. fig_rp_ratio_summary_v3     : Figure 3, full period set, IL4 at k=1.5, log-T axis
  4. fig_rp_ratio_summary_v3_linT: as above with linear-T axis
  5. fig_appendix_IL4k13   : two-panel appendix figure, IL4 k=1.3 (normalized curves + RP ratios)
Existing figure files are not overwritten.
"""
import numpy as np, pandas as pd, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
ROOT = str(_REPO)
FIGD = str(_REPO / "figures")
OUTD = str(_REPO / "output")
Path(FIGD).mkdir(exist_ok=True)
Path(OUTD).mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
                     "lines.linewidth": 1.2, "figure.dpi": 150})

hz = pd.read_csv(f"{ROOT}/data/hazard/hazard-curves.csv", skiprows=1)
sites = pd.read_csv(f"{ROOT}/data/hazard/site_lookup.csv")
im_cols = [c for c in hz.columns if c.startswith("annual poe")]
im_vals = np.array([float(re.search(r"- ([\d.]+) g", c).group(1)) for c in im_cols])
key = {(f"{r.lat:.3f}", f"{r.lon:.3f}"): r.code for r in sites.itertuples()}
hz["code"] = [key[(f"{la:.3f}", f"{lo:.3f}")] for la, lo in zip(hz.lat, hz.lon)]
curves = {}   # (code, period, statistic) -> (im, lam)
for _, r in hz.iterrows():
    lam = r[im_cols].astype(float).values
    m = lam > 1e-12
    curves[(r["code"], r["period"], str(r["statistic"]))] = (im_vals[m], lam[m])

def im_at_lambda(code, per, lam_t, stat="mean"):
    x, y = curves[(code, per, stat)]
    return float(np.exp(np.interp(np.log(lam_t), np.log(y[::-1]), np.log(x[::-1]))))

def lam_at_im(code, per, im_t, stat="mean"):
    x, y = curves[(code, per, stat)]
    return float(np.exp(np.interp(np.log(im_t), np.log(x), np.log(y))))

SITES = ["AKL", "WLG", "CHC", "DUD"]
scol = {"AKL": "#0072B2", "WLG": "#D55E00", "CHC": "#009E73", "DUD": "#CC79A7"}
sname = {"AKL": "Auckland", "WLG": "Wellington", "CHC": "Christchurch", "DUD": "Dunedin"}
mk = {"AKL": "o", "WLG": "s", "CHC": "^", "DUD": "D"}
PERS_ALL = ["PGA"] + [f"SA({t})" for t in ["0.1","0.2","0.3","0.4","0.5","0.7","1.0","1.5","2.0","3.0","4.0","5.0","6.0","7.5","10.0"]]
T_ALL = [0.01] + [0.1,0.2,0.3,0.4,0.5,0.7,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.5,10.0]
IMS8 = ["PGA", "SA(0.2)", "SA(0.5)", "SA(1.0)", "SA(2.0)", "SA(3.0)", "SA(5.0)", "SA(10.0)"]
sls8 = {"PGA": "-", "SA(0.2)": (0, (6, 1.5)), "SA(0.5)": "--", "SA(1.0)": "-.",
        "SA(2.0)": (0, (3, 1, 1, 1)), "SA(3.0)": ":", "SA(5.0)": (0, (1, 1)),
        "SA(10.0)": (0, (4, 1, 1, 1, 1, 1))}

def rp_cals(code, per, rp_uls, k, stat="mean"):
    return 1.0 / lam_at_im(code, per, k * im_at_lambda(code, per, 1.0/rp_uls, stat), stat)

# =============== 1. Figure 1 with fractile bands (k parameterised) ===============
def make_fig1(k, fname):
    rp_uls = 2500
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    lam_uls = 1.0/rp_uls
    xlim, ylim = (0.04, 6.0), (1e-5, 1e-2)
    info = []
    for code, name, col in [("WLG", "Wellington", "k"), ("AKL", "Auckland", "#0072B2")]:
        x, y = curves[(code, "PGA", "mean")]
        x10, y10 = curves[(code, "PGA", "0.1")]
        x90, y90 = curves[(code, "PGA", "0.9")]
        m = (y > ylim[0]*0.5) & (y < ylim[1]*2) & (x > xlim[0]*0.5) & (x < xlim[1]*2)
        # fractile band: interpolate 0.1/0.9 curves onto mean-curve IM grid
        lam10 = np.exp(np.interp(np.log(x[m]), np.log(x10), np.log(y10)))
        lam90 = np.exp(np.interp(np.log(x[m]), np.log(x90), np.log(y90)))
        ax.fill_between(x[m], lam10, lam90, color=col, alpha=0.13, lw=0, zorder=1)
        ax.loglog(x[m], y[m], "-", color=col, lw=1.4, label=name, zorder=3)
        im_uls = im_at_lambda(code, "PGA", lam_uls)
        im_cals = k*im_uls
        lam_cals = lam_at_im(code, "PGA", im_cals)
        info.append((name, col, im_uls, im_cals, lam_cals))
        ax.plot([im_uls, im_uls], [ylim[0], lam_uls], color=col, lw=0.7, ls=":", zorder=3)
        ax.plot([im_cals, im_cals], [ylim[0], lam_cals], color=col, lw=0.7, ls=":", zorder=3)
        ax.plot([xlim[0], im_cals], [lam_cals, lam_cals], color=col, lw=0.7, ls=":", zorder=3)
        ax.annotate("", xy=(im_cals, 1.5e-5), xytext=(im_uls, 1.5e-5),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=col, shrinkA=0, shrinkB=0))
        ax.plot(im_uls, lam_uls, "o", color=col, ms=4.5, mfc="w", mew=1.2, zorder=4)
        ax.plot(im_cals, lam_cals, "s", color=col, ms=4.5, zorder=4)
    ax.plot([xlim[0], max(i[3] for i in info)], [lam_uls, lam_uls], color="0.3", lw=0.7, ls=":")
    ax.annotate(f"$\\lambda_{{ULS}}=1/{rp_uls}$", xy=(xlim[0]*1.1, lam_uls*1.12), fontsize=7.5, color="0.25")
    lam_sorted = sorted(info, key=lambda i: -i[4])
    (nA, cA, _, _, lA), (nB, cB, _, _, lB) = lam_sorted
    ax.annotate(f"$\\lambda_{{CALS}}=1/{1/lA:.0f}$", xy=(xlim[0]*1.1, lA*1.10), fontsize=7.5, color=cA, va="bottom")
    ax.annotate(f"$\\lambda_{{CALS}}=1/{1/lB:.0f}$", xy=(xlim[0]*1.1, lB*0.90), fontsize=7.5, color=cB, va="top")
    ax.annotate(f"$m={k}$", xy=(min(i[2] for i in info)*0.92, 1.5e-5), ha="right", va="center",
                fontsize=8.5, color="0.25")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("Peak ground acceleration, $PGA$ (g)")
    ax.set_ylabel("Annual exceedance rate, $\\lambda$")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.4, which="major")
    fig.tight_layout(pad=0.3)

    fig.savefig(f"{FIGD}/{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTD}/{fname}.png", bbox_inches="tight", dpi=160)
    for (name, col, imu, imc, lamc) in info:
        print(f"fig1 k={k} {name}: RP_CALS={1/lamc:.0f}")

make_fig1(1.3, "fig_concept_frac")
make_fig1(1.5, "fig_concept_frac_k15")

# =============== 2. Figure 2 v3: 8 IMs, IL4 at k=1.5 ===============
CASES_V3 = [("IL2", 500, 1.5), ("IL3", 1000, 1.5), ("IL4", 2500, 1.5)]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
for ax, (case, rp, k) in zip(axes, CASES_V3):
    lam_uls = 1.0/rp
    ratios = []
    for code in SITES:
        for im in IMS8:
            im_uls = im_at_lambda(code, im, lam_uls)
            x, y = curves[(code, im, "mean")]
            xn, yn = x/im_uls, y/lam_uls
            m = (xn > 0.45) & (xn < 3.2) & (yn > 1e-3)
            ax.plot(xn[m], yn[m], color=scol[code], ls=sls8[im], lw=0.9, alpha=0.85)
            ratios.append(rp_cals(code, im, rp, k)/rp)
    lo, hi = 1/max(ratios), 1/min(ratios)
    ax.plot([k, k], [1e-3, 10], color="k", lw=0.8)
    ax.plot([0.4, k], [lo, lo], color="0.4", lw=0.6)
    ax.plot([0.4, k], [hi, hi], color="0.4", lw=0.6)
    ax.axhspan(lo, hi, color="0.85", alpha=0.5, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 3.0); ax.set_ylim(3e-2, 8)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1.0, 1.5, 2.0, 3.0]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1.0", "1.5", "2.0", "3.0"]))
    ax.annotate(f"$\\lambda_{{CALS}}/\\lambda_{{ULS}}$\n{lo:.2f}–{hi:.2f}",
                xy=(0.55, lo*0.9), fontsize=7.5, va="top")
    ax.set_title(f"{case}: $RP_{{ULS}}$={rp} yr, $m$={k}", fontsize=9)
    print(f"fig2v3 {case}: lambda ratio {lo:.2f}-{hi:.2f}, RP ratio {min(ratios):.2f}-{max(ratios):.2f}")
h1 = [plt.Line2D([], [], color=scol[c], lw=1.4, label=sname[c]) for c in SITES]
h2 = [plt.Line2D([], [], color="0.3", ls=sls8[i], lw=1.1, label=i) for i in IMS8]
axes[0].legend(handles=h1, fontsize=6.5, loc="upper right", framealpha=0.95, handlelength=1.8)
axes[1].legend(handles=h2, fontsize=5.8, loc="upper right", framealpha=0.95, handlelength=2.6)
axes[0].set_ylabel("Exceedance rate ratio, $\\lambda/\\lambda_{ULS}$")
axes[1].set_xlabel("Intensity measure ratio, $IM/IM_{ULS}$")
fig.tight_layout(pad=0.4)
fig.savefig(f"{FIGD}/fig_normalized_hazard_v3.pdf", bbox_inches="tight")
fig.savefig(f"{OUTD}/fig_normalized_hazard_v3.png", bbox_inches="tight", dpi=160)

# =============== 3/4. Figure 3 v3: full T set, IL4 k=1.5; log and linear axes ===============
TS26 = {"IL2": 2.0, "IL3": 2.5}
res = {}
for case, rp, k in CASES_V3:
    for code in SITES:
        res[(case, code)] = [rp_cals(code, p, rp, k)/rp for p in PERS_ALL]

for tag, logx in [("", True), ("_linT", False)]:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for iax, (ax, (case, rp, k)) in enumerate(zip(axes, CASES_V3)):
        ax.axhspan(2.0, 3.0, color="0.88", alpha=0.65, zorder=0)
        for code in SITES:
            ax.plot(T_ALL, res[(case, code)], mk[code]+"-", color=scol[code], ms=3.2,
                    lw=0.8, label=sname[code], mfc="white", mew=0.9)
        if iax == 0:
            ax.annotate("Recommended\nrange", xy=(5.0 if not logx else 3.0, 2.8),
                        xytext=(5.0 if not logx else 3.0, 4.1), fontsize=7.5,
                        color="0.3", ha="center", va="center",
                        arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"))
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(0.008, 12)
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.01, 0.1, 0.3, 1.0, 3.0, 10.0]))
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["PGA", "0.1", "0.3", "1", "3", "10"]))
        else:
            ax.set_xlim(-0.3, 10.3)
            ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_title(f"{case}: $RP_{{ULS}}$={rp} yr, $m$={k}", fontsize=9)
        ax.set_ylim(1.0, 5.1)
    axes[0].set_ylabel("Return period ratio, $RP_{CALS}/RP_{ULS}$")
    axes[1].set_xlabel("Period, $T$ (s)")
    axes[2].legend(fontsize=7, loc="upper right", framealpha=0.95)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{FIGD}/fig_rp_ratio_summary_v4{tag}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTD}/fig_rp_ratio_summary_v4{tag}.png", bbox_inches="tight", dpi=160)
for case, rp, k in CASES_V3:
    allr = np.array([res[(case, c)] for c in SITES])
    print(f"fig3v3 {case} (k={k}): RP ratio {allr.min():.2f}-{allr.max():.2f} "
          f"(RP {rp*allr.min():.0f}-{rp*allr.max():.0f} yr)")

# =============== 5. Appendix figure: IL4 with k=1.3 (two panels) ===============
case, rp, k = "IL4", 2500, 1.3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
lam_uls = 1.0/rp
ratios = []
for code in SITES:
    for im in IMS8:
        im_uls = im_at_lambda(code, im, lam_uls)
        x, y = curves[(code, im, "mean")]
        xn, yn = x/im_uls, y/lam_uls
        m = (xn > 0.45) & (xn < 3.2) & (yn > 1e-3)
        ax1.plot(xn[m], yn[m], color=scol[code], ls=sls8[im], lw=0.9, alpha=0.85)
        ratios.append(rp_cals(code, im, rp, k)/rp)
lo, hi = 1/max(ratios), 1/min(ratios)
ax1.plot([k, k], [1e-3, 10], color="k", lw=0.8)
ax1.plot([0.4, k], [lo, lo], color="0.4", lw=0.6)
ax1.plot([0.4, k], [hi, hi], color="0.4", lw=0.6)
ax1.axhspan(lo, hi, color="0.85", alpha=0.5, zorder=0)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlim(0.5, 3.0); ax1.set_ylim(3e-2, 8)
ax1.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1.0, 1.5, 2.0, 3.0]))
ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax1.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1.0", "1.5", "2.0", "3.0"]))
ax1.annotate(f"$\\lambda_{{CALS}}/\\lambda_{{ULS}}$\n{lo:.2f}–{hi:.2f}", xy=(0.55, lo*0.9),
             fontsize=7.5, va="top")
ax1.set_title(f"$RP_{{ULS}}$=2500 yr, $m$=1.3", fontsize=9)
ax1.set_xlabel("Intensity measure ratio, $IM/IM_{ULS}$")
ax1.set_ylabel("Exceedance rate ratio, $\\lambda/\\lambda_{ULS}$")
h2 = [plt.Line2D([], [], color="0.3", ls=sls8[i], lw=1.1, label=i) for i in IMS8]
ax1.legend(handles=h2, fontsize=5.6, loc="upper right", framealpha=0.95, handlelength=2.6)

for code in SITES:
    vals = [rp_cals(code, p, rp, k)/rp for p in PERS_ALL]
    ax2.plot(T_ALL, vals, mk[code]+"-", color=scol[code], ms=3.2, lw=0.8,
             label=sname[code], mfc="white", mew=0.9)
ax2.set_xlim(-0.3, 10.3)
ax2.set_xticks([0, 2, 4, 6, 8, 10])
ax2.set_ylim(1.0, 5.1)
ax2.set_title(f"$RP_{{ULS}}$=2500 yr, $m$=1.3", fontsize=9)
ax2.set_xlabel("Period, $T$ (s)")
ax2.set_ylabel("Return period ratio, $RP_{CALS}/RP_{ULS}$")
ax2.legend(fontsize=7, loc="upper right", framealpha=0.95)
fig.tight_layout(pad=0.5)
fig.savefig(f"{FIGD}/fig_appendix_IL4k13_v2.pdf", bbox_inches="tight")
fig.savefig(f"{OUTD}/fig_appendix_IL4k13_v2.png", bbox_inches="tight", dpi=160)
print("appendix IL4 k=1.3 figure saved")
