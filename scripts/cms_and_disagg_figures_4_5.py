"""Q2: UHS vs approximate CMS (Baker & Jayaram 2008 correlation); Q3: disaggregation."""
import numpy as np, pandas as pd, re, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

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

# ================= hazard curves (as q1) =================
hz = pd.read_csv(f"{ROOT}/data/hazard/hazard-curves.csv", skiprows=1)
sites = pd.read_csv(f"{ROOT}/data/hazard/site_lookup.csv")
im_cols = [c for c in hz.columns if c.startswith("annual poe")]
im_vals = np.array([float(re.search(r"- ([\d.]+) g", c).group(1)) for c in im_cols])
hz = hz[hz["statistic"] == "mean"].copy()
key = {(f"{r.lat:.3f}", f"{r.lon:.3f}"): r.code for r in sites.itertuples()}
hz["code"] = [key[(f"{la:.3f}", f"{lo:.3f}")] for la, lo in zip(hz.lat, hz.lon)]
curves = {}
for _, r in hz.iterrows():
    lam = r[im_cols].astype(float).values
    m = lam > 1e-12
    curves[(r["code"], r["period"])] = (im_vals[m], lam[m])

PER_LABELS = ["PGA", "SA(0.1)", "SA(0.2)", "SA(0.3)", "SA(0.4)", "SA(0.5)", "SA(0.7)",
              "SA(1.0)", "SA(1.5)", "SA(2.0)", "SA(3.0)", "SA(4.0)", "SA(5.0)"]
PER_T = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

def im_at_lambda(code, per, lam_t):
    x, y = curves[(code, per)]
    return float(np.exp(np.interp(np.log(lam_t), np.log(y[::-1]), np.log(x[::-1]))))

def lambda_at_im(code, per, im_t):
    x, y = curves[(code, per)]
    return float(np.exp(np.interp(np.log(im_t), np.log(x), np.log(y))))

def uhs(code, lam):
    return np.array([im_at_lambda(code, p, lam) for p in PER_LABELS])

# ================= Baker & Jayaram (2008) correlation =================
def bj08(t1, t2):
    tmin, tmax = min(t1, t2), max(t1, t2)
    c1 = 1 - np.cos(np.pi/2 - 0.366*np.log(tmax/max(tmin, 0.109)))
    c2 = 1 - 0.105*(1 - 1/(1 + np.exp(100*tmax - 5)))*((tmax - tmin)/(tmax - 0.0099)) if tmax < 0.2 else 0.0
    c3 = c2 if tmax < 0.109 else c1
    c4 = c1 + 0.5*(np.sqrt(c3) - c3)*(1 + np.cos(np.pi*tmin/0.109))
    if tmax < 0.109: return c2
    if tmin > 0.109: return c1
    if tmax < 0.2:  return min(c2, c4)
    return c4

# ================= disaggregation =================
def load_disagg(path):
    with open(path) as f:
        hdr = f.readline()
    im_target = float(re.search(r"([\d.]+e[+-]\d+)g", hdr).group(1))
    df = pd.read_csv(path, skiprows=1)
    df.columns = ["M", "R", "TRT", "eps", "apoe", "contrib"]
    df = df[df.contrib > 0]
    return df, im_target

def dmeans(df):
    w = df.contrib / df.contrib.sum()
    return (w*df.M).sum(), (w*df.R).sum(), (w*df.eps).sum()

DIS = {}
for f in glob.glob(f"{ROOT}/data/disagg/disagg_*.csv"):
    b = os.path.basename(f).replace(".csv", "")
    _, site, vs, imt, lvl = b.split("_")
    df, imt_g = load_disagg(f)
    DIS[(site, imt, lvl)] = (df, imt_g)

print("=== disagg means (M, R, eps) and target IM ===")
stats = {}
for k in sorted(DIS):
    df, img = DIS[k]
    m, r, e = dmeans(df)
    stats[k] = (m, r, e, img)
    print(k, f"M={m:.2f} R={r:.1f} eps={e:.2f} IM={img:.3f}g")

# ================= Q2: CMS construction =================
# eps_bar interpolation in ln(RP) between 475 (10/50) and 2475 (2/50)
def epsbar(site, imt, rp):
    e1 = stats[(site, imt, "10p50")][2]; e2 = stats[(site, imt, "2p50")][2]
    return float(np.interp(np.log(rp), [np.log(475.0), np.log(2475.0)], [e1, e2]))

SIG = 0.65  # representative total sigma_lnSa (assumption; see paper text)

def cms(code, dsite, tstar_lab, tstar, rp_target):
    """approx CMS anchored to UHS at rate 1/rp_target, conditioned at T*"""
    lam = 1.0/rp_target
    u = uhs(code, lam)
    eb = epsbar(dsite, tstar_lab, rp_target)
    rho = np.array([bj08(t, tstar) for t in PER_T])
    return u * np.exp((rho - 1.0) * eb * SIG), u, eb

# Wellington IL2 example: k=1.5, RP_ULS=500
code, dsite = "WLG", "WEL"
k, rp_uls = 1.5, 500.0
cases_t = [("SA0p5", 0.5, "SA(0.5)"), ("SA3p0", 3.0, "SA(3.0)")]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7))
uhs_uls = uhs(code, 1/rp_uls)
res_q2 = {}
for ax, (tl, ts, plab) in zip(axes[:2], cases_t):
    im_uls_t = im_at_lambda(code, plab, 1/rp_uls)
    lam_cals = lambda_at_im(code, plab, k*im_uls_t)
    rp_cals = 1/lam_cals
    cms_u, _, eb_u = cms(code, dsite, tl, ts, rp_uls)
    cms_c, uhs_cals, eb_c = cms(code, dsite, tl, ts, rp_cals)
    res_q2[tl] = dict(rp_cals=rp_cals, eb_u=eb_u, eb_c=eb_c,
                      cms_u=cms_u, cms_c=cms_c, uhs_cals=uhs_cals)
    ax.loglog(PER_T, uhs_uls, "k-", label="UHS, $\\lambda_{ULS}$")
    ax.loglog(PER_T, k*uhs_uls, "k--", label="$1.5\\times$UHS, $\\lambda_{ULS}$")
    ax.loglog(PER_T, cms_u, "-", color="#0072B2", label="CMS, ULS")
    ax.loglog(PER_T, cms_c, "-", color="#D55E00", label="CMS, CALS")
    ax.plot([ts], [np.interp(ts, PER_T, cms_c)], "o", color="#D55E00", ms=4, mfc="w")
    ax.plot([ts], [np.interp(ts, PER_T, cms_u)], "o", color="#0072B2", ms=4, mfc="w")
    ax.set_title(f"$T^*={ts}$ s", fontsize=9)
    ax.set_xlim(0.05, 5); ax.set_ylim(0.05, 8)
    ax.set_xticks([0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.1", "0.3", "1", "3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel("Spectral acceleration, $Sa$ (g)")
axes[0].legend(fontsize=6.5, loc="lower left", framealpha=0.9)

ax = axes[2]
for (tl, ts, plab), col in zip(cases_t, ["#D62728", "#1B7837"]):
    ratio = res_q2[tl]["cms_c"] / res_q2[tl]["cms_u"]
    ax.semilogx(PER_T, ratio, "-", color=col, label=f"$T^*={ts}$ s")
    ax.plot([ts], [np.interp(ts, PER_T, ratio)], "o", color=col, ms=4, mfc="w")
ax.axhline(k, color="0.55", ls="--", lw=0.9)
ax.annotate("$m=1.5$", xy=(0.06, 1.51), fontsize=8, color="0.35")
ax.set_ylabel("CMS$_{CALS}$ / CMS$_{ULS}$")
axes[1].set_xlabel("Period, $T$ (s)")
ax.set_xlim(0.05, 5); ax.set_ylim(1.0, 1.7)
ax.set_xticks([0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.1", "0.3", "1", "3"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
fig.tight_layout(pad=0.4)
fig.savefig(f"{FIGD}/fig_cms_m.pdf", bbox_inches="tight")
fig.savefig(f"{OUTD}/fig_cms_m.png", bbox_inches="tight", dpi=160)

for tl in res_q2:
    d = res_q2[tl]
    print(f"Q2 {tl}: RP_CALS={d['rp_cals']:.0f}, eps_ULS={d['eb_u']:.2f}, eps_CALS={d['eb_c']:.2f}")

# ================= Q3: trivariate disagg figure =================
def trivar_panel(ax, df, title, zmax=None, zstep=None):
    Mb = np.arange(5.0, 9.4, 0.2); Rb = np.array([0,5,10,15,20,30,40,50,60,80,100,140,180,220,260,320,380,500])
    eps_groups = [(-np.inf, 0, "#2166ac"), (0, 1, "#92c5de"), (1, 2, "#f4a582"), (2, np.inf, "#b2182b")]
    # aggregate contributions
    df = df.copy()
    df = df[(df.R <= 180) & (df.contrib >= 0.03)]
    piv = {}
    for (lo, hi, c) in eps_groups:
        sub = df[(df.eps > lo) & (df.eps <= hi)]
        g = sub.groupby(["M", "R"]).contrib.sum()
        piv[(lo, hi)] = g
    allMR = sorted(set().union(*[set(p.index) for p in piv.values()]))
    dx, dyfrac = 0.16, 0.6
    for (M, R) in allMR:
        z0 = 0.0
        for (lo, hi, c) in eps_groups:
            v = piv[(lo, hi)].get((M, R), 0.0)
            if v <= 0: continue
            # R bin width for dy
            i = np.searchsorted(Rb, R, side="right")
            dy = (Rb[min(i, len(Rb)-1)] - Rb[i-1]) * dyfrac if i < len(Rb) else 20
            ax.bar3d(M-dx/2, R, z0, dx, dy, v, color=c, shade=True,
                     edgecolor="0.4", linewidth=0.15, alpha=1.0)
            z0 += v
    ax.set_xlabel("Magnitude, $M_w$", fontsize=7.5, labelpad=-2)
    ax.set_ylabel("Distance, $R_{rup}$ (km)", fontsize=7.5, labelpad=0)
    ax.set_zlabel("% contribution", fontsize=7, labelpad=-4)
    ax.text2D(0.5, 0.92, title, transform=ax.transAxes, ha="center", va="top",
              fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_xlim(5, 9.6); ax.set_ylim(0, 180)
    ax.set_xticks([5, 6, 7, 8, 9])
    ax.set_yticks([0, 40, 80, 120, 160])
    if zmax is not None:
        ax.set_zlim(0, zmax)
        ax.set_zticks(np.arange(0, zmax + 0.01, zstep))
    ax.view_init(elev=28, azim=-60)
    ax.tick_params(labelsize=6.5, pad=-1)

def stackmax(df):
    d = df[(df.R <= 180) & (df.contrib >= 0.03)]
    return d.groupby(["M", "R"]).contrib.sum().max()

for imt, tag in [("SA0p5", "0.5"), ("SA3p0", "3.0")]:
    fig = plt.figure(figsize=(7.0, 5.6))
    panels = [("WEL", "10p50", "Wellington"), ("WEL", "2p50", "Wellington"),
              ("CHC", "10p50", "Christchurch"), ("CHC", "2p50", "Christchurch")]
    # common z-axis per site (row): same limit and ticks for both hazard levels
    zl = {}
    for s in ["WEL", "CHC"]:
        zm = max(stackmax(DIS[(s, imt, l)][0]) for l in ["10p50", "2p50"])
        step = 2 if zm > 6 else 1
        zl[s] = (np.ceil(zm / step) * step, step)
    for i, (s, lvl, nm) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i+1, projection="3d")
        df, img = DIS[(s, imt, lvl)]
        rp = "475" if lvl == "10p50" else "2475"
        m, r, e, _ = stats[(s, imt, lvl)]
        trivar_panel(ax, df, f"{nm}, RP={rp} yr\n$\\bar{{M}}$={m:.1f}, $\\bar{{R}}$={r:.0f} km, $\\bar{{\\epsilon}}$={e:.1f}",
                     zmax=zl[s][0], zstep=zl[s][1])
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for (_, _, c) in
               [(-np.inf, 0, "#2166ac"), (0, 1, "#92c5de"), (1, 2, "#f4a582"), (2, np.inf, "#b2182b")]]
    fig.legend(handles, ["$\\epsilon<0$", "$0<\\epsilon<1$", "$1<\\epsilon<2$", "$\\epsilon>2$"],
               loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.02, right=0.96, top=0.99, bottom=0.09, wspace=0.12, hspace=0.05)
    fig.savefig(f"{FIGD}/fig_disagg_SA{tag.replace('.', 'p')}.pdf",
                bbox_inches="tight", pad_inches=0.25)
    fig.savefig(f"{OUTD}/fig_disagg_SA{tag.replace('.', 'p')}.png",
                bbox_inches="tight", pad_inches=0.25, dpi=150)
print("q23 figures saved")

# stats table for paper
rows = []
for (s, imt, lvl), (m, r, e, img) in stats.items():
    rows.append(dict(site=s, imt=imt, level=lvl, Mbar=round(m, 2), Rbar=round(r, 1),
                     epsbar=round(e, 2), IM_g=round(img, 3)))
pd.DataFrame(rows).sort_values(["site", "imt", "level"]).to_csv(f"{OUTD}/q3_disagg_stats.csv", index=False)
