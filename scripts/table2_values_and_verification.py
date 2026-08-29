"""Table 2 values (implied CALS return periods, m=1.5, eight IMs) and
verification of the interpolation workflow against the independently
computed summary in data/hazard/cals_rp_summary_mean.csv (which used
k=1.5 for IL2/IL3 and k=1.3 for IL4).

Outputs: output/table2_values.csv, output/table2_rows.tex, and a printed
verification report.
"""
import numpy as np, pandas as pd, re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
OUTD = _REPO / "output"
OUTD.mkdir(exist_ok=True)

hz = pd.read_csv(_REPO / "data/hazard/hazard-curves.csv", skiprows=1)
sites = pd.read_csv(_REPO / "data/hazard/site_lookup.csv")
im_cols = [c for c in hz.columns if c.startswith("annual poe")]
im_vals = np.array([float(re.search(r"- ([\d.]+) g", c).group(1)) for c in im_cols])
hz = hz[hz["statistic"] == "mean"].copy()
key = {(f"{r.lat:.3f}", f"{r.lon:.3f}"): r.code for r in sites.itertuples()}
hz["code"] = [key[(f"{la:.3f}", f"{lo:.3f}")] for la, lo in zip(hz.lat, hz.lon)]
curves = {}
for _, r in hz.iterrows():
    lam = r[im_cols].astype(float).values
    ok = lam > 1e-12
    curves[(r["code"], r["period"])] = (im_vals[ok], lam[ok])

def im_at_lambda(c, p, l):
    x, y = curves[(c, p)]
    return float(np.exp(np.interp(np.log(l), np.log(y[::-1]), np.log(x[::-1]))))

def lam_at_im(c, p, i):
    x, y = curves[(c, p)]
    return float(np.exp(np.interp(np.log(i), np.log(x), np.log(y))))

def rp_cals(c, p, rp, m):
    return 1.0 / lam_at_im(c, p, m * im_at_lambda(c, p, 1.0 / rp))

SN = {"AKL": "Auckland", "WLG": "Wellington", "CHC": "Christchurch", "DUD": "Dunedin"}
IMS8 = ["PGA", "SA(0.2)", "SA(0.5)", "SA(1.0)", "SA(2.0)", "SA(3.0)", "SA(5.0)", "SA(10.0)"]
CASES = [("IL2", 500), ("IL3", 1000), ("IL4", 2500)]  # m = 1.5 throughout (draft3 basis)

rows, tex = [], []
for code in ["AKL", "WLG", "CHC", "DUD"]:
    for j, im in enumerate(IMS8):
        cells = []
        for case, rp in CASES:
            r = rp_cals(code, im, rp, 1.5)
            rows.append(dict(city=SN[code], im=im, case=case, RP_ULS=rp, m=1.5,
                             RP_CALS=round(r, 1), ratio=round(r / rp, 3)))
            cells.append(f"{r:,.0f} ({r/rp:.2f})")
        name = SN[code] if j == 0 else ""
        tex.append(f"{name} & {im} & " + " & ".join(cells) + " \\\\")
    tex.append("\\midrule")
pd.DataFrame(rows).to_csv(OUTD / "table2_values.csv", index=False)
(OUTD / "table2_rows.tex").write_text("\n".join(tex[:-1]))
print("table2 values written (m=1.5, 8 IMs, 3 cases)")

# ---- verification against the independent summary (k=1.5 IL2/IL3; k=1.3 IL4) ----
chk = pd.read_csv(_REPO / "data/hazard/cals_rp_summary_mean.csv")
errs = []
for _, r in chk.iterrows():
    calc = rp_cals(r["code"], r["imt"], r["RP_ULS_yr"], r["k_CALS"])
    errs.append(abs(calc - r["RP_CALS_yr"]) / r["RP_CALS_yr"])
print(f"verification vs cals_rp_summary_mean.csv: n={len(errs)}, "
      f"max relative difference = {max(errs):.2e}")
