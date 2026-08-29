# On the hazard-consistency of Collapse Avoidance Limit State (CALS) ground motion

Data and code to reproduce all figures and tables in:

> Bradley BA. On the hazard-consistency of Collapse Avoidance Limit State (CALS)
> ground motion. *Bulletin of the New Zealand Society for Earthquake Engineering*
> (in preparation).

## Repository structure

```
data/
  hazard/     Mean and fractile hazard curves from the 2022 NZ NSHM (v1.0.4) for
              Auckland, Wellington, Christchurch, and Dunedin (Vs30 = 250 m/s),
              a site lookup table, and an independently computed summary of CALS
              return periods used for verification.
  disagg/     NSHM disaggregation results for Wellington and Christchurch
              (Vs30 = 400 m/s), for SA(0.5) and SA(3.0), at 10% and 2% in
              50 year exceedance probabilities.
scripts/
  figures_1_2_3_appendix.py          Figure 1 (hazard-curve concept with 0.1-0.9
                                     fractile bands; m=1.5 and m=1.3 variants),
                                     Figure 2 (normalised hazard curves, 8 IMs),
                                     Figure 3 (implied return period ratios,
                                     full period set, linear- and log-axis
                                     variants), and the Appendix m=1.3 figure.
  cms_and_disagg_figures_4_5.py      Figure 4 (approximate CMS via the Baker &
                                     Jayaram (2008) correlation model) and
                                     Figure 5 (trivariate disaggregation
                                     comparisons; SA(3.0) supplementary variant).
  table2_values_and_verification.py  Table 2 values (implied CALS return
                                     periods, m=1.5, eight IMs) and verification
                                     against the independent summary.
figures/    Output figures (PDF, as included in the paper; PNG copies in output/).
output/     Derived CSV results, LaTeX table rows, and PNG previews.
```

Figure files used in the paper: `fig_concept_frac_k15` (Fig. 1),
`fig_normalized_hazard_v3` (Fig. 2), `fig_rp_ratio_summary_v4_linT` (Fig. 3),
`fig_cms_m` (Fig. 4), `fig_disagg_SA0p5` (Fig. 5), `fig_appendix_IL4k13_v2`
(Fig. 6, Appendix). Remaining outputs are supplementary variants.

## Requirements and usage

Python >= 3.9 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
python scripts/figures_1_2_3_appendix.py
python scripts/cms_and_disagg_figures_4_5.py
python scripts/table2_values_and_verification.py
```

Each script is standalone and writes into `figures/` and `output/`.

## Data provenance

Hazard curves and disaggregation results were downloaded from the 2022 NZ
National Seismic Hazard Model web services (https://nshm.gns.cri.nz; NSHM
v1.0.4, mean hazard and fractiles; last accessed July 2026). See:

> Gerstenberger MC, et al. (2024). The 2022 Aotearoa New Zealand National
> Seismic Hazard Model: Process, overview, and results. *Bulletin of the
> Seismological Society of America* 114(1): 7-36.
> https://doi.org/10.1785/0120230182

## License

MIT (see `LICENSE`) — free for anyone to use, modify, and redistribute with
attribution. The underlying hazard and disaggregation data are from the
publicly available 2022 NZ National Seismic Hazard Model (see Data provenance).
