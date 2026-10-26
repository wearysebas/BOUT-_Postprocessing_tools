
---

## Appendix A. `hermes-drifts-0.py`: the §10 plots as built (2026-10-20)

Script: `~/Dev/My_working_scripts/Hermes-3_pp/hermes-drifts-0.py`. It imports
`Hermes-3_postprocessing.py` (polygon maps, overlays), `hermes-heat-pp-method-comparison.py`
(`load_run`, guarded-frame target faces, `method1_energy_flow`,
`sheath_potential_check`) and `SF_heat_transport_pp_xhermes_withfluxes.py`
(`discover_cases`) by path.

### A.1 Running it

Same calling convention as `hermes-heat-pp-method-comparison.py`: a folder of
`SF<angle>`/`SN` runs plus a grid folder. `discover_cases` pairs each run with
its grid.

```bash
R=/Users/sruiz/Dev/PhD/SF_Sims/Modified_SF_Runs
G=$R/grids/Hermes-3/sf-new-limiter

# drift runs, every tier, matched no-drift runs for Tier 4 / overlays
python hermes-drifts-0.py $R/CVODE_test $G --compare $R/No_Drifts

# one case, convergence only, on screen
python hermes-drifts-0.py $R/CVODE_test $G --only SF45 --tiers 0 --show

# the no-drift runs on their own
python hermes-drifts-0.py $R/No_Drifts $G --drifts False
```

| Option | Default | Meaning |
|---|---|---|
| `--only SF45 ...` | all | case folders to load |
| `--drifts True/False` | `True` | picks the output folder; checked against `BOUT.settings` (warns if `relax_potential`/`vorticity` presence disagrees) |
| `--compare DIR` | none | second runs folder, paired case by case by folder name (`CVODE_test/SF45` ↔ `No_Drifts/SF45`) |
| `--label`, `--compare_label` | `drifts` / `no drifts` (swapped with `--drifts False`) | legend labels |
| `--tiers` | `0,1,2,3,4` | tiers to draw |
| `--twindow` | 5 | maps/profiles are averages over the last N outputs |
| `--seps`, `--branch_cuts` | `True` | overlays on every polygon map |
| `--outdir` | `.../Modified_SF_Runs/Post_Processing` | see A.2 |

### A.2 Output layout

```
Post_Processing/Drifts/            (No_drifts/ with --drifts False)
  T3_all_cases.png                 all-case summary (Tier 3)
  SF45/  T0_*.png T1_*.png T2_*.png T3_strike_points.png T4_*.png
  SF75/  ...
```

### A.3 What each tier now contains

- **Tier 0**: unchanged (φ at probes, max/rms |dφ/dt|, target flows, totals).
- **Tier 1**: maps of φ, φ/T_e, **E_r, E_pol, v_E,r, v_E,pol** (new), n·v_E,r,
  |Γ_E×B|/|Γ_anom|, j∥, **∇·J_dia** (now reconstructed, A.4), n_e, T_e, all
  with separatrices and branch cuts overlaid. The 1D radial and poloidal
  cuts of the components are kept.
- **Tier 2**: OMP radial profiles; per-target profiles, now with **raw φ** next
  to φ/T_e; along-ring profiles.
- **Tier 3**: power and ion flow per strike point; **full current budget** (A.5);
  sheath-potential check; **all-case summary** of power and ion flow per strike
  point, drifts vs no drifts, for every case (SN fills SP1/SP2 only).
- **Tier 4** (needs `--compare`): **side-by-side n_e and T_e** on a shared colour
  scale; Δ maps of T_e, **T_i**, n_e (relative), **M∥, j∥**, φ (only if both
  runs have it); **Δ target profiles** (n_e, T_e, Γ∥, q∥) per strike point.

### A.4 Quantities rebuilt from the dump

Neither `pf<ion>_tot_xlow` nor `DivJdia` is in our dumps (§9 item 3:
`[relax_potential] diagnose` is off). The script rebuilds both the way the
Hermes source computes them:

- **Anomalous radial particle flow** (for the ratio map), from the x-face flow of
  `Div_a_Grad_perp_upwind_flows` (`src/div_ops.cxx`):

  flow_{i+½} = −[(J g¹¹)_i + (J g¹¹)_{i+1}] (N_{i+1} − N_i)/(dx_i + dx_{i+1}) · D_upwind · dy · dz

  It is averaged to cell centres before dividing |n v^x J dy dz| by it.
  Note that E×B advection in `evolve_density` never writes `particle_flow_xlow`.
- **Diamagnetic current**, as `RelaxPotential::calculateDivJdia`:

  J_dia = Σ_s P_s Curlb_B, with Curlb_B^{x,y} = 2 bxcv{x,y}/B (SI; grid `bxcvx`, `bxcvy`)

  and ∇·J_dia = [∂_x(J J_dia^x) + ∂_y(J J_dia^y)]/J, which is BOUT's
  `Div(Vector3D)` with ∂_z = 0. The sum runs over e and the positive ions;
  neutrals have charge 0. `sinty` only enters Curlb_B^z, which drops out in 2D.

### A.5 Current budget (Tier 3)

Current leaving the domain, which should sum to ≈ 0 once φ has converged
(polarisation current ignored):

- j∥ into each target: e(Σ Z n_i V_i − n_e V_e) at the sheath face × dA_y;
- J_dia through each target face: J_dia^y J dx dz (P at the face, as the
  domain/guard average);
- J_dia out through the inner (core + PF) and outer (wall) x boundaries:
  J_dia^x J dy dz at the x-face.

Check: with the y-connectivity of A.6, ∫ ∇·J_dia dV equals the sum of the
J_dia boundary terms (SF45: 119.6 A vs 118.8 A). The residual is the
one-sided difference at the targets.

### A.6 y-derivatives across branch cuts

y-derivatives (E_pol, v_E,r, ∇·J_dia) used to go one-sided at every branch
cut, which puts a fake boundary at each X-point. Now the y-neighbour of each
cell comes from the gridue corner mesh: the cell across a face is the one
that shares both of that face's corners, on the same x ring. With that
neighbour, the centred difference also works across a cut, as BOUT's
communication does. Only the targets stay one-sided. If a face has no exact
corner match, the nearest face whose corners agree to within 1% of the face
length is accepted.

That tolerance is needed for **SF45**: two faces at the secondary X-point,
(x=17, j=47|64) and (x=18, j=63|64), have an X-point corner that is 0.05 mm
off between the two cells. BOUT never reads corners, so this does not affect
the simulation, but it is a wart in the gridue/INGRID output.

### A.7 First results on CVODE_test vs No_Drifts (all cases)

1. **No anomalous particle or heat transport in either set.** Every run dumps
   `anomalous_D_*` = `anomalous_Chi_*` = 0. Only `anomalous_nu_d+` = 2 is set,
   and the grids carry no `D_*`/`chi_*`. Cross-field ion transport is E×B
   (drift runs) plus whatever the neutrals carry, and the no-drift runs have
   none. The |Γ_E×B|/|Γ_anom| map is therefore skipped. **Check whether this
   was intended** before comparing to SOLPS or older runs.
2. **The drift runs are not charge-balanced yet.** The net current out of the
   domain is about −160 to −1000 A, the same size as Σ|I∥|. The electrons
   reach the targets at v_e/c_s ≈ 5–11, because φ_sh/T_e ≈ 0–0.16 against an
   ambipolar ≈ 1.6–2.3. At the last step, max |dφ/dt| is still 3–9×10⁵ V/s
   across the cases. Consistent with §9 item 3: φ has not converged, so the
   drift-vs-no-drift differences are not final.
3. **SF165** is the outlier: 34 kW to the targets with drifts against 2.3 kW
   without, with T_e ≈ 7.5 eV at SP1 in the drift run against ≈ 1 eV
   elsewhere. Check both runs reached comparable states before reading
   anything into it.
