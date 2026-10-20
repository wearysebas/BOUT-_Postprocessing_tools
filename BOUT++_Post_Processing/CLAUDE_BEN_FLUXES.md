# Approach D — `Ben_example_fluxes`

Port of the supervisor's energy/particle balance
(`Hermes-Final/Final-Pro-Max/examples/tokamak-2D/recycling/analyse_fluxes.py`, "Ben's script")
into `SF_heat_transport_pp_xhermes_withfluxes.py` as a **new, selectable approach D**, so it can be
compared side by side with what we already had:

| Approach | Selected by | Target power from |
|---|---|---|
| A (default) | — | Energy-flow diagnostics `efe_tot_ylow + efd+_tot_ylow` |
| C | `--analytic_power` | Simple analytic sheath `γ n T Cs` on time-averaged fields |
| **D (new)** | `--ben_example_fluxes` (`--ben_strict` for Ben's exact formulas) | Ben's sheath reconstruction (Hermes-exact), plus full power & particle balance |

Reference run used to check this: `SF_Sims/Modified_SF_Runs/No_Drifts/SF75`
(SF+ LFS, nx=34 incl. guards, ny=72, ixseps1=18, ixseps2=16, jyseps = 7/39/47/63, ny_inner=56,
MXG=MYG=2, γe=4.5, γi=2.5, Neumann on every x-boundary, core power via `Pe_src`/`Pd+_src`).

Hermes source referenced below: `Hermes-Final/hermes-3` (the source of the Final-Pro-Max build).

---

## 1. What Ben's script computes

1. Radial (x) power through the inner (`x=2`) and outer (`x=-3`) boundary, from anomalous
   diffusion (`chi`, `D`), summed over all θ.
2. Sheath power and particle flux at the lower (`θ=0`) and upper (`θ=-1`) targets.
3. Radiation: `Rd+_ex`, `Rd+_rec`.
4. `Pnet = Pin − Pout,x − Pout,d − Pout,u − Prad`.
5. Particle balance: radial in/out, sheath flux, recycling fraction, ionisation and recombination totals.
6. Rate of change of stored energy (internal and kinetic).

---

## 2. SF adaptations (needed, not disagreements)

### 2.1 Four targets instead of two
In the no-y-guard frame (`keep_yboundaries=False`):

| Target | θ index | Boundary side |
|---|---|---|
| SP1 NW | `0` | lower |
| SP2 NE | `ny_inner − 1` | upper |
| SP3 SE | `ny_inner` | lower |
| SP4 SW | `ny − 1` | upper |

SN keeps only `0` (lower) and `ny − 1` (upper). The targets are chosen from `mesh_topology`.

### 2.2 The inner x-boundary is split into core and PF
At `x = xstart` the column covers both core and private-flux regions (W_PFR/E_PFR), so summing
the whole column (as Ben does for SN) mixes the core input with the PF loss. The core y-range comes from the dumped
`snowflake_type` (same as the MYSUB checks in `boutmesh.cxx`):

| `snowflake_type` | Core y-range at inner x-boundary |
|---|---|
| SN, `sf_plus_low_field_side`, `SF` | `(jyseps1_1, jyseps2_1]` |
| `sf_plus_high_field_side` | `(jyseps2_1, jyseps1_2]` |
| `sf_minus_low_field_side`, `sf_minus_high_field_side` | `(jyseps1_1, jyseps2_1] ∪ (jyseps1_2, jyseps2_2]` |

Everything else at the inner boundary is PF. The outer boundary (SOL wall + south PF) is reported as one total.

### 2.3 Diffusion coefficients from the dump, not the options
Ben reads scalar `anomalous_chi`/`anomalous_D` from `bd.options` (l.255–257). Our runs set them
from the grid, so the options are absent. The dumps contain the fields `anomalous_Chi_e`,
`anomalous_Chi_d+`, `anomalous_D_e` and `anomalous_D_d+` [m²/s], which D uses per species.

### 2.4 Core power is a volumetric source
Our core boundary is Neumann, and the dumped x-guard/interior ratio is exactly 1 for Ne, Te, Td+, Pe and Nd+, so the
radial flux through the core boundary is ≈ 0 by construction. D adds

    P_core = 3/2 ∫ (Pe_src + Pd+_src) dV        [W]

(`P<name>_src` is `final_source`, `evolve_pressure.cxx:478`). The radial x-boundary terms are still
computed, as a check: they should come out ≈ 0 for these runs.

---

## 3. Disagreements between Ben's script and the Hermes source

**Rule (decided): where they disagree, D follows the Hermes source; `--ben_strict` switches back to Ben's version** (see §4, decision 8).

| # | Ben's script | Hermes source | D does |
|---|---|---|---|
| 1 | D-driven energy flow `Div_a_Grad_perp(qe·D·T, N)` (l.265, 270) | `(3/2)·T·D·∇N` (`anomalous_diffusion.cxx:170,173`) | factor 3/2 |
| 2 | Face coefficient = average `0.5(a_i + a_{i+1})` (l.77); upwind branch commented out (l.79–84) | Upwind: `a(i+1)` if gradient > 0, else `a(i)` (`div_ops.cxx:1496–1497`) | upwind |
| 3 | Electron sheath q: lower end uses `(γe − 2.5)·T − ½me v²` (l.168–170), upper end uses `γe·T − ½me v²` (l.222). Inconsistent between ends. | Total energy leaving = `γe·Te·ne·ve` (the diagnostic `q`, `sheath_boundary_simple.cxx:383,410,449,476`); `2.5T + ½mv²` is only subtracted for the part already advected by the pressure equation (l.394, 460) | `γ·T·n·v` for both species at both ends |
| 4 | Ion sheath velocity = `Cs` (l.159–163) | `max(Vi, Cs)` lower/upper (`sheath_boundary_simple.cxx:594,664`) | `max(|Vi|, Cs)` using dumped `Vd+` |
| 5 | Guard extrapolation `f² / f_next`, always (l.146–148, 210–212) | `limitFree(…, mode)` (`sheath_boundary_simple.cxx:29–51`); modes from `density_/temperature_boundary_mode` (l.130, 140), default 1 = exponential, which is the same as Ben's | `limitFree` with the modes read from `BOUT.settings` |
| 6 | Radiation × 1.5 (l.351–356) | `Rd+_ex`, `Rd+_rec` are already the electron energy loss [W/m³] (`izn_rec_reaction.cxx:180–193`; dump `units = W / m^3`) | no 1.5 factor; add `Rc` when dumped |
| 7 | Kinetic dE/dt (l.418–425) is never multiplied by the cell volume or summed, so `float()` (l.427) fails on an array. It would also `KeyError` first on `ddt(NVe)`/`ddt(Ne)`, which don't exist with `quasineutral + zero_current` | — | volume-integrate; only species with both `ddt(N…)` and `ddt(NV…)` (d+, d) |
| 8 | Recycling fraction `1 − (Γin − Γout)/Γsheath` (l.385–387) | — | keep Ben's (it is trivially 1 when x-flux = 0, as in our Neumann runs); **add** the direct `∫Sd_target_recycle dV / Γsheath` (should be ≈ `target_recycle_multiplier` = 0.99) |
| 9 | dE/dt includes neutral `d`, but the neutral wall/pump energy losses aren't in the balance, so it cannot close | `neutral_boundary` / recycling energy sources dumped as `Ed_target_recycle`, `Ed_wall_refl`, `Ed_target_refl`, `Ed_pump` [W/m³] | add `∫(Σ Ed_*) dV` as a neutral term, for whichever of these are dumped |

Electron sheath velocity: with a single ion species, Z=1, quasineutrality and `phi` not evolved,
Hermes' `ve = sqrt(Te/2πme)·exp(−φ/Te)` reduces to `ve = vi`, so D uses `ve = vi`. **This is not valid
if `phi` is evolved (drift runs)**, so D should warn in that case.

Sheath face area: `da = J·dx·dz / sqrt(g_22)` at the boundary cell (no y-guards, so the guard metric
is taken equal to the boundary cell, same as Ben). Hermes averages the boundary cell with the guard cell (l.398–401).

---

## 4. Integration decisions

All decided by Sebas on 2026-10-08.

| # | Decision |
|---|---|
| 1 | D is a new approach in `compute_target_power`; it fills the same `q`, `q_norm`, `P_targets`, `fractions` keys, so all existing figures/tables work and can be compared with A and C. Named `Ben_example_fluxes`; `data["approach"] = "D (Ben_example_fluxes)"`. |
| 2 | CLI flag `--ben_example_fluxes` (boolean); error if combined with `--analytic_power`. A stays the default. |
| 3 | γe, γi for D read per case from `BOUT.settings` `[sheath_boundary_simple]` (SF75: 4.5 / 2.5). `--gamma_e/--gamma_i` override them when given explicitly; their CLI defaults become `None` so "not given" can be detected (C keeps falling back to 3.5). |
| 4 | Time: **last time step only** (`t = −1` after the existing clean-step clipping), as in Ben's script. D ignores `--twindow`. ⚠ A and C still average over `--twindow`, so in the comparison figures D is a snapshot and A/C are window means. |
| 5 | Balance table (`print_balance_table`, same layout as `print_power_table`: rows = terms, columns = cases; P_core, x-boundary core/PF/outer, 4 sheaths e/i, Prad, neutrals, Pnet, dE/dt, residual as % of P_core; then the particle balance) is printed whenever D is selected. |
| 6 | Neutral energy terms (§3 item 9) included as their own row. |
| 7 | Radial flow: port Ben's `Div_a_Grad_perp_upwind` as a **full x-loop** returning `F_L, F_R` on every face; the boundaries are sliced afterwards. Keeps the full radial flow field available for later plots. |
| 8 | Disagreements (§3): D follows the Hermes source by default; `--ben_strict` reproduces Ben's formulas exactly (items 1–6 and 8 as Ben wrote them, no neutral row). The kinetic dE/dt crash (item 7) is fixed in both modes. |

Implementation order (one diff at a time, each approved before applying):
1. Helpers: full-x radial flow (Hermes upwind / Ben average under `ben_strict`); sheath per target (power per species + particle flux).
2. `compute_balance(...)` + hook in `load_simulation` (while `ds` is open) → `data["balance"]`; approach D branch in `compute_target_power`.
3. `print_balance_table` + CLI flag + `main` wiring.

---

## 5. Unrelated issues in the existing script (not part of D; noted for later)

- ~~`target_y` used `ny_inner + 1` for SE~~ **fixed 2026-10-13:** `target_y = [0, ny_inner − 1, ny_inner, ny − 1]`
  (l.387; module docstring l.36). `ny_inner` is the first cell of the second leg (the converter removed the
  2 guard cells there), so `efe_tot_ylow[ny_inner]` is the real SE target face; at the last step it matches D's SE electron
  power exactly. Effect on fractions (window mean, NW/NE/SE/SW):
  A SF45 31.6/52.1/6.5/9.8 → 31.6/51.9/6.7/9.7 %, A SF75 30.3/55.1/6.1/8.5 → 30.2/55.0/6.3/8.5 %,
  C SF45 30.8/50.1/8.2/11.0 → 31.0/50.5/7.4/11.1 %, C SF75 29.5/53.4/7.7/9.5 → 29.7/53.8/6.9/9.5 %.
- **Left as is (documented):** approach A at NE/SW reads `efX_tot_ylow` at the upper target cell, i.e. the face one
  cell upstream of the target. Hermes stores the upper sheath power at `yend + 1`, a y-guard that is not in the
  data: `BOUT.squash.nc` has `y = 72` (no y-guards at all), and the script uses `keep_yboundaries=False` (as
  does Ben's, `yboundaries = False`). Guard cells hold only extrapolated values (`limitFree`), so no physics
  is lost. D rebuilds the same quantity with the same formula; it matched Hermes exactly at the lower targets, and
  the upper sheath is the same code mirrored. **Possible cross-check:** compare D against `efe_tot_ylow[yend+1]` from
  per-rank `BOUT.dmp.*.nc` files (which keep the guards) or a squash made with y-guards included.
- Approach C uses CLI γ defaults 3.5/3.5, while the runs use γe = 4.5, γi = 2.5.
- ~~SN: `target_y` read the crown cells `ny_inner ± 1` as NE/SE~~ **fixed 2026-10-13** (§6.1).
- ~~Figure filenames don't depend on the approach, so D overwrites the A/C figures~~ **fixed
  2026-10-13:** `approach_tag` in `main` (l.1087–1088) ∈ {`A`, `C`, `D`, `D_strict`} is appended to
  `SF_heat_flux_targets_*`, `SF_heat_flux_targets_normalised_*`, `SF_power_fractions_*` (.pdf/.png).
  `SF_grids` and `SF_flux_surface_profiles` are approach-independent and keep their names. Old
  unsuffixed figures in the folder are stale copies from before the change.

Comparing D with A and C should make these visible directly in the per-target fractions.

---

## 6. Progress and validation notes

### 6.1 Implementation status
- **Chunk 1 applied (2026-10-08):** `ME` constant (l.69); helpers `div_a_grad_perp_flows`,
  `_limit_free`, `sheath_target_flows`, `sheath_targets`, `core_y_mask` (l.502–616).
  `div_a_grad_perp_flows` covers every x face but is vectorised (numpy slicing) rather than Ben's
  Python `for x` loop; the numbers are the same. `ben_strict` uses our SI constants, not Ben's rounded ones (~0.1%).
- **Chunk 2 applied (2026-10-09):** `load_simulation(..., ben_example, ben_strict, gamma_e, gamma_i)`
  (l.257) calls `compute_balance` at the last clean step → `data["balance"]` (l.376–383);
  `compute_target_power(..., ben=False)` (l.437) has the D branch (l.479–482) using
  `slot_power`/`slot_area`; `_opt` (l.643) and `compute_balance` (l.650–797). SN is detected by
  `jyseps2_1 == jyseps1_2` and fills the NW/SW slots only. With `ben=True` the γ arguments of
  `compute_target_power` are ignored; D's γ goes through `load_simulation`.
- **Chunk 3 applied (2026-10-09):** `print_balance_table` (l.528–585); CLI `--ben_example_fluxes`,
  `--ben_strict`, `--gamma_e/--gamma_i` default `None` (l.1045–1057); flag checks (l.1071–1074);
  `main` passes D arguments to `load_simulation`/`compute_target_power` and prints the balance table
  when D is selected (l.1089–1114). C still falls back to γ = 3.5/3.5 when no `--gamma_*` is given.
- **SN target fix applied (2026-10-13), affects A, C and the sanity table (not D):** in a single null
  `ny_inner = jyseps2_1 = jyseps1_2` is the top of the core, so `target_y` read the crown cells
  `ny_inner ± 1` as "NE"/"SE". Now `target_y = [0, None, None, ny − 1]` when
  `jyseps2_1 == jyseps1_2` (l.387–389), `col` returns zeros for `None` (l.392), and approach A
  divides with `where=area > 0` (l.496–497). SF results are bit-identical before/after.

**Usage:**

    python SF_heat_transport_pp_xhermes_withfluxes.py <sims_dir> <grid_dir> --ben_example_fluxes [--ben_strict] [--gamma_e G --gamma_i G] [--only SF45 SF75 SN]

### 6.1a Full-CLI check, `--only SF45 SF75 SN --ben_example_fluxes` (No_Drifts runs)

| | SF45 | SF75 | SN |
|---|---|---|---|
| P_core [kW] | 314.3 | 325.4 | 325.3 |
| Sheath total [kW] | 15.7 | 16.0 | 10.6 |
| dE/dt [kW] | 293.3 | 303.3 | 309.9 |
| Residual [% P_core] | −0.067 | −0.081 | +0.013 |
| Recycling (direct) | 0.99 | 0.99 | 0.99 |
| D fractions NW/NE/SE/SW [%] | 31.3/53.3/5.9/9.5 | 30.1/56.3/5.5/8.1 | 33.4/0/0/66.6 |

SN before/after the target fix (approaches A and C):

| SN | Before | After |
|---|---|---|
| Sanity Pe / Te / area | 134 Pa / 1670 eV / 121 cm² (crown cells) | 0.69 Pa / 17.8 eV / 41.3 cm² |
| A fractions NW/NE/SE | 26.3 / 10.0 / 12.1 % | 33.8 / 0 / 0 % |
| C fractions NW/NE/SE | 0.1 / 47.4 / 52.4 % | 33.8 / 0 / 0 % |

**Earlier SN figures/fractions made with A or C are contaminated by the crown cells.**

### 6.2 `BOUT.settings` "unused" flags are meaningless in these runs
In `Modified_SF_Runs/No_Drifts/SF75/BOUT.settings` every option is flagged
`# unused value (NOT marked conditionally used)`, including `[solver] type = cvode`, which the run
clearly used. The file was written before the components read their options. Consequences for D:
- Read the values regardless of the flag (e.g. `gamma_e = 4.5`, `gamma_i = 2.5` are correct).
- Options the components would have added with defaults (`sheath_ion_polytropic`,
  `density_boundary_mode`, `temperature_boundary_mode`, …) are **absent**, so D falls back to the
  Hermes defaults (1.0, 1, 1).

### 6.3 Chunk 1 check on SF75 (last clean step, scratch script, nothing written)
γe = 4.5, γi = 2.5, power summed over interior x.

| Target | D Hermes Pe / Pi [W] | Dumped `efe_tot_ylow` / `efd+_tot_ylow` at that row [W] | D `ben_strict` Pe [W] |
|---|---|---|---|
| NW, y=0, lower | 2639 / 2183 | 2639 / 2163 | 1173 |
| NE, y=55, upper | 5051 / 3986 | 5141 / 3708 (one cell upstream) | 5050 |
| SE, y=56, lower | 300.2 / 581.8 | 300.2 / 582.5 | 133.4 |
| SW, y=71, upper | 523 / 774 | 529 / 702 (one cell upstream) | 523 |

- Lower targets: the D electron power matches Hermes's own sheath diagnostic **exactly**, and the ion power is within 1%
  (probably cell vs. face `Vi` in `max(Vi, Cs)`).
- Ben's `γe − 2.5` at lower targets only (§3 item 3) cuts the lower-target electron power by ~55%.
- `core_y_mask("sf_plus_low_field_side")` matches exactly where `Pe_src[xstart] ≠ 0` (y = 8…39).

### 6.4 Open physics question: SF75 has zero anomalous transport
`anomalous_D_e`, `anomalous_D_d+`, `anomalous_Chi_e` and `anomalous_Chi_d+` are 0 everywhere; only `anomalous_nu_d+ = 2`.
Neither `BOUT.inp` nor the grid (`mastu_sf75p_bout.grd.nc` has no `D_*`/`chi_*` variables)
provides them, so Hermes leaves the D/χ terms off (`anomalous_diffusion.cxx:26–40`). With no drifts,
plasma energy can only leave the core via neutrals: ~1 MW injected (`power_in = 0.5` MW per
species) vs ~16 kW at the four targets at the last step. The run is known not to be converged.
**Sebas is asking Ben whether χ = D = 0 is intended** (Ben asked to see the input file). The radial terms of the D balance will be
exactly 0 for this run until that's resolved.

### 6.5 Chunk 2 balance check on SF75 (last clean step, t index 100)

| Term [W] | D (Hermes) | D (`ben_strict`) |
|---|---|---|
| P_core = 3/2 ∫(Pe_src + Pd+_src) dV | 325 400 | 325 400 |
| Radial in (core / PF) / out | 0 / 0 / 0 | 0 / 0 / 0 |
| Sheath e (sum of 4 targets) | 8 513 | 6 880 |
| Sheath i (sum of 4 targets) | 7 525 | 7 525 |
| P_rad | 2 506 | 3 760 |
| Neutral (target_recycle +114, wall_refl −3 867, target_refl −100) | −3 852 | — |
| dE/dt internal + kinetic | 303 100 + 175 | 303 100 + 175 |
| **Residual** | **−264 (0.08 % of P_core)** | **+3 969 (1.2 %)** |
| Recycling fraction: Ben's formula / direct | 1.00 / 0.99 | 1.00 / 0.99 |

- The Hermes-mode balance closes to ~1 % of the total losses; the direct recycling fraction matches
  `target_recycle_multiplier = 0.99`.
- D vs A target fractions (NW/NE/SE/SW): D 30.1 / 56.3 / 5.5 / 8.1 %, A (window mean) 30.3 / 55.1 / 6.1 / 8.5 %.
- **Only 325 kW is injected, not the nominal 1 MW** (`power_in = 0.5` MW per species): the source is
  `H(x)·H(0.035−x)·pressure_source` with `core_volume = 0.29689`, so either the mask covers only part
  of the core or `core_volume` doesn't match the grid.
- dE/dt ≈ 303 kW (93 % of the input): the core is heating up, consistent with χ = D = 0. To raise with Ben.
