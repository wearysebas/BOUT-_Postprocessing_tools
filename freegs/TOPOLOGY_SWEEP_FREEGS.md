# TOPOLOGY_SWEEP_FREEGS — MAST-U Snowflake G-EQDSK Factory

`freegs_sf_creator.py` generates **INGRID-ready G-EQDSK files** for MAST-U
lower-divertor snowflake configurations, with the two lower X-points placed
at user-requested positions. Its purpose is to produce *families* of
smoothly-varying equilibria that transition between snowflake configurations
(SF+, SF−, exact), so each member can be gridded by INGRID and used to study
how edge/divertor transport changes across the topology transition.

Everything below documents not just *what* the script does but *why* — most
of the design decisions exist because the obvious alternative demonstrably
breaks either the Grad-Shafranov solve or INGRID's grid tracing.

---

## 1. Quick start

```bash
python3 freegs_sf_creator.py RXPT1 RXPT2 ZXPT1 ZXPT2 [options]
```

⚠️ **Argument order is `R1 R2 Z1 Z2`, not pairwise `(R1 Z1) (R2 Z2)`.**

| Argument | Meaning |
|---|---|
| `RXPT1 ZXPT1` | primary (active) X-point [m] |
| `RXPT2 ZXPT2` | secondary X-point [m] |

Key options:

| Option | Default | Purpose |
|---|---|---|
| `--ref FILE` | `freegs/g049467.070005_modified` | reference eqdsk supplying profiles, core shape, upper-half psi |
| `--wall FILE` | `freegs/target_plates/simplified_limiter_mastu.txt` | limiter (R,Z comma-separated) |
| `--steps N` | 1 | continuation steps from the reference's natural nulls to the targets; raise for targets > ~2 cm away |
| `--psi-sign S` | −1.0 | sign matching file p'/ff' to FreeGS' internal psi orientation |
| `--gamma G` | 1e-8 | Tikhonov regularisation on coil currents |
| `--px A` | 1000.0 | frozen Px circuit current (see §4) |
| `--no-leg-anchors` | off | disable reference-derived divertor leg isoflux anchors |
| `--no-graft` | off | skip the upper-half graft (leaves a competing upper X-point — see §6) |
| `--graft-band Z0 Z1` | 0.1 0.6 | Z blend window of the graft |
| `--out FILE` | auto from X-points | output eqdsk name (a matching `.png` is always written) |

Validated invocations:

```bash
# Fidelity test vs the original (hand-doctored) reference:
python3 freegs_sf_creator.py 0.6733 0.6812 -1.2624 -1.3219

# SF75 case from Sebastian's grids (near-double-null reference):
python3 freegs_sf_creator.py 0.65 0.69 -1.22 -1.34 \
    --ref ~/Documents/files_for_sebastian/grids/SF75/g049465.084044

# SF15 / SF-minus (g049468 ideal SF-): horizontal null pair, anchors OFF,
# Px halved (1000 A spawns an O-point at (0.31,-1.25) psiN 1.021 here):
python3 freegs_sf_creator.py 0.5859 0.7500 -1.3152 -1.3161 \
    --ref '.../MAST-U/SF- ideal/g049468.068024' --no-leg-anchors --px 500
```

---

## 2. What the reference file supplies

The script is **not** a from-scratch equilibrium designer. It re-solves a
free-boundary equilibrium that keeps the *physics* of an experimental
reference shot while moving the divertor nulls:

| Taken from the reference | Used for |
|---|---|
| `pprime(psiN)`, `ffprim(psiN)` tables | plasma current profile (`ProfilesPprimeFfprime`; profiles fixed, Ip emerges ≈ reference Ip) |
| `fpol` edge value | vacuum `R·Bt` |
| boundary (`bdry`) array | core LCFS shape constraints (§5) |
| separatrix leg geometry | divertor leg / strike-point anchors (§5) |
| upper-half psi map | grafted onto every output (§6) |
| grid/domain (65×65, R∈[0.06,2.0], Z∈[−2.2,2.2]) | output header identical to the reference |
| sign conventions | output COCOS (§6) |

### Known references

* **`freegs/g049467.070005_modified`** (default). From the paper this work
  builds on. **Hand-doctored**: its author edited the upper-half psi to
  delete the upper X-point (the psi-map core tops at Z≈0.5 while its bdry
  array claims Z≈1.118 — not self-consistent, but INGRID accepts it).
  Natural lower nulls: (0.6733, −1.2624), (0.6812, −1.3219).
* **`SF75/g049465.084044`** (Sebastian's grids). **Not doctored** —
  near-double-null, upper X-point at psiN ≈ 1.007. Natural lower nulls:
  (0.6478, −1.2267), (0.6895, −1.3419). Same domain, signs and Ip
  (≈0.594 MA) as the default reference, so `--psi-sign -1` is unchanged.
* **`INGRID_Final/.../MAST-U/SF- ideal/g049468.068024`** (SF−/SF15).
  Near-exact snowflake-minus: natural lower nulls (0.5859, −1.3152) and
  (0.7500, −1.3161) — a near-**horizontal** pair, secondary 16 cm
  *outboard*, both at psiN ≈ 1.000. Upper X-points at psiN 1.0023/1.0064
  (near-DN class, needs `remove_upper_divertor`). Same domain/signs.
  Validated recipe: `--no-leg-anchors --px 500` (see §9).

Both references run through the full chain (solve → eqdsk → INGRID grid).

---

## 3. Pipeline overview

```
parse reference eqdsk  (raw fortran-format parser, full profile control)
        │
build MAST-U machine   (coil set with calibrated freezes — §4)
        │
[stage A]  LSN solve at the reference's NATURAL primary null   (§5)
        │
[stage B]  snowflake solve at the requested X-points
           (optionally walked in --steps continuation increments)
        │
write FreeGS eqdsk → re-parse → flip to reference sign conventions (§6)
        │
graft reference upper-half psi onto the solution (§6)
        │
final-file audit: X-point positions, null census, INGRID yaml block (§7)
        │
psiN contour plot (.png)
```

---

## 4. The machine model (and why each coil is frozen)

`build_machine()` builds MAST-U as: solenoid + Pc, divertor coils
D1–D7 + Dp (upper & lower, 23–35 turns), P4/P5 up-down-symmetric circuits,
Px circuit, P6 vertical-control circuits. The configuration was calibrated
against the default reference (2026-06-10); every freeze fixes a concrete
failure:

| Coil(s) | State | Reason |
|---|---|---|
| D1L…D7L, DpL | **controlled** | these sculpt the snowflake |
| D1U…D7U, DpU | frozen 0 A | if controllable, the regularised solution drifts to a connected double-null (upper X at psiN = 1.000 steals the separatrix) |
| P4, P5 | controlled, symmetric circuits | real machine wiring; gives vertical field without sculpting an upper null |
| P61, P62 | frozen 0 A | when controllable, least-squares drives them to ±50–75 kA (2-turn coils!) fighting each other, spawning nulls near (1.2, −1.1) |
| Px | frozen **+1000 A** | +2000 A → spurious O/X pair at (0.30, −1.24), psiN ≈ 1.01, in the inner-leg path (breaks INGRID leg tracing); 0 A hands the boundary to the upper region. +1000 A → clean null census, physical few-kA currents |

---

## 5. The two-stage solve

### Stage A — LSN at the reference's *natural* null

Stage A solves a clean lower-single-null **at the reference file's own
detected primary X-point**, not the requested one. This matters:

> **Near-double-null limit cycle (found 2026-06-11, SF75 reference).**
> With a reference whose upper X-point is close to the separatrix
> (psiN ≈ 1.007), asking stage A for an X-point even a few mm off the
> file's natural null throws Picard into a **period-5 limit cycle**: the
> boundary-defining null flips between the lower null and the upper
> X-point, Ip slams between 0.6 and 1.0 MA, and the iteration never
> converges (`RuntimeError: Picard iteration failed to converge`).
> Solving at the natural null always converges, because that *is* the
> file's own equilibrium.

### Stage B — snowflake at the requested targets

Adds the secondary X-point and moves both to the requested positions,
starting from the converged stage-A state. With `--steps N > 1` the targets
are approached in N equal increments from the natural nulls (continuation),
re-solving at each — use this for sweep members far (> ~2 cm) from the
reference nulls.

**No null–null flux tie**: constraining the two X-points to identical flux
drives a degenerate closed pocket (spurious divertor O-point). The flux
split between the nulls is whatever the requested geometry implies.

### Constraint set (both stages)

1. **X-point constraints** at the target null position(s).
2. **Core LCFS shape isoflux**: the reference boundary is sampled **evenly
   in poloidal angle** (≈10 points; even sampling stops clustering near the
   X-point) and each point is tied to the primary null's flux. Covering
   the upper half locks the core vertically and stops upward drift.
3. **Divertor leg anchors** (`REF_ANCHORS_X1/X2`): psiN = 1.0 crossings
   extracted from the default reference — W1/E1 strike points and leg
   midpoints for X1, W2/E2 for X2, plus the outboard lobe extremity. Each
   is tied to the primary X-point flux and **rigidly translated with its
   parent X-point** when the targets move. Without them, the unconstrained
   divertor field grows a psiN ≈ 1 lobe out to R ≈ 1.35 that deflects the
   X2-SE leg off the E2 plate (INGRID: *"target does not intersect field
   line"*).
4. **Private-flux depth anchors** (`REF_PF_W1/W2` ↔ `REF_PF_CORE`):
   psiN = 0.963 (INGRID's `psi_pf_1`) crossings on each W plate tied to the
   outboard-midplane point of the same surface. Without them the flux span
   along W1 falls 4×10⁻⁴ short of `psi_pf_1` and INGRID's A1_S trace
   misses the plate. **INGRID's tolerances are ~1e-3 in psiN** — this is
   the precision class the constraints must hit.

Legacy warning: do **not** anchor an outer leg at (1.20, −1.55) the way the
old `sf_switch.py` did — it forces a large spurious separatrix lobe through
the divertor chamber. `sf_switch.py` (repo root) is unvalidated prior art.

---

## 6. Post-processing

### Sign conventions (COCOS match)

FreeGS writes psi *decreasing* outward with positive `fpol`; the references
have psi *increasing* outward with `fpol < 0` (Bt along −φ). INGRID's
gridue export checks Bp orientation against the grid and **rejects the
FreeGS convention** (`Bp_dot_grady and Bpxy have opposite signs`). The
output is therefore flipped (psi, simag, sibry, pprime, ffprim, fpol,
bcentr) to the reference's conventions; q is invariant under the combined
flip.

### Upper-half graft

FreeGS solutions of this plasma *always* grow an upper X-point at
psiN ≈ 1.002 — inside INGRID's SOL band (`psi_1` = 1.06) — and no physical
coil trick pushes it further out. The fix: **graft the reference's
upper-half psi onto the solution**. The reference psi is first mapped to
the solve's gauge with the affine transform matching axis and boundary
flux (exact, since both maps share the axis position and the
X-point-defined separatrix), then blended with a cosine ramp in Z over
`--graft-band` [0.1, 0.6]: pure solve below, pure reference above.

Two consequences worth remembering:

* The grafted upper half is **identical across all sweep members** — only
  the divertor varies between grids, which is exactly what a clean
  transition study wants.
* The output inherits the reference's upper-half nulls. With the doctored
  default reference there are none; with SF75 (`g049465.084044`) the upper
  X-points sit at psiN 1.007/1.0115 — *inside* the psi_1 = 1.06 band, but
  INGRID accepts this with `remove_upper_divertor: true` (verified: both
  the original file and our output gridded fine).

---

## 7. Reading the output

For each run you get the solved-equilibrium report, then the **final-file
audit** (re-parsed from the written file, i.e. exactly what INGRID sees):

* **Lower X-points (file vs target)** with the miss in mm. Expect < 1 mm
  on the default reference, ~5 mm on SF75 (see §9).
* **INGRID yaml (grid_settings)** — paste-ready block:

  ```yaml
  rmagx: 0.9472
  zmagx: -0.0177
  rxpt: 0.6540      # primary  (matched to your requested primary)
  zxpt: -1.2164
  rxpt2: 0.6910     # secondary
  zxpt2: -1.3416
  ```

* **Null census** (divertor window, then upper half): every X- and O-point
  near the separatrix with its psiN. A griddable snowflake shows **only
  the two intended X-points** within psiN ≈ [0.99, 1.05]; anything else in
  that band (spurious O-points, extra X-points) is what breaks INGRID's
  leg tracing. Nulls at psiN ≲ 0.76 or ≳ 1.15 are harmless.
* **`<out>.png`** — psiN contour map with separatrix, SOL levels
  (1.02/1.04/1.06), wall and target X-points.

---

## 8. INGRID handoff

1. Point the yml's `eqdsk:` at the new file; paste the audited
   `rmagx/zmagx/rxpt/zxpt/rxpt2/zxpt2` into `grid_settings` (they are
   refinement guesses, mm accuracy not required).
2. Settings that work: `num_xpt: 2`, `remove_upper_divertor: true`,
   psi levels in the class of `psi_1: 1.06`, `psi_2: 0.97`,
   `psi_core: 0.85`, `psi_pf_1: 0.97`, `psi_pf_2: 1.06`
   (SF75 yml: `mastu_sfp.yml`; original validation: `mastu_sfexact.yml`
   with `psi_pf_1: 0.963`).
3. Headless driver lives in `MAST-U_Topology_Sweep/ingrid_test/`
   (`test_sf.yml` = the validated yml repointed).
4. **INGRID_Final bug**: `ExportGridue` is missing the CDN import — shim
   with `INGRID.ingrid.CDN = INGRID.ingrid.UDN` before calling.

Acceptance protocol for any new sweep member: built-in audit clean (null
census, §7) **and** a successful headless INGRID run.

---

## 9. Known sharp edges

* **Argument order** is `R1 R2 Z1 Z2`. The pairwise misread fails loudly
  (Picard error), not silently — but you lose a solve to find out.
* **~5 mm primary X-point miss on SF75-class references** (vs < 1 mm on
  the default). Suspected cause: the leg anchors (`REF_ANCHORS_*`) are
  extracted from the *default* reference's legs and only rigidly
  translated. If mm fidelity ever matters, re-extract the anchors from the
  new reference's psiN = 1.0 crossings. INGRID does not care.
* **Quartic-spline saddle finder** (`lower_xpoints`): the grid-based
  finder misses shallow snowflake saddles, hence the high-resolution
  spline + Newton refinement. The same machinery powers the null census.
* **Picard convergence is fragile near double-null** — see §5. If a run
  fails with `Picard iteration failed to converge`, first suspects in
  order: targets far from the natural nulls (raise `--steps`), a
  near-double-null reference, wrong argument order.
* **Profiles are not Ip-constrained**: `ProfilesPprimeFfprime` integrates
  whatever p'/ff' imply; Ip lands at the reference value because the
  profiles and shape do. Harmless `IntegrationWarning`s from quad are
  expected.

## 10. Validation history

| Date | Case | Result |
|---|---|---|
| 2026-06-10 | machine/coil calibration vs `g049467.070005_modified` | frozen-coil configuration of §4 |
| 2026-06-11 | fidelity test: reproduce default reference from its own X-points | < 1 mm misses; full INGRID chain: SF75 topology, 27 patches, 68×30 gridue, dims identical to reference control |
| 2026-06-11 | SF75 `g049465.084044`, yml targets (0.65,−1.22)/(0.69,−1.34) | limit cycle found & fixed (stage A at natural null, `--steps`); misses 5.4/1.9 mm; INGRID grid created successfully |

Detailed session memory: `/Users/sruiz/Dev/Claude_memory_brain/freegs_SF_sweep/memory.json`
(MCP memory server; search with **single-word** queries).
