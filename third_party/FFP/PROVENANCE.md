# Kljun et al. (2015) FFP — the OFFICIAL implementation, vendored

This directory holds the reference implementation of the Flux Footprint Prediction
parameterisation, unmodified. **Nothing in it is edited.** The project's adapter is
`lpdm/kljun_ffp.py`, which calls into this code and never reimplements a formula.

## Provenance

| | |
|---|---|
| source | <https://footprint.kljun.net/downloads/v1.42/FFP_Python.zip> |
| reached from | <https://footprint.kljun.net/> → Downloads → skip registration (`download_2.php`) |
| version | **1.42** (`calc_footprint_FFP.py` docstring: "version: 1.42", last change 11/12/2019, ported to Python 3 by Gerardo Fratini) |
| retrieved | 2026-08-30 |
| zip sha256 | `50e188c51eb84e18896cf0cac39fcd504ddc32d91afca4582352dd1c299864ba` |
| mtimes in zip | 2025-08-07 |

sha256 of the extracted files:

```
3bbf5d2f95bf388df037234e7b373468b30129a274113e694ea79d6b8615af3c  calc_footprint_FFP.py
6bb896aace825aeff576ab2fc5bb4da07f2d21b159d7ba735d96124197856167  calc_footprint_FFP_climatology.py
d1c9d82cb7e0c674542f760a6cf761a512513384dcf6c2ddb2f46c30d6f5220e  license.txt
```

`FFP_readme_Python.pdf` is in the zip and is deliberately NOT vendored — it is 184 kB of
binary that git would carry forever, and the code is the artifact.

Cite: N. Kljun, P. Calanca, M. W. Rotach, H. P. Schmid, "A simple two-dimensional
parameterisation for Flux Footprint Prediction (FFP)", Geosci. Model Dev. 8, 3695-3713,
2015, doi:10.5194/gmd-8-3695-2015.

## Licence

ISC-style; see `license.txt`. Redistribution is permitted provided the copyright notice
and permission notice travel with the copies, which is why `license.txt` is vendored
beside the code rather than summarised here.

## Why this is here at all, and what it replaced

`lpdm/kljun.py` is a **reimplementation from the paper's equations**, written before this
directory existed. It is not wrong in the regimes the project has used it in, but it is a
second implementation of a published model, and every training pair carries a Kljun raster
as an input channel — so a divergence would corrupt the corpus silently rather than
loudly. The reference implementation removes the question instead of answering it.

**Two divergences were found on sight while writing the adapter, and both are recorded
because one of them matters:**

1. **The near-neutral crosswind width was 25% too wide.** FFP's `sigma_y` carries a
   `scale_const` (paper Eq. 13, `p_s1`). The official code resets `ol = -1e6` whenever
   `|ol| > oln = 5000` and then evaluates
   `scale_const = 1e-5 |ol/z_m| + 0.80`, which at `z_m = 30 m` gives **1.133, clipped to
   1.0**. `lpdm/kljun.py` short-circuits `|L| > 1e5` to `ps1 = 0.8` *without* the clip, so
   it divides by 0.8 and returns a `sigma_y` **1.25x** the official one. It bites exactly
   where the project's standing regression lives — the **flat/neutral control**, whose `L`
   is effectively infinite — and nowhere else: at `|L| < 5000` the two agree to roundoff
   (`case_2023121921`, `L = -738`: both give 0.80025).
2. **The neutral cutoff is `oln = 5000`, not `1e5`.** For `5000 < |L| < 1e5`
   `lpdm/kljun.py` takes its own neutral branch (`psi_m = 0` or the stable form) where the
   official takes the *convective* form. The resulting `Psi_M` difference is ~3e-4 at
   `L = +50000`, i.e. negligible next to (1) — recorded for completeness, not as a
   concern.

`lpdm/kljun.py` stays in the tree: `bin/corpus_monitor.py` and the seed-stationarity gates
were validated against it and re-pointing them is a separate change with its own
regression cost. **What must not happen is the two drifting apart silently**, so
`bin/test_kljun_adapter.py` scores them against each other on corpus-representative inputs
and prints the difference rather than asserting it away.
