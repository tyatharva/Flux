# Data

Binary inputs are never committed except the production surface. Provenance, sizes and
regeneration: `docs/getting-started/data.md`; how the surface is built: `docs/problem/site.md`.

| path | in git | what |
|---|---|---|
| `grid30_raised/` | yes (184 kB) | the production surface for the 122² @ 30 m grid: `topo`, `z0m`, `htFlux`, `lcclass`, `array`, `water`, `dmap`, `meta` (`.npy`), `topo.bin`, `dem24.tif`, `lc24.tif`; built by `bin/prep_surface.py --raise-topo` |
| `raw/` | no | USGS 3DEP 1/3 arc-second elevation and ESA WorldCover v200 (2021) tiles, the CONUS404 site extract, the tower's `H_and_sigma_w.csv` |
| `hrrr/` | no | the HRRR sounding cache; every case re-fetches from the public archive |
| `grid16*`, `grid24*`, `grid_cbl`, `grid`, `case_grids/`, `smokelib/` | no | retired grids and the stubbed-LES smoke library, regenerable |

Tower: `42.957160, −89.292362` (EPSG:3071 577719.1, 276299.5), surveyed; the single source of
truth is `TOWER_LON/TOWER_LAT` in `bin/prep_stage6.py`. The array rectangle is 60 m east and
west, 250 m north, 100 m south of it.
