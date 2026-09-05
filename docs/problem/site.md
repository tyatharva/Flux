# The site

The UW-Madison Kegonsa Solar Array, southern Wisconsin, with one eddy-covariance tower
inside the array.

## Tower and array

| | |
|---|---|
| tower, surveyed | `42.957160, −89.292362` (EPSG:3071 `577719.1, 276299.5`). The single source of truth is `TOWER_LON`, `TOWER_LAT` in `bin/prep_stage6.py`. |
| instrument height | about 10 m above ground. The model receptor is 30 m. See [motivation](motivation.md). |
| solar array | the tower is inside it: 60 m east and west, 250 m north, 100 m south. A 120 × 350 m rectangle, 4.20 ha, defined in EPSG:3071. Nothing about it depends on the wind. |
| aerodynamic receptor height | the array surface is raised 1.5 m (`--raise-topo`), so the 30 m receptor is 28.5 m above the aerodynamic surface. That is the value every record stores. |

The array is in the footprint almost only for northerly flow. The measured N-wind array
share is 30.28%, against the 30.7% Kljun predicts for north at `z_m = 30 m`. The two agree
independently to 1.4%.

## Land cover and terrain

- **Land cover**: ESA WorldCover v200 (2021), 10 m. **Terrain**: USGS 3DEP 1/3 arc-second.
  Both raw tiles are in `data/raw/` (not in git, see [data](../getting-started/data.md)).
- `bin/prep_surface.py` warps both onto the 122 × 122 grid at 30 m centred on the tower. It
  mode-resamples land cover and averages elevation, and writes `data/grid30_raised/`
  (tracked, 184 kB): `topo.npy`, `z0m.npy`, `htFlux.npy`, `lcclass.npy`, `array.npy`,
  `water.npy`, `dmap.npy`, `meta.npy` and the two clipped rasters `dem24.tif` and `lc24.tif`.
- **Roughness per class**: water 1e-4 m, grass 0.03, cropland 0.10, built 0.5, tree 1.0.
  The array rectangle then overrides it. WorldCover labels the array as cropland because it
  does not see photovoltaics.
- The geometric-mean `z0` over the domain is 0.0615 m. Water covers 13.61% of it (Lake Kegonsa
  is to the east and is inside the footprint for easterlies). The array is 0.30% (44 cells).
- **Terrain is tapered at the periodic seams. Land cover is not.** Terrain height enters the
  coordinate transform and its metric tensor, so a step at the seam is a numerical
  discontinuity. The taper knee is at pad 12, so real geography extends to 1470 m from the
  tower. Roughness and heat flux are local boundary conditions, and a seam in them is no
  different from a coastline.

## Panels as a bulk patch

Panels are never explicit geometry. Row spacing is 5–7 m, well under one 30 m cell. They are
a surface patch with elevated `z0`, displacement height `d ≈ 1.5 m` and a raised heat flux.

- Production uses `--raise-topo`: `z0_array = 0.25 m` against cropland's 0.10 (2.50×). At
  `z0_array = 0.10` the array is aerodynamically identical to the cropland it replaced and its
  entire neutral signal is zero. `prep_surface.py` warns when the two coincide.
- **A stable case is a roughness-only array case.** The per-class flux table is a daytime
  enhancement table with no nocturnal equivalent. At night the physics inverts, so a stable
  case gets a uniform negative `htFlux`. Never claim a thermal array response at night.
  (There are no stable cases in the corpus. See
  [limitations](../limitations-and-future-work.md).)

## The surface heat flux is per cell, and virtual

The run is dry, and the heat flux exists to drive buoyancy, so `htFlux` is the *virtual*
kinematic flux. Literature ratios are sensible-flux ratios. The conversion depends on the
Bowen ratio and therefore on the class:

$$
\overline{w'\theta_v'} = \overline{w'\theta'}\,\left(1 + \frac{0.0735}{B}\right)
$$

| class | Bowen ratio B | sensible ratio | virtual factor (cropland = 1) |
|---|---|---|---|
| cropland (reference) | 0.4 | 1.00 | 1.000 |
| array | 4 | 1.60 | **1.376** |
| built | | | 1.314 |
| tree | | | 1.100 |
| grassland | | | 1.066 |
| water | 0.15 | 0.12 | **0.151** |

Working in virtual flux compresses the wet–dry contrast (array-to-water falls about 32%)
because the wetter surface's latent flux adds buoyancy. That is correct, and it is the cost
of running dry.

The `.in` scalar `surflayer_wth` must be the *domain mean* of that map, not the cropland
reference. A flat spin-up has no restart injection, so the scalar is its flux.

**Albedo has no pathway, and that is not an omission.** There is no radiation scheme.
`surflayerSelector = 1` prescribes the kinematic heat flux directly, so everything albedo would
control is already inside `htFlux`.

## Wind direction is set by rotating the forcing, not the map

Direction is set by rotating the geostrophic vector. The surface is bit-identical for every
direction, so any directional difference is flow rather than a resampling artifact. A square
periodic domain with `dx = dy` over a flat uniform surface is exactly equivariant under 90°
rotation. That is what lets one seed serve four headings.

**Achieved direction is not forcing direction.** Ekman turning measured from the seed library
is 5.2° convective (n = 18) and 16.9° neutral (n = 12). Cases are labelled by achieved
direction.

## Climatology

CONUS404 (45 years) characterises the site and sets the sweep ranges: the wind rose, the stable
fraction (about 44% of QC'd hours) and the boundary-layer depth distribution. It never forces a
run. HRRR analyses do (see [case generation](../les/case-generation.md)). The scored artifacts
are `results/conus404_site.{txt,npz}` and `results/stable_fraction.txt`.

| direction | corpus share | site rose | mean array share in the footprint |
|---|---|---|---|
| N | 6.9% | 10.6% | **30.3%** |
| NW | 14.6% | 14.5% | 6.9% |
| NE | 6.4% | 10.2% | 2.7% |
| W | 21.4% | 14.4% | 0.3% |
| SW | 19.4% | 14.3% | 0.9% |
| E | 4.5% | 10.4% | 0.2% |

Convective afternoons here favour SW and W, so the corpus rose is skewed away from the one
direction that holds the site-specific signal.
