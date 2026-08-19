# Data provenance

Binary data files are **never committed** to this repository. This file records
enough provenance that the pipeline is reproducible without them.

---

## Dane County, WI LiDAR bare-earth DEM

| | |
|---|---|
| Archive | `Dane2024_beDEM45cm_ELm_WTM.zip` (5,634,734,305 B uncompressed, 5 files) |
| Raster | `Dane2024_beDEM45cm_ELm_WTM.tif` (4,653,810,191 B) |
| Product | Bare-earth DEM derived from the 2024 Dane County LiDAR collection |
| Vintage | 2024 LiDAR; mosaic derived 2025-10-15, raster written 2025-10-24, metadata 2025-11-07 |
| **CRS** | **EPSG:3071** — NAD83(HARN) / Wisconsin Transverse Mercator, linear unit metre |
| **Cell size** | **0.4572 m** exactly ( = 1.5 US survey feet) — note the filename says "45cm"; the true value is 45.72 cm |
| Vertical unit | metres ("ELm"), converted from US survey feet by multiplication by 0.3048 |
| Pixel type | 32-bit floating point, LZW-compressed GeoTIFF |
| Producer | Dane County, via Esri ArcGIS 13.3.2.52636 |
| **Source URL** | **TODO — fill in** |
| **Download date** | **TODO — fill in** |

### Derived product

`data/dem/kegonsa_30m_wtm.tif` — 267 x 267 cells at 30 m, EPSG:3071, produced by
`docker/prep_dem30.sh`. Resampled with `gdalwarp -r average`, not nearest: this is a 65.6x
linear reduction (0.4572 -> 30 m), and nearest neighbour would alias individual LiDAR posts
into LES surface elevations. The tile is +/-4 km about the tower, which covers the
4380 x 1500 m LES domain at **any** rotation about it (farthest corner 3.37 km).
Both this file and the extracted full-resolution `.tif` are gitignored.

### Extent

WTM (EPSG:3071) bounding box, from the raster metadata:

```
X:  532285.419389  ->  602560.716989   (70275.3 m)
Y:  262579.305851  ->  314474.249051   (51894.9 m)
```

Geographic bounding box:

```
lon:  -89.849674  ->  -88.982150
lat:   42.831329  ->   43.302998
```

World file (`.tfw`) upper-left pixel centre: `X = 532285.6479894917`,
`Y = 314474.0204508007`; pixel size `0.4572`, `-0.4572`.

### Elevation statistics (band 1, from `.tif.aux.xml`)

```
count   15,522,191,071 cells
min     219.45600891113 m
max     456.59039306641 m
mean    289.41810711886 m
stddev   28.638435244641 m
```

### Processing lineage (from the embedded ArcGIS metadata)

The county's native product is in **NAD83(2011) / WISCRS Dane County, US survey feet**
(Lambert Conformal Conic). It was transformed to the delivered raster by:

1. `RasterCalculator`: `"Dane_ccs.tif" * .3048` — converts **elevation** from feet to metres.
2. `ProjectRaster` to `NAD_1983_HARN_Wisconsin_TM` (EPSG:3071), resampling
   **nearest neighbour**, output cell size `0.4572 0.4572`, geographic transformation
   `NAD_1983_HARN_To_NAD_1983_2011`, `NO_VERTICAL`.
3. Renamed `...beDEM60cm...` -> `...beDEM45cm...` (the "60cm" label was superseded;
   the actual cell size throughout is 0.4572 m).

Nearest-neighbour resampling means the delivered raster is **not** independently
smoothed — relevant when it is downsampled to the 10 m LES grid in Stage 6.

### Notes for use

- **Tower coordinate — SURVEYED.** `42.957160, -89.292362`
  (EPSG:3071: 577719.1, 276299.5). This replaces an earlier surrogate that was chosen by a
  water-avoidance rule; every Stage 6 result produced with that surrogate is void and has
  been regenerated. The coordinate lives in one place, `TOWER_LON/TOWER_LAT` in
  `bin/prep_stage6.py`, and `docker/prep_dem30.py` imports it from there.

  Sanity checks at the surveyed position: bare-earth elevation **268.61 m**, sub-cell
  elevation spread **0.287 m** (i.e. classified as land, not water), nearest open-water cell
  **346 m** away, 60 m of relief across the +/-4 km tile.

- **Open water is a land-cover class, not an exclusion.** Lake Kegonsa lies east of the
  tower and is inside the footprint for easterly cases, so it is represented rather than
  masked or tapered away.

  *Detection is a measurement, not a flatness guess.* `docker/prep_dem30.py` aggregates the
  0.4572 m source DEM to 30 m and emits, alongside the mean, the **standard deviation of the
  source elevations within each 30 m cell**. A LiDAR bare-earth surface over open water is
  specular — returns are sparse and interpolated to a level plane — so that spread collapses
  to millimetres, while land keeps centimetres to metres. The histogram is strongly bimodal:

  ```
    sub-cell std (m)    cells
    0.000 - 0.010      12214     <- open water
    0.010 - 0.020        351     <- the gap
    0.020 - 0.050       1252
    0.050 - 0.100       2780
    0.100 - 0.200      11539     <- land
    0.200 - 0.500      23360
  ```

  The threshold sits **in the gap** (`WATER_STD_MAX = 0.02 m`), not at a tuned value, and is
  combined with a +/-1 m band about the modal water elevation (256.64 m). Classifying on
  "flat at 30 m" alone would sweep in ploughed fields and road corridors; classifying on
  sub-cell spread separates them. Water gets `z0 = 1e-4 m`; grass `0.03`; the array `0.20`.

  **Albedo has no pathway and this is not an omission.** FastEddy in this configuration has
  no radiation scheme at all — `surflayerSelector = 1` prescribes the kinematic surface heat
  flux directly — so what albedo would have controlled is subsumed by `htFlux`, which IS
  per-cell (`cuda_surfaceLayerDevice.cu:191` reuses the array when `surflayer_idealsine = 0`,
  and `htFlux` is IO-registered so it survives the restart read). The built-in
  `surflayer_offshore` wave-roughness parameterisations are a **global** switch and cannot be
  applied to water cells only, so per-cell `z0` is used instead.

  Rotation check, same tower, same maps:

  | wind from | water | solar array |
  |---|---|---|
  | 270 deg (westerly) | 840-1050 m **downwind** | 150-240 m **upwind** |
  | 90 deg (easterly) | 840-3300 m **upwind**, 62% of the upwind half | **downwind** |

- The solar array polygon is an assumption (100 m east-west x 400 m north-south, centred
  200 m from the tower at bearing 270 deg), from CLAUDE.md's "~100 m x 400 m" and "array at
  270 deg". It is defined as a **geographic** object and rotates with the domain. It was
  previously specified as an "upwind distance", which silently moved the array whenever the
  wind direction changed — wrong for a multi-direction corpus, where the array must be
  upwind for a westerly and downwind for an easterly.
- The county-wide elevation range (237 m) is much larger than the ~30 m of relief across
  the tower's source area; do not use the county statistics as a site-scale expectation.
- Stage 6 resamples this into a **wind-aligned** frame and supplies rotated `lat(y,x)` /
  `lon(y,x)` to `GeoSpec.py`. Reprojection from EPSG:3071 happens there.

---

## CONUS404

Not yet retrieved. When it is, record here: variables, extraction bounds, time range,
source endpoint, and access date. Same rule — extracts are gitignored.
