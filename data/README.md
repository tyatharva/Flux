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

- **Tower coordinate — SURROGATE, not surveyed.** The surveyed position of the EC tower is
  not in this repository and could not be established from public sources: the published
  descriptions give "3725 Schneider Dr, Stoughton WI", "160-acre Kegonsa campus", "~17 acres
  of panels, 5424 modules", "flux tower ~100 ft", and "west of Lake Kegonsa", but no
  coordinates.

  The first estimate, `-89.2450 / 42.9686`, is **wrong**: the DEM reads 256.64 m there and
  the nearest water cell is 6 m away — that is Lake Kegonsa's surface. 25% of the original
  +/-4 km tile is flat at 256.6 +/- 0.6 m, and its centroid (-89.2504, 42.9652) matches the
  published lake position, which is what identifies it.

  What is used instead, and recorded in `bin/prep_stage6.py`, is
  **`-89.2539 / 42.9419`** — chosen by an explicit rule rather than a guess: the nearest
  land position whose 4380 x 1500 m westerly LES domain contains no water at all. It sits
  810 m from the shore, at 281 m elevation, with 38 m of relief across the domain, which
  matches PROJECT_BRIEF.md's "~30 m of elevation change across the area".

  **This must be replaced with the surveyed coordinate before any Stage 6 result is treated
  as site-specific.** It is a single constant, `TOWER_LON/TOWER_LAT` in
  `bin/prep_stage6.py`, and `docker/prep_dem30.sh` imports it from there.

- The solar array polygon is likewise an assumption (100 m along-wind x 400 m crosswind,
  centred 200 m upwind of the tower), from PROJECT_BRIEF.md's "~100 m x 400 m" and "array at
  270 deg". Also a single constant block in `bin/prep_stage6.py`.
- The county-wide elevation range (237 m) is much larger than the ~30 m of relief across
  the tower's source area; do not use the county statistics as a site-scale expectation.
- Stage 6 resamples this into a **wind-aligned** frame and supplies rotated `lat(y,x)` /
  `lon(y,x)` to `GeoSpec.py`. Reprojection from EPSG:3071 happens there.

---

## CONUS404

Not yet retrieved. When it is, record here: variables, extraction bounds, time range,
source endpoint, and access date. Same rule — extracts are gitignored.
