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

- The site of interest (UW-Madison Kegonsa Solar Array, ~43.0 N, ~-89.25 E) falls well
  inside this extent.
- The county-wide elevation range (237 m) is much larger than the ~30 m of relief across
  the tower's source area; do not use the county statistics as a site-scale expectation.
- Stage 6 resamples this into a **wind-aligned** frame and supplies rotated `lat(y,x)` /
  `lon(y,x)` to `GeoSpec.py`. Reprojection from EPSG:3071 happens there.

---

## CONUS404

Not yet retrieved. When it is, record here: variables, extraction bounds, time range,
source endpoint, and access date. Same rule — extracts are gitignored.
