#!/usr/bin/env bash
# Resample the Dane County 2024 bare-earth LiDAR DEM (0.4572 m, EPSG:3071) to a 30 m
# tile centred on the Kegonsa tower.
#
# -r average, not nearest: this is a 65.6x linear reduction (0.4572 -> 30 m). Nearest
# would alias single LiDAR posts into LES surface elevations; averaging is the correct
# sub-grid treatment for a surface the model feels only through its mean height.
#
# The tile is deliberately larger than any single rotated domain (+/-4 km covers the
# 4380 x 1500 m box at ANY rotation about the tower, whose farthest corner is 3.37 km
# away), so Stage 6 can cut a wind-aligned window out of it without touching the 4.4 GB
# source again.
set -euo pipefail
SRC=data/dem/Dane2024_beDEM45cm_ELm_WTM.tif
OUT=data/dem/kegonsa_30m_wtm.tif
# Tower position, EPSG:3071 (see data/README.md for how this was derived and its uncertainty)
# Single source of truth for the tower coordinate: bin/prep_stage6.py.
TX=$(python3 -c "
import sys; sys.path.insert(0,'bin')
from pyproj import Transformer
from prep_stage6 import TOWER_LON, TOWER_LAT
t=Transformer.from_crs('EPSG:4326','EPSG:3071',always_xy=True)
x,y=t.transform(TOWER_LON, TOWER_LAT); print(f'{x:.3f} {y:.3f}')")
read -r X Y <<<"$TX"
R=4000
# Snap the window to whole 30 m cells so the grid is reproducible.
read -r X0 Y0 X1 Y1 <<<"$(python3 -c "
import math
x,y,r=$X,$Y,$R
f=lambda v: math.floor(v/30.0)*30.0
print(f(x-r), f(y-r), f(x-r)+30*round(2*r/30), f(y-r)+30*round(2*r/30))")"
echo "tower EPSG:3071  X=$X  Y=$Y"
echo "window  $X0 $Y0 -> $X1 $Y1   ($(python3 -c "print(int(($X1-$X0)/30))") x $(python3 -c "print(int(($Y1-$Y0)/30))") cells @ 30 m)"
gdalwarp -overwrite -t_srs EPSG:3071 -te "$X0" "$Y0" "$X1" "$Y1" \
         -tr 30 30 -r average -ot Float32 -co COMPRESS=LZW -co TILED=YES \
         "$SRC" "$OUT"
gdalinfo -stats "$OUT" | grep -E 'Size is|Pixel Size|Minimum=|Upper Left|Lower Right'
