# Motivation

An eddy-covariance tower measures the turbulent flux of heat, water or CO₂ crossing a plane
at its sensor height. The **flux footprint** says where that flux came from. It is a weight
over the upwind surface that gives how much each patch contributed to what the sensor saw in
that half hour. To interpret a tower over heterogeneous ground you need its footprint.

The standard tool is the Kljun et al. (2015) parameterisation (FFP). It takes six scalars
that describe the half hour: boundary-layer depth `h`, friction velocity `u*`, crosswind spread
`σ_v`, Obukhov length `L`, wind speed and wind direction. It returns a two-dimensional
footprint in a fraction of a second. It is a fit to Lagrangian dispersion simulations over flat,
homogeneous terrain. It knows nothing about any particular site.

## The site is not flat and homogeneous

The tower at the UW-Madison Kegonsa Solar Array is inside the array. The panels extend 60 m
east and west of it, 250 m north and 100 m south: 4.2 ha of photovoltaic panels on cropland,
with Lake Kegonsa to the east and a tree line in the domain. The array is rougher than the
cropland it replaced (`z0` 0.25 m against 0.10) and warmer in daytime (a virtual heat flux
1.376× the cropland reference). Both change the near-field flow and therefore the footprint.
Both are averaged away by Kljun's flat-terrain fit.

For northerly flow about 30% of the footprint is on the array. That is where a site-specific
footprint and Kljun's generic one disagree most, and where the difference decides what the
tower's flux means.

## What the project builds

A **site-calibrated emulator** for this one tower. It takes exactly Kljun's six scalars and
returns the two-dimensional footprint that a large-eddy simulation of this site would produce
under those conditions. It costs as little to run as Kljun and it knows the site.

The training targets come from an offline pipeline that is never part of inference:

1. **FastEddy**, NCAR's GPU large-eddy simulation model, runs the atmospheric boundary layer
   over the real terrain and land cover at 30 m resolution, forced by a real HRRR analysis
   for the hour in question.
2. A **backward Lagrangian particle model** releases particles at the receptor and follows
   them back through the stored turbulent fields to where they touched the surface. The
   surface-normal approach rate at touchdown gives the flux footprint.
3. **1366 such cases**, one per day over five years, make the corpus. The days are drawn from
   the weather, not from a parameter sweep.

Two emulators are trained on it. A Fourier neural operator (FNO) predicts the correction to
Kljun. A conditional flow-matching model (CFM) does the same and also gives the spread between
turbulence realisations.

## What it is not

- **Not transferable.** The emulator is calibrated to one tower and one grid. Zero transfer to
  other sites is a stated limitation. Requests to add scope were refused throughout.
- **Not a tower measurement.** The model receptor is at 30 m. The instrument is at about 10 m.
  This is a resolution decision. At a 10 m receptor on this grid the footprint peak did not
  respond to meteorology at all (48 m in all three validation targets, max/min 1.00×), because
  the near field was closure output rather than resolved turbulence. A 30 m receptor is
  1.52 filter widths above the surface and its peak moves with the weather (144 and 288 m
  between two pre-registered targets, six times the sampling floor). Comparisons with the
  tower use a Monin-Obukhov translation.
- **Not defined at night.** There are no stable cases. A stable boundary layer collapses at
  this grid ([limitations](../limitations-and-future-work.md)).

## The result in one line

On the untouched 2025 test split (294 cases), both emulators beat Kljun on every metric. The
CFM's mean puts the peak within 30.6 m RMSE of the LES against Kljun's 104 m and the centroid
within 69 m against 129 m. It halves the sliced Wasserstein distance (40.9 against 75.0 m).
The FNO is close behind at 33.1 / 92.8 / 53.5 m. The full tables are on the
[results](../emulator/results.md) page.

## How to read this site

- [Method overview](method-overview.md) is the map of the pipeline.
- [The site](site.md) is what the model knows about Kegonsa.
- The **LES pipeline** section explains how a target is made. The
  [gates](../les/gates-and-diagnostics.md) page explains how a target is checked.
- **Corpus** is the dataset. **Emulator** is the model, its training and its results.
- **Development history** is how every decision was reached, pass by pass, including what
  was tried and failed. The [standing rules](../reference/standing-rules.md) collect those
  lessons.
