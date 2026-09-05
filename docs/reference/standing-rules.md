# Standing rules

Nine rules. Each one cost real GPU time at least once before it was written down. Code
comments cite them by number ("standing rule 5"), so the numbering is stable.

## 1. Validate the state the model actually loaded, never the config handed to it

Five times a configured value was not the value the model ran with. Each produced a
plausible wrong number rather than an error, and each was found by looking at an artifact
instead of a setting.

| configured | what the model actually had | found by |
|---|---|---|
| a per-cell convective `htFlux` map | all zeros; the case would have run neutral silently | reading the field out of the file |
| receptor at 10 m | every footprint landed on the level nearest the 30 m default | reading the call, not the flag |
| `dt` inside the stability limit | inside the accuracy-vs-stability window: exit 0, no message, near-surface `w` is acoustic noise | `k0/k1` on the dump |
| `surflayer_wth = −0.012` in the `.in` | `+0.000000` in every dump | reading `htFlux` out of the dump |
| a `.in` template in the image | absent; `.dockerignore` took it. 81 cases, 0 records, on all 8 machines | reproducing the build locally |

Every parameter that is also an IO-registered field is a property of the restart file, not
of the `.in`: `htFlux`, `z0m`, `z0t`, `tskin`, `topoPos`, `zPos`, `xPos`, `yPos`. For those
the `.in` is a request and the restart is the answer. The rule is wider than that list. A
default silently taken, a parameter silently reverted for being out of range, and a `dt`
inside the accuracy window are all the same shape.

So every step asserts on the artifact it produced, and on the quantity, not on the presence
of a file.

## 2. A check that stubs the thing it is checking is a statement about the harness

The 8-machine dry run was green while the image had no `.in` template, because `--stub`
replaces the screener and the case and opens no file the case path reads. Every seed-side
artifact was asserted at build time; no case-side one was.

Fixes: the image build asserts the case path's inputs and that the template has `Nz = 122`;
the corpus driver refuses at startup by name; and one real case runs with only the LES and
LPDM stubbed (`STUB_LES=1`, about 4 minutes, no GPU). That last one is what closes the gap.

## 3. A diagnostic is only as scale-free as its reference

A diagnostic whose denominator or reference varies with anything but the quantity being
measured reports that variation as signal. It fails quietly every time: the number stays
finite, the check runs, the verdict prints. Four instances, and the fourth is the fix that
was applied to the second.

| diagnostic | the reference that moved | what it reported instead |
|---|---|---|
| `z_i` at 5% of the running TKE peak | the peak falls with `u*²` on the inertial oscillation | +11.67 %/h of deepening while three independent depths said +1.71 to +2.33 |
| `TKE/u*²` over the whole column | the column is mostly free atmosphere, so it scales with `z_i/H` | two rungs 44% apart in a quantity that is 5.7% apart when scale-free, and it failed a seed |
| `k0/k1`, first-to-second-level `w` variance | both levels collapse together when the layer dies | 0.442, a clean pass, through a stable seed whose boundary layer had died |
| `TKE_BL/u*²`, the fix applied to row 2 | `u*²` falls ~10 %/h on the oscillation and the averaging depth `z_i` entrains upward | +22.5 %/h "drifting" on a rung whose absolute BL TKE is flat, and the wrong sign on another |

The fix is one of two things, never a looser threshold. Make the reference scale-free
(`z_i` moved to a fixed 0.01 m²/s² threshold), or pair the check with one that fails
differently. `docker/turb_alive.py` runs everywhere `k0/k1` runs and answers "is there any
turbulence at all?" A SKIP from it is not a PASS.

The diagnostic for the diagnostic is its own sampling error. `bin/seed_stationarity.py`
reports each trend's AR(1)-corrected standard error and `n_eff`, and returns INDETERMINATE
rather than PASS or FAIL when the threshold sits inside that spread.

## 4. A tolerance must be the size of the failure it is looking for

Not the smallest number you can write down. `--strict-rel` exists to catch losing a 5 s dump
and was scored against 1e-6 s, so it failed a correct production run on half a millisecond,
at stage 7, after 74 minutes of GPU. One lost dump exceeds that deficit by a factor of
10,016. Tolerances are now expressed in the unit of the thing they protect (dumps, output
intervals), and the margin is printed on success too. A configuration designed to sit at
zero margin leaves no evidence of how close it came unless the passing path says so.

The same rule with the sign flipped: a tolerance must also reject nonsense, not only excess.
`t_start_s` and `t_end_s` were rounded to different precisions, which gave sub-100 ms days a
negative duration, and the load-balance check passed on "−73% imbalance" and "108% of wall
time saved".

## 5. A tolerance measured from one difference is not a tolerance

It has one degree of freedom, and its own sampling error is the size of the thing it bounds.
Phase E said DIFFERS on a 2-group floor and PASSED at p ≈ 0.54 on a 10-group one: a factor
of 5 in the estimated floor from nothing but the number of groups. Use `--cover-groups N`
with N ≥ 8, quote the standard error, and never quote a tolerance without saying how many
independent realisations went into it.

Score a second moment against its own sampling spread, not against a number you picked. The
convective B6 gate first used a fixed 3e-2 on `sigma_w²` and reported DIFFERS at 3.587e-2.
Re-scored against the block standard error of the same field, that is 0.38× one
realisation's own spread. The reframing is not a loosening. It is what located the problem.

## 6. Validation must exercise the production code path and the production regime

- **Wrong regime.** The neutral well-mixed gate passed a closure carrying nine turnovers in
  `sigma_w²`, because the floor is nearly inert neutrally (receptor factor 1.000 there, 1.59
  convectively). A regime where a component is inert is no evidence at all about that
  component.
- **Wrong code path.** `stage4_wellmixed.py` carried its own copy of the `sigma_w` floor,
  which had drifted from the production one. Gates import the production function. They
  never reimplement it.
- **Quote the no-op control beside the result.** The convective failure was localised by
  scoring the same window with no floor at all. A gate result without its control says only
  "a number came out".

## 7. Assert on the artifact, not on the exit status

Analyses get piped into `grep`, so bash reports grep's status, and a Python traceback lands
quietly in a redirected `.txt`. A `SyntaxError` in `stage5_footprint.py` was launched six
times that way. Every step checks the JSON it was supposed to write, and `bin/preflight.sh`
parses every Python entry point and shell driver before a campaign starts (about 10 s; the
drivers refuse to run without it).

Same rule for exit codes. `run_seed.sh` was `[ "$VERDICT" = "PASS" ] || exit 1` and returned
1 for all thirty seeds. A status identical for every outcome discriminates nothing, in the
dangerous direction. The verdict lives in the artifact.

## 8. One run per directory, or it is not a series

FastEddy names a dump `<outFileBase>.<step>`, so a directory that has held two runs holds two
families with overlapping step numbers. Sorting the union on the step interleaves them into a
"history" with two different states at the same time. Every glob of a dump directory filters
on one base name.

## 9. Every script greps for `CORRUPTED` and tests `np.isfinite(...).all()` first

`inf` is not NaN, and a NaN passes every `>` comparison.
