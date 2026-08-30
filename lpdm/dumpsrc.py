"""Where a dump comes FROM, separated from what a reader does with it.

WHY THIS EXISTS. `lpdm/fields.py:FieldSet` and `lpdm/les_stats.py:window_stats` are the two
functions every gate in this project runs through, and both open a netCDF file by path.
That is correct and it is also the only reason a footprint requires ~10 GB of scratch per
window: FastEddy encodes the fields, the filesystem stores them, and the readers decode
them straight back. Nothing else in the pipeline needs the file.

The in-process hook removes the file. It cannot remove the readers -- PROJECT_BRIEF.md's standing
rule is that a gate exercises the PRODUCTION code path, so re-implementing `window_stats`
against arrays would produce a second statistics function whose agreement with the first is
an assumption rather than a fact, and this project has already paid twice for exactly that
shape (`stage4_wellmixed.py` carried its own drifted copy of the sigma_w floor).

So the readers keep their single implementation and gain ONE indirection: they call
`open_dump(handle)` instead of `Dataset(path)`. A handle that is a string is a netCDF path
and behaves exactly as before, byte for byte. A handle that is a `MemDump` is a snapshot
already in RAM -- streamed from the live LES -- and presents the small, closed slice of the
`Dataset` interface those readers actually use:

    ds["name"][:]        variable data
    "name" in ds         presence test, for the fields ioLPDMmode omits
    with ... as ds:      context management

That list is not a guess. It is every netCDF access in both files, enumerated before this
module was written; `bin/test_dumpsrc.py` re-enumerates it and FAILS if a reader grows a
new one, because a silently missing attribute on a duck type is the same class of failure
as a silently defaulted parameter -- a plausible wrong number instead of an error.

THE TRANSPARENCY TEST IS THE POINT. `bin/test_dumpsrc.py` reads real dumps, wraps them as
`MemDump`s, and requires the two readers to return BIT-IDENTICAL results through both
handles. Until that passes, the indirection is not established to be free.
"""
from __future__ import annotations

import re

import numpy as np
from netCDF4 import Dataset

# Everything the two readers ask a dump for. Enumerated from the source, not assumed; the
# test asserts the readers ask for nothing outside it.
DUMP_VARS_3D = ("u", "v", "w", "theta", "TKE_0")
DUMP_VARS_2D = ("fricVel", "z0m", "invOblen", "htFlux")
DUMP_VARS_GEOM = ("xPos", "yPos", "zPos", "topoPos")


class MemDump:
    """One snapshot held in RAM, quacking like the netCDF Dataset the readers open.

    `arrays` maps variable name -> ndarray. 3-D fields are (nz, ny, nx) and 2-D ones are
    (ny, nx) -- i.e. ALREADY SQUEEZED, where netCDF carries a leading singleton time
    dimension. Every reader squeezes anyway, so the two are interchangeable; nothing here
    re-adds an axis just to have it removed.

    `step` is the absolute timestep, which is what the netCDF path's trailing integer
    carries and what the readers use to build the time axis. It is passed rather than
    parsed because a streamed snapshot has no filename to parse.

    Arrays are held by REFERENCE and are not copied or freed on exit. A window's worth of
    snapshots is ~10 GB; copying it per read would cost more than the file this exists to
    delete.
    """

    __slots__ = ("_a", "step")

    def __init__(self, arrays, step):
        self._a = dict(arrays)
        self.step = int(step)

    # -- the reader interface, and deliberately nothing more ------------------------------
    def __getitem__(self, k):
        return self._a[k]

    def __contains__(self, k):
        return k in self._a

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def variables(self):
        return self._a

    # -- convenience for building one from a real dump, which is how the test works -------
    @classmethod
    def from_netcdf(cls, path, names=None, dtype=np.float32):
        """Read a real dump into RAM. Used by the transparency test and by --ring-replay.

        `names=None` takes every variable the readers know about that the file actually
        carries -- ioLPDMmode omits several, and geometry appears only in the first file
        of a run, so 'missing' is normal and must not be an error here.
        """
        want = names or (DUMP_VARS_3D + DUMP_VARS_2D + DUMP_VARS_GEOM)
        out = {}
        with Dataset(path) as ds:
            for n in want:
                if n in ds.variables:
                    out[n] = np.squeeze(np.asarray(ds[n][:], dtype=dtype))
        return cls(out, step_of(path))


def step_of(handle) -> int:
    """The absolute timestep of a dump handle, whichever kind it is."""
    if isinstance(handle, MemDump):
        return handle.step
    return int(re.search(r"\.(\d+)$", str(handle)).group(1))


def open_dump(handle):
    """Open a dump handle: a path opens a netCDF file, a MemDump is already open."""
    if isinstance(handle, MemDump):
        return handle
    return Dataset(handle)
