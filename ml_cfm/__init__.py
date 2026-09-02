"""Conditional flow matching anchored on the Kljun prior -- the second emulator, built beside
the FNO in ml/ and importing it as a library (ml.data, ml.features, ml.metrics, ml.evaluate).

    z_t = x_prior + t (x_les - x_prior) + (1 - t) eps,   t ~ U(0,1),  eps ~ N(0, sigma^2)

in the file's asinh target space, x_prior = asinh(kljun / s_target). The model regresses the
velocity (x_les - x_prior) - eps (or x_les itself), and a sample integrates dz/dt = v from
z_0 = x_prior + eps to t = 1. Nothing under ml/ is modified. The test split is never read:
ml.data.load_split refuses it, and nothing here sets the flag that would allow it.
"""
