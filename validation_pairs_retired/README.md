# Retired-grid validation records — NOT the corpus

These six records were produced during passes 5-9 on the **16 m (1952 m box)** and **24 m
(2928 m box)** grids. They are the validation record of those passes and they are kept for
that reason. They are **not corpus cases** and they are not in `pairs_npz/`, because a
training loader globbing that directory must not mix 122² rasters of three different cell
sizes: the arrays have identical shape and incompatible meaning, which is the quietest
possible way to corrupt a corpus.

`bin/check_npz.py` fails every one of them, naming the grid — that is the check working.
The production corpus is 122³ @ 30 m only.
