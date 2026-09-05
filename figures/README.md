# Figures

| | |
|---|---|
| `cone/` | the nine pair figures on `corpus_cone.h5`, the training set |
| `cone_mask_effect.png` | how the cone was derived and what it removed |
| `poster/` | the five final figures (600 dpi): showcase, generative, sectors, distributions, domain |

How to read each figure, and how to regenerate it: `docs/corpus/figures.md` and
`docs/emulator/results.md`. The nine figures on `corpus_raw.h5` are one command away:

    docker/pyrun.sh bin/fig_corpus_pairs.py --h5 corpus/corpus_raw.h5 --outdir figures/raw
