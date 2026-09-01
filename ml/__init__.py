"""FNO footprint emulator for the Kegonsa tower: loader, model, losses, metrics, trainer.

Trains on corpus/corpus_cone.h5 (see corpus/README.md and docs/ML_TARGETS.md). The design
is an FNO predicting a RESIDUAL on Kljun in asinh space, conditioned on the six Kljun
scalars by FiLM. THE TEST SPLIT IS NEVER READ BY ANYTHING IN THIS PACKAGE: ml/data.py
refuses it unless an explicit flag is passed, and nothing here passes it.
"""
