# Quick start

Five paths, from cheapest to most expensive. Each one is independent of the others.

## 1. Look at the corpus and the results (no GPU)

```bash
git clone https://github.com/tyatharva/flux-kegonsa.git && cd flux-kegonsa
bin/fetch_assets.sh corpus                      # corpus_cone.h5 + corpus_raw.h5 -> corpus/ (76 MB)
docker build -t flux-fasteddy:cuda118 .         # the analysis image (the host Python has no h5py)
docker run --rm -v "$PWD":/w -w /w -u $(id -u):$(id -g) -e MPLCONFIGDIR=/tmp/mpl \
    flux-fasteddy:cuda118 python3 bin/fig_corpus_pairs.py --h5 corpus/corpus_cone.h5 --outdir /tmp/figs
```

The scored record is already in the repository. `results/ml_cfm/final_recipe/metrics_test.md`
is the headline table. `results/ml/eval/final_ensemble/eval.md` is the FNO's val evaluation and
`results/ml_cfm/eval/final/eval.md` is the CFM's. The [results](../emulator/results.md) page
explains them.

## 2. Run the emulator (one GPU, minutes)

```bash
conda env create -f ml/environment.yml && conda activate LESNet
bin/fetch_assets.sh corpus weights              # + FNO seeds 0-4 (542 MB) and CFM seeds (68 MB)
python -m ml.evaluate --ckpt results/ml/final/seed{0..3}/best.pt --tag check    # FNO ensemble on val
python -m ml_cfm.final_recipe --split val       # both emulators under the frozen recipe, on val
python -m ml_cfm.report_metrics --split val     # the reporting metrics
```

Outputs are written under `results/ml/eval/check/` and `results/ml_cfm/final_recipe/`. To
evaluate on the test split, pass `--split test --allow-test`. Every such read is logged in
`results/ml/loader_audit.jsonl`.

## 3. Retrain the emulators (one RTX 4080, about 2 hours for everything)

The final configurations are the `config` blocks of the tracked `run.json` files.

```bash
python -c "import json;json.dump(json.load(open('results/ml/final/seed0/run.json'))['config'],open('/tmp/fno.json','w'))"
for s in 0 1 2 3 4; do
  python -m ml.train --config /tmp/fno.json --set seed=$s --set save_pred=true --out results/ml/final/seed$s
done
python -m ml.evaluate --ckpt results/ml/final/seed*/best.pt --tag final_ensemble

python -m ml_cfm.campaign --runs results/ml_cfm/final/runs.json -K 3 --outdir results/ml_cfm/final --tag final \
    --base '{"epochs": 500, "save_samples": 32, "steps_final": 16}' --baseline-prefix seed
```

`ml/phase1.py`, `ml/phase2_optuna.py` and `ml/final.py` reproduce the exploration, the Optuna
study and the seed selection described in [training](../emulator/training.md). The Optuna
database is not shipped, so `ml.final` needs `--trial` or the config above. `ml_cfm/run_all.sh`
runs the CFM's whole study unattended.

## 4. Run one LES case on the workstation (one GPU, about an hour)

```bash
fasteddy/fetch.sh                               # NCAR v5.0.1 + fasteddy/patches -> FastEddy-model-5.0.1/
docker/run.sh ./docker/build_fasteddy.sh        # compile for the local GPU (sm_89 by default; SM=sm_86 ...)
bin/fetch_assets.sh seeds                       # the 30 restarts (2.1 GB) -> seeds/*/return/seed_restart.nc
bin/preflight.sh                                # parse every entry point, host and container
STUB_LES=1 bin/run_corpus_case.sh 2023-01-18T18:00 stubcheck   # the whole path with the LES stubbed, ~4 min, CPU
bin/run_corpus_case.sh 2023-01-18T18:00         # the full case: sounding, seed, restart, LES, LPDM, record
```

The case needs network access to the HRRR archive for its sounding (about 170 MB per case) and
about 13 GB of host RAM for the LPDM's field cache. It writes `pairs_npz/case_2023011818.npz`,
which `bin/check_npz.py` validates. [Case generation](../les/case-generation.md) explains each
stage. [Configuration](../les/configuration.md) describes the grid it runs on.

## 5. Regenerate the seed library or the corpus (rented GPUs)

```bash
docker/build_image.sh                           # the deployment image, with the seeds baked in
# then, on a rented box with the image pulled:
verify && run_seeds --gpu-count 16              # the 30-seed library, ~1 h on 16 x RTX 5090
verify && nohup run_corpus --machine 0 --out /out &   # one eighth of the corpus, ~12 h on 8 x RTX 5090
```

[Deployment](../les/deployment.md) has the Vast.ai procedure, the sizing and the consolidation
of the eight machines' output into the two HDF5 files. The failed days of the existing corpus
are named in `corpus/provenance/manifests/`. The hour draw is seeded from the date, so a top-up
reproduces exactly the cases that would have been there.

## Tests

```bash
bash bin/test_corpus_machine.sh                 # the orchestrator, stubbed, 15 s, no GPU
docker/pyrun.sh bin/test_bl_depth.py            # and test_displacement, test_floor_health, test_kljun_adapter,
                                                #     test_negative_lobes, test_sgs_floor, test_sounding
python bin/test_ml_data.py; python bin/test_ml_model.py; python bin/test_cfm.py     # in LESNet
```

Nine further tests (`test_dumpsrc`, `test_estimator`, `test_gpu_lpdm`, `test_lpdmonline`,
`test_parallel_lpdm`, `test_ringsrc`, `test_streaming`, `test_toolkit_parity`, `test_unchained`)
need LES fields and run after a case has produced a window. Each has a results file from its
last run. [Scripts](../reference/scripts.md) lists them all.
