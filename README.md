# Texas-Hold-Em
AI 4444

## Active entry points

- `python3 -u bot.py` connects Learned V1 to the local poker server.
- `python3 train_model.py --help` trains and benchmarks the Learned V1 approach.
- `python3 learned_v2.py --help` runs the newer model experiments.
- `python3 -m cfr_lab.run --help` runs the optional CFR experiment.
- `python3 betting_test.py --help` compares betting and payout experiments.
- `python3 -B -m unittest discover -s tests -q` runs the tests.

The live model and frozen source snapshots are under `baselines/`. The
reference Learned V1 run and the latest completed shove audit are under
`experiments/`. The older scripts, logs, checkpoints, and smoke runs are
in `legacy-poker.zip`. It also includes the pre-consolidation versions of
the changed files. Extract into a separate directory to inspect the archive.
