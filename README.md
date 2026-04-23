# scp_stage4_main

Self-Collapse Probing (SCP) framework — Stage 4. Implements the algorithm proposed in
`cap/005/proposal/main.pdf` (Algorithm 1).

**Current status: PoC for Stage A (Self-Collapse Probe).** The full K-round SCP loop
(Stage B selection + teacher correction + Stage C main-LoRA update) and the RL
post-training stage will be layered on top of this scaffolding once the Stage A
collapse signal is characterized.

## What Stage A does

For a subset D_k of English monolingual sentences:

1. Generate self-translations T_k from the current model (greedy).
2. Measure pre-probe QE: `q_before(s) = QE(s, t)` (reference-free).
3. Attach a **fresh** aggressive probe LoRA (small rank, high lr, no
   regularization) and train it on `(D_k, T_k)`.
4. Regenerate translations T'_k with the probed model.
5. Measure post-probe QE and compute `dQE(s) = q_before(s) - q_after(s)`.
6. Discard the probe LoRA.

Vulnerable sentences are those with `dQE(s) > delta`. The PoC is about
discovering under which `(r_p, lr_p, E_p, temperature, subset_size)` settings
this collapse signal is most informative.

## Layout

- `configs/` — Hydra config tree. Drop new YAMLs into `configs/data/` or
  `configs/probe/` to extend without touching code.
- `src/` — reusable library:
  - `data.py` — unified CSV / HF dataset loader.
  - `prompt_utils.py` — stage3-compatible prompt rendering and template hashing.
  - `model_io.py` — Unsloth loader and probe-LoRA attach/reset lifecycle.
  - `generate.py` — batched greedy generation.
  - `probe.py` — response-only-loss SFT loop for the probe LoRA.
  - `qe.py` — COMET-Kiwi primary + optional reference-based analysis.
  - `scp_a.py` — end-to-end Stage A orchestrator.
- `notebooks/scp_poc_self_collapse.ipynb` — main PoC analysis notebook.
- `data/test.csv` — 546 economic-domain EN→KO pairs (copied from stage3_it).

## No hard-coding

Everything the user will iterate on is a config:

- Model → `configs/model/*.yaml` (currently `alwaysgood/qwen3-it`, `qwen35-it`,
  `gemma4-it`; extend for new bases).
- Data → `configs/data/*.yaml`. `testcsv.yaml` for the PoC,
  `sec_10k_mono.yaml` and `earnings_call_mono.yaml` for full training.
  Dropping a Reuters YAML in the same directory is sufficient.
- Probe hparams → `configs/probe/aggressive.yaml` and a grid in `sweep.yaml`.
- Generation → `configs/generation/greedy.yaml`.
- QE backend → `configs/qe/comet_kiwi.yaml`.

## COMET + transformers 5.5.4 incompatibility

`unbabel-comet` 2.2.7 (the current latest) pins
`transformers>=4.17,<5.0`. The unsloth / stage-3 SFT stack requires
`transformers==5.5.4`. **You cannot install both in the same venv** — pip
resolution fails, and `--no-deps` bypasses the pin but risks breaking COMET
at runtime (transformers 5.0 removed several symbols COMET's encoder
wrappers use).

Workaround: run COMET in a **separate venv** and point the QE backend at it
via the `COMET_PYTHON` environment variable. `src/qe.py` auto-detects this
and calls COMET over subprocess — no COMET import happens in the unsloth
process.

```bash
python3 -m venv ~/.venvs/comet
~/.venvs/comet/bin/pip install --upgrade pip
~/.venvs/comet/bin/pip install 'unbabel-comet>=2.2.7'
export COMET_PYTHON=$HOME/.venvs/comet/bin/python
```

If `COMET_PYTHON` is unset, `build_qe_primary()` returns a dummy zero
scorer with a warning, so the notebook still runs and you can inspect
translations — but the `delta_qe` signal will all be zero. `test.csv`
carries gold Korean references, so you can optionally wire in
`build_chrf_fallback()` (reference-based chrF) as a PoC proxy, noting it is
not the same signal as the reference-free QE the algorithm actually uses.

## Notebook workflow

Open `notebooks/scp_poc_self_collapse.ipynb`. It:

1. Bootstraps the environment (unsloth + transformers 5.5.4).
2. Loads the active Hydra config via `compose`.
3. Runs a single baseline + probe pair with default settings.
4. Sweeps `(rank, learning_rate, num_train_epochs)` from
   `configs/probe/sweep.yaml`.
5. Aggregates per-sentence `dQE` and plots the vulnerable-sentence rate as a
   function of probe configuration, sector, and source type.

## References

- Paper: `cap/005/proposal/main.pdf` §4.1 (SCP) and Algorithm 1.
- Upstream: `scp_stage3_it` — SFT-trained translator bases reused here.
