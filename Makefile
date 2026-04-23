SHELL := /bin/bash

config ?= poc_scp_probe
ovr ?=

.PHONY: set show-config

set:
	pip install --upgrade pip
	pip install -e .
	pip install --no-deps "transformers==5.5.4" "trl>=0.15.0"
	python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"

# Quick sanity-print of the resolved Hydra config for the active PoC.
show-config:
	python -c "from hydra import compose, initialize_config_dir; from omegaconf import OmegaConf; from pathlib import Path; \
cfg_dir = Path.cwd() / 'configs'; initialize_config_dir(version_base=None, config_dir=str(cfg_dir)).__enter__(); \
cfg = compose(config_name='$(config)', overrides=('$(ovr)'.split() if '$(ovr)' else [])); \
print(OmegaConf.to_yaml(cfg, resolve=True))"
