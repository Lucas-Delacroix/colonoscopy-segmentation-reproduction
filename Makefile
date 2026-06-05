UV ?= uv
MODEL ?=

.PHONY: setup train predict table2

setup:
	@$(UV) run python scripts/download_dataset.py
	@$(UV) run python -m scripts.prepare_upstream_kvasir --force
	@$(UV) run python -m scripts.bootstrap_upstream_repos --force
	@$(UV) run python -m scripts.apply_upstream_overlays
	@$(UV) run python -m scripts.create_upstream_envs

train:
	@test -n "$(MODEL)" || (echo 'Usage: make train MODEL=cascade'; exit 1)
	@$(UV) run python -m scripts.run_upstream $(MODEL)

predict:
	@test -n "$(MODEL)" || (echo 'Usage: make predict MODEL=cascade'; exit 1)
	@$(UV) run python -m scripts.run_upstream $(MODEL) export

table2:
	@$(UV) run python -m scripts.evaluate_predictions
