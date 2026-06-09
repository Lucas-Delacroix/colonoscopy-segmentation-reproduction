UV ?= uv
MODEL ?=
RECREATE_ENVS ?= 0
UPDATE_ENVS ?= 0
CONDA_SOLVER ?= libmamba
MODEL_ARG = $(if $(MODEL),--only $(MODEL),)
RECREATE_ENVS_ARG = $(if $(filter 1 true yes,$(RECREATE_ENVS)),--recreate-existing,)
UPDATE_ENVS_ARG = $(if $(filter 1 true yes,$(UPDATE_ENVS)),--update-existing,)
CONDA_SOLVER_ARG = $(if $(CONDA_SOLVER),--solver $(CONDA_SOLVER),)

.PHONY: setup train predict table2

setup:
	@$(UV) run python scripts/download_dataset.py
	@$(UV) run python scripts/download_hardnet_weights.py
	@$(UV) run python scripts/download_kingnet_weights.py
	@$(UV) run python scripts/download_mit_weights.py
	@$(UV) run python scripts/download_pvt_weights.py
	@$(UV) run python -m scripts.prepare_upstream_kvasir --force
	@$(UV) run python -m scripts.bootstrap_upstream_repos --force $(MODEL_ARG)
	@$(UV) run python -m scripts.apply_upstream_overlays $(MODEL_ARG)
	@$(UV) run python -m scripts.create_upstream_envs $(MODEL_ARG) $(RECREATE_ENVS_ARG) $(UPDATE_ENVS_ARG) $(CONDA_SOLVER_ARG)

train:
	@test -n "$(MODEL)" || (echo 'Usage: make train MODEL=cascade'; exit 1)
	@$(UV) run python -m scripts.run_upstream $(MODEL)

predict:
	@test -n "$(MODEL)" || (echo 'Usage: make predict MODEL=cascade'; exit 1)
	@$(UV) run python -m scripts.run_upstream $(MODEL) export

table2:
	@$(UV) run python -m scripts.evaluate_predictions
