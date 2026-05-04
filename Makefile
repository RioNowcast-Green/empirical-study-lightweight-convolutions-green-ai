PYTHON := $(shell which python)
PIP := $(shell which pip)

setup:
	@$(PIP) install -r requirements.txt

setup-dev:
	@$(PIP) install -r requirements-dev.txt

setup-experiment:
	@$(PIP) install -r requirements-experiment.txt

format:
	@black .
