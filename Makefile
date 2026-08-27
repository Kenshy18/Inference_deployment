PYTHON ?= python3
NPM ?= npm

.PHONY: check check-python check-postprocess check-orchestration \
	check-overlay check-deployment check-gui compile clean-caches

check:
	$(PYTHON) scripts/check_repository.py

check-python:
	$(PYTHON) scripts/check_repository.py postprocess orchestration overlay deployment

check-postprocess:
	$(PYTHON) scripts/check_repository.py --no-compile postprocess

check-orchestration:
	$(PYTHON) scripts/check_repository.py --no-compile orchestration

check-overlay:
	$(PYTHON) scripts/check_repository.py --no-compile overlay

check-deployment:
	$(PYTHON) scripts/check_repository.py --no-compile deployment

check-gui:
	$(PYTHON) scripts/check_repository.py --no-compile gui

compile:
	$(PYTHON) scripts/check_repository.py --compile-only

clean-caches:
	find InstanceSegmentation orchestration postprocess overlay deployment \
		deployment_tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find InstanceSegmentation orchestration postprocess overlay deployment \
		deployment_tests -type d -name .pytest_cache -prune -exec rm -rf {} +
