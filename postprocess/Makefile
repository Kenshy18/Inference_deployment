PYTHON ?= python3

.PHONY: install test unit compile smoke clean

install:
	$(PYTHON) -m pip install -e . --no-deps

test: unit compile

unit:
	$(PYTHON) -m unittest discover -s tests

compile:
	$(PYTHON) -m compileall -q \
		common contracts preprocessing nms cut_detection tracking \
		evaluation artifacts visualization face_privacy production \
		diagnostics tests run_pipeline.py

smoke:
	$(PYTHON) -m unittest \
		tests.test_architecture.ArchitectureTests.test_cut_detection_can_be_replaced_in_raw_to_output_e2e

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
