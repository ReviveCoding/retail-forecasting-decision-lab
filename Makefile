PYTHON=python
CONFIG=configs/pipeline.yaml

run:
	$(PYTHON) -m src.run_pipeline --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf outputs reports data/processed data/interim .pytest_cache

run-full:
	$(PYTHON) -m src.run_pipeline --config $(CONFIG) --full
