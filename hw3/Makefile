.PHONY: install fetch repro v1 v2 diff dag diversity contamination check test push
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
install:
	$(UV) sync --locked
fetch:
	$(UV) run --locked python -m src.fetch_source
repro:
	$(UV) run --locked dvc repro
v1:
	$(UV) run --locked python scripts/set_version.py v1
	$(UV) run --locked dvc repro
v2:
	$(UV) run --locked python scripts/set_version.py v2
	$(UV) run --locked dvc repro
diff:
	$(UV) run --locked dvc metrics diff hw3-v1 hw3-v2 --md
dag:
	$(UV) run --locked dvc dag
diversity:
	$(UV) run --locked python -m src.diversity
contamination:
	$(UV) run --locked python scripts/check_contamination.py
check:
	@PATH="$(dir $(UV)):$$PATH" bash tests/check.sh
test:
	$(UV) run --locked python -m unittest discover -s tests -v
push:
	$(UV) run --locked dvc push
