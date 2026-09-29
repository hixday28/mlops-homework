.PHONY: install tokenize check repro pull
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
install:
	$(UV) sync --locked
tokenize:
	$(UV) run --locked python -m src.tokenize_data
check:
	@PATH="$(dir $(UV)):$$PATH" bash tests/check.sh
repro:
	$(UV) run --locked dvc repro
pull:
	$(UV) run --locked dvc pull
