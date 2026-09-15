.PHONY: install inspect check test clean
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
install:
	$(UV) sync --locked
inspect:
	@$(UV) run --locked python -m src.inspect_model
check:
	@PATH="$(dir $(UV)):$$PATH" bash tests/check.sh
test:
	@$(UV) run --locked python -m unittest discover -s tests -v
clean:
	rm -rf src/__pycache__ tests/__pycache__
