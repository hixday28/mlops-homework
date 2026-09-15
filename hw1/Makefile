.PHONY: install generate bench check unit clean

UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

install:
	$(UV) sync --locked

generate:
	@$(UV) run --locked python -m src.generate

bench:
	@$(UV) run --locked python -m src.bench

check:
	@PATH="$(dir $(UV)):$$PATH" bash tests/check.sh

unit:
	@$(UV) run --locked python -m unittest discover -s tests -v

clean:
	rm -f docs/bench.json out1.txt out2.txt params.yaml.bak
