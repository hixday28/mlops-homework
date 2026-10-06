export PYTHONUNBUFFERED := 1
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

.PHONY: install train train-all train-freeze compare plot check clean distclean

install:
	$(UV) sync --locked

# Два прогона: адаптеры на всех слоях и с замороженными первыми 14.
train: train-all train-freeze plot

train-all:
	$(UV) run --locked python -m src.train --variant all_layers

train-freeze:
	$(UV) run --locked python -m src.train --variant freeze14

plot:
	$(UV) run --locked python -m src.plot

# Базовая модель против адаптера на пяти фиксированных промптах.
compare:
	$(UV) run --locked python -m src.compare --variant all_layers

check:
	@PATH="$(dir $(UV)):$$PATH" bash tests/check.sh

# clean — артефакты обучения (адаптеры, метрики, графики); всё пересоздаётся make train.
# distclean — ещё и .venv (~0,8 ГБ): когда ДЗ сдано и место нужнее.
clean:
	rm -rf models metrics/*.json docs/curves.png docs/compare.md .check_*.log
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +

distclean: clean
	rm -rf .venv
