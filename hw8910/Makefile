UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
RUN := $(UV) run --frozen --no-sync
export PREFECT_HOME := $(CURDIR)/.prefect
export PREFECT_API_URL := http://127.0.0.1:4208/api
export PREFECT_SERVER_ANALYTICS_ENABLED := false

.PHONY: install install-model check check-stdlib check-fastapi check-broken prefect-up materialize flow api-up api-restart api-stop benchmark
install:
	$(UV) sync --frozen
install-model:
	$(UV) sync --frozen --extra model
check: check-stdlib check-fastapi
check-stdlib:
	$(RUN) python -m compileall -q release.py publication.py lab.py service.py serving.py api.py flow.py prepare.py gate_bridge.py model_backend.py tools tests
	$(RUN) python -m unittest discover -s tests -v
check-fastapi:
	HW8910_HTTP_DRIVER=fastapi $(RUN) python -m unittest discover -s tests -v
check-broken:
	$(RUN) python tools/check_broken.py
prefect-up:
	$(RUN) python tools/prefect_server.py
materialize:
	$(RUN) python tools/materialize.py
flow:
	HF_HUB_OFFLINE=1 $(RUN) python flow.py --workspace .real
api-up:
	HF_HUB_OFFLINE=1 $(RUN) python tools/api_server.py start
api-restart:
	HF_HUB_OFFLINE=1 $(RUN) python tools/api_server.py restart
api-stop:
	$(RUN) python tools/api_server.py stop
benchmark:
	HF_HUB_OFFLINE=1 $(RUN) python tools/benchmark.py
