"""Замер производительности машины на выбранной модели.

Три числа меряются РАЗДЕЛЬНО — смешивать их бессмысленно:
  * время загрузки модели  — разовая стоимость старта;
  * tokens/sec             — скорость генерации, только после прогрева;
  * пиковая RSS            — максимум за процесс, а не снимок в конце.
"""

import json
import platform
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version

from src.config import ROOT, load_params
from src.model import generate_tokens, load_model, prepare_inputs, synchronize


def peak_rss_mb() -> float:
    """Пиковая резидентная память процесса.

    ru_maxrss на macOS в байтах, на Linux в килобайтах.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    params = load_params()
    prompt = params["bench"]["prompt"]

    t0 = time.perf_counter()
    tokenizer, model = load_model(params)
    synchronize(model)
    load_time = time.perf_counter() - t0

    inputs = prepare_inputs(tokenizer, model, params, prompt)
    for _ in range(params["bench"]["warmup_runs"]):
        generate_tokens(model, inputs, params)
    synchronize(model)

    speeds = []
    runs = []
    for _ in range(params["bench"]["measure_runs"]):
        synchronize(model)
        t0 = time.perf_counter()
        tokens = generate_tokens(model, inputs, params)
        synchronize(model)
        elapsed = time.perf_counter() - t0
        n_tokens = len(tokens)
        speeds.append(n_tokens / elapsed)
        runs.append({"seconds": round(elapsed, 6), "new_tokens": n_tokens})

    # Медиана устойчивее среднего к одиночному выбросу.
    report = {
        "model": params["model"]["name"],
        "device": str(model.device),
        "dtype": params["model"]["dtype"],
        "load_time_sec": round(load_time, 2),
        "tokens_per_sec": round(statistics.median(speeds), 2),
        "tokens_per_sec_all": [round(s, 2) for s in speeds],
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "runs": runs,
        "params": params,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("torch", "transformers", "accelerate")},
        "timing_scope": "model.generate + device synchronization; excludes tokenization and decoding",
        "rss_unit": "MiB (2**20 bytes); process high-water mark, not total GPU/unified memory",
    }

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/bench.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
