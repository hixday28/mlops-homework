"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(params, dict):
        raise ValueError("Конфигурация должна содержать model, generate и bench")
    model, gen, bench = (params[key] for key in ("model", "generate", "bench"))
    if not isinstance(model["name"], str) or not model["name"].strip():
        raise ValueError("model.name должен быть непустой строкой")
    if model["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("model.device: допустимы auto, cpu, cuda, mps")
    if model["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("model.dtype: допустимы float32, float16, bfloat16")
    for group, key, minimum in ((gen, "max_new_tokens", 1), (bench, "warmup_runs", 1),
                                (bench, "measure_runs", 1), (gen, "seed", 0)):
        if type(group[key]) is not int or group[key] < minimum:
            raise ValueError(f"{key} должен быть целым числом >= {minimum}")
    if not isinstance(gen["temperature"], (int, float)) or not 0 <= gen["temperature"] < float("inf"):
        raise ValueError("temperature должна быть конечным числом >= 0")
    if type(gen["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking должен быть true или false")
    if not isinstance(bench["prompt"], str) or not bench["prompt"].strip():
        raise ValueError("bench.prompt должен быть непустым текстом")
    return params
