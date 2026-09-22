"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    threshold = params['clean']['near_dup']['threshold']
    if not 0 < threshold <= 1:
        raise ValueError('clean.near_dup.threshold должен быть в (0, 1]')
    if params['contamination']['threshold'] != threshold:
        raise ValueError('Пороги clean.near_dup и contamination должны совпадать')
    ratios = params['split']['ratios']
    if set(ratios) != {'train','val','test'} or any(v <= 0 for v in ratios.values()) or abs(sum(ratios.values())-1)>1e-9:
        raise ValueError('Нужны положительные доли train/val/test с суммой 1')
    return params


def source_files(params: dict) -> list[Path]:
    """Файлы-источники для текущей версии датасета.

    Версия живёт в params, а не в аргументах командной строки: иначе
    dvc.lock не запомнит, из чего собран артефакт.
    """
    version = params["collect"]["version"]
    sources = params["collect"]["sources"]
    if version not in sources:
        raise SystemExit(
            f"collect.version = {version!r}, но в collect.sources "
            f"есть только {sorted(sources)}"
        )
    files = [Path(p) for p in sources[version]]
    missing = [f for f in files if not f.exists()]
    if missing:
        # Первое, обо что спотыкается каждый: пакет приходит настроенным на
        # курсовой датасет, которого у студента нет. Сообщение должно говорить,
        # что делать, а не печатать FileNotFoundError с чужим абсолютным путём.
        raise SystemExit(
            "стадия collect не нашла источник:\n  "
            + "\n  ".join(str(f) for f in missing)
            + "\n\nТак и должно быть, если вы ещё не подключили СВОЙ датасет.\n"
              "Выполните make fetch либо uv run dvc pull при доступном remote."
        )
    return files
