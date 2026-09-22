"""Стадия split: разбиение на train/val/test."""

import json
import random
import hashlib
import time
from pathlib import Path

from src.config import load_params
from src.contamination import report, is_clean
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def row_split(count: int, ratios: dict[str, float], seed: int) -> list[str]:
    """Раздать строкам метки сплита в заданных долях."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    labels = [""] * count
    start = 0
    names = list(ratios)
    for i, name in enumerate(names):
        stop = count if i == len(names) - 1 else start + round(count * ratios[name])
        for pos in order[start:stop]:
            labels[pos] = name
        start = stop
    return labels


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    sizes: dict[str, int] = {}
    for ex in examples:
        key = normalize_group(ex.topic)
        sizes[key] = sizes.get(key, 0) + 1

    # Хеш целой нормализованной группы: добавление v2 не переносит старую
    # группу в другой сплит. Доли приблизительны, поскольку группы неделимы.
    if not examples:
        raise SystemExit('split: очищенный набор пуст')
    group_labels = {}
    for group in sorted(sizes):
        value = int(hashlib.sha256(f"{cfg['seed']}:{group}".encode()).hexdigest(),16)/2**256
        boundary = 0.0
        for name, ratio in cfg['ratios'].items():
            boundary += ratio
            if value < boundary:
                group_labels[group] = name
                break
    labels = [group_labels[normalize_group(ex.topic)] for ex in examples]
    buckets: dict[str, list[Example]] = {name: [] for name in cfg["ratios"]}
    for label, ex in zip(labels, examples):
        buckets[label].append(ex)

    for name, rows in buckets.items():
        if not rows:
            raise SystemExit(f'split: пустой {name}; проверьте число групп и seed')
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )
    if not is_clean(rep):
        raise SystemExit(f'КОНТАМИНАЦИЯ: {rep}')

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(sizes),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(sizes)}, {time.perf_counter() - started:.2f} с)"
    )


if __name__ == "__main__":
    main()
