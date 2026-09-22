"""Скачать закреплённый снимок ru StackOverflow и выделить предметный срез.

Запускается явно через make fetch. dvc repro работает с локальным снимком,
поэтому изменение внешнего сайта не меняет уже записанную версию данных.
"""
import hashlib
import io
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa
from src.config import load_params


def main():
    cfg = load_params()['source']
    archive = Path(cfg['download'])
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        tmp = archive.with_suffix('.part')
        urllib.request.urlretrieve(cfg['url'], tmp)
        os.replace(tmp, archive)
    wanted, excluded = set(cfg['tags']), set(cfg['excluded_tags'])
    out = Path(cfg['snapshot'])
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    tags_count = Counter()
    digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    if digest != cfg['archive_sha256']:
        raise SystemExit('SHA256 архива не совпадает с params.yaml; источник изменён или скачан не полностью')
    with io.TextIOWrapper(pa.input_stream(archive, compression='zstd'), encoding='utf8') as source, \
            out.with_suffix('.tmp').open('w', encoding='utf8') as dest:
        for line in source:
            row = json.loads(line)
            counts['scanned'] += 1
            tags = set(row['tags'])
            if not (tags & wanted) or tags & excluded:
                counts['dropped_scope'] += 1
                continue
            # Сохраняем исходные поля и все ответы, а не только выбранный ответ.
            dest.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts['selected'] += 1
            tags_count.update(tags & wanted)
    os.replace(out.with_suffix('.tmp'), out)
    manifest = dict(counts, source_url=cfg['url'], revision=cfg['revision'],
                    archive_sha256=digest, selected_tags=dict(tags_count),
                    snapshot_sha256=hashlib.file_digest(out.open('rb'), 'sha256').hexdigest())
    Path(cfg['metadata']).write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(f"source: {counts['scanned']} → {counts['selected']} вопросов; {out}")


if __name__ == '__main__':
    main()
