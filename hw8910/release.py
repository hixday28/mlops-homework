"""Контракты метаданных, проверка файлов и атомарный JSON-указатель."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def cache_key(data_path: Path, config: dict) -> str:
    return digest(canonical({'data_sha256':file_digest(data_path),'config':config}))


def validate_release(manifest: dict, report: dict) -> None:
    try:
        if type(manifest['schema_version']) is not int or manifest['schema_version']!=1:
            raise ValueError('Неподдерживаемая схема manifest')
        for key in ('model_id','artifact_uri'):
            if not isinstance(manifest[key],str) or not manifest[key].strip():raise ValueError(f'Нет {key}')
        hashes=('model_sha256','protocol_sha256','golden_sha256','runtime_sha256')
        for key in hashes:
            if not isinstance(manifest[key],str) or not re.fullmatch('[0-9a-f]{64}',manifest[key]):raise ValueError(f'Неверный {key}')
        if not isinstance(manifest['runtime'],dict) or not manifest['runtime']:
            raise ValueError('Нет runtime')
        if digest(canonical(manifest['runtime']))!=manifest['runtime_sha256']:
            raise ValueError('Runtime изменён')
        if report['status']!='complete' or report['passed'] is not True:
            raise ValueError('Неполная оценка или отказ gate')
        if not isinstance(report['evaluation_run_id'],str) or not report['evaluation_run_id'].strip():raise ValueError('Нет evaluation_run_id')
        for key in ('model_id',*hashes):
            if report[key]!=manifest[key]:raise ValueError(f'Отчёт не соответствует {key}')
        for key in ('accuracy','format_rate'):
            v=report['metrics'][key]
            if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError(f'Неверная метрика {key}')
    except (KeyError,TypeError,OverflowError) as exc:
        raise ValueError(f'Неполный или неверный контракт: {exc}') from exc


def atomic_json(target,value):
    """Commit point — os.replace. До него целевой файл не меняется."""
    target=Path(target);payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    if target.exists() and target.read_bytes()==payload:return
    target.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.tmp',delete=False) as f:
            temporary=Path(f.name);f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(temporary,target)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def verify_files(manifest,root):
    files=manifest.get('file_hashes')
    if not isinstance(files,dict) or not files:raise ValueError('Нет списка файлов модели')
    for name,expected in files.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
            raise ValueError('Неверный путь/хеш артефакта')
        if not (Path(root)/p).is_file() or file_digest(Path(root)/p)!=expected:
            raise ValueError(f'Файл артефакта изменён: {name}')
    if digest(canonical(files))!=manifest['model_sha256']:raise ValueError('Список файлов не соответствует model_sha256')
    if digest(canonical(manifest['runtime']))!=manifest['runtime_sha256']:raise ValueError('Runtime изменён')


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest,report)
    if 'file_hashes' in manifest:verify_files(manifest,Path(target).parent)
    atomic_json(target,manifest)
