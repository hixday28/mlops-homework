"""Читает готовый training run ДЗ 6+7; не обучает и не меняет Registry alias."""
import json
import argparse
import io
import time
import urllib.request
import urllib.error
import zipfile
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from importlib.metadata import version
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import torch
from release import atomic_json,canonical,digest,file_digest

PROJECT=Path(__file__).resolve().parents[1]
OLD=PROJECT.parent/'hw67'
VERSIONS=('torch','transformers','peft','tokenizers','numpy','accelerate','safetensors','huggingface-hub')


def build_manifest(root,artifact,model_id,uri,model_kind,protocol,eval_report):
    cfg=json.loads((OLD/'evaluation/params.json').read_text())
    tok=AutoTokenizer.from_pretrained(artifact,local_files_only=True)
    files={str(p.relative_to(root)):file_digest(p) for p in sorted(artifact.rglob('*')) if p.is_file()}
    token_files={Path(n).name:h for n,h in files.items() if Path(n).name in ('tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','special_tokens_map.json','added_tokens.json')}
    runtime=dict(base_revision=json.loads((OLD/'evaluation/baseline.json').read_text())['revision'],
                 tokenizer_revision=json.loads((OLD/'evaluation/baseline.json').read_text())['revision'],
                 tokenizer_sha256=digest(canonical(token_files)),tokenizer_files=token_files,
                 template_sha256=digest(tok.chat_template.encode()),
                 generation=dict(system=cfg['system'],max_new_tokens=cfg['max_new_tokens'],do_sample=False,seed=cfg['seed'],enable_thinking=cfg['enable_thinking']),
                 dtype='bfloat16',device='mps',library_versions={n:version(n) for n in VERSIONS},max_input_tokens=512)
    return dict(schema_version=1,model_id=model_id,artifact_uri=uri,model_kind=model_kind,
                local_artifact=str(artifact.relative_to(root)),file_hashes=files,model_sha256=digest(canonical(files)),
                protocol_sha256=protocol,golden_sha256=eval_report['golden_sha256'],
                runtime=runtime,runtime_sha256=digest(canonical(runtime)),
                evaluation_run_id=eval_report['evaluation_run_id'])


def download(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=30) as response:return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code<500 and exc.code!=429:raise
            if attempt==2:raise
        except (urllib.error.URLError,TimeoutError):
            if attempt==2:raise
        time.sleep(attempt+1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact-source',choices=['release','mlflow'],default='release');args=parser.parse_args()
    root=PROJECT/'.real';root.mkdir(exist_ok=True)
    artifact=root/'artifacts/base';artifact.mkdir(parents=True,exist_ok=True)
    spec=json.loads((OLD/'evaluation/baseline.json').read_text())
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:cached=Path(snapshot_download(spec['name'],revision=spec['revision'],local_files_only=True))
    except LocalEntryNotFoundError:cached=Path(snapshot_download(spec['name'],revision=spec['revision'],allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json']))
    if not (OLD/'data/eval/golden.jsonl').exists():
        metadata=json.loads((OLD/'reports/evaluation-data-asset.json').read_text())
        blob=download(metadata['url'])
        if digest(blob)!=metadata['sha256']:raise ValueError('Неверный SHA256 набора оценки')
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in ('golden.jsonl','behavioral.jsonl','training-index.jsonl','training-attribution.jsonl'):
                target=OLD/'data/eval'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read('data/eval/'+name))
    for name in ('config.json','generation_config.json','model.safetensors','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'):
        target=artifact/name
        if not target.exists():target.symlink_to((cached/name).resolve())
    a=json.loads((OLD/'reports/baseline-eval.json').read_text());b=json.loads((OLD/'reports/candidate-eval.json').read_text())
    baseline=build_manifest(root,artifact,'baseline:'+spec['name']+'@'+spec['revision'],'hf://'+spec['name']+'@'+spec['revision'],'base',a['protocol_sha256'],a)
    candidate_spec=json.loads((OLD/'reports/candidate-model.json').read_text())
    if args.artifact_source=='mlflow':
        import mlflow
        from mlflow import MlflowClient
        os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES','2')
        os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT','10')
        destination=root/'artifacts/candidate-bundle';destination.mkdir(exist_ok=True)
        mlflow.set_tracking_uri('http://127.0.0.1:5006');client=MlflowClient()
        version_info=client.get_model_version(candidate_spec['name'],candidate_spec['version'])
        if version_info.run_id!=candidate_spec['run_id']:raise ValueError('Registry version changed')
        modelroot=Path(client.download_artifacts(candidate_spec['run_id'],'model',str(destination)))
        meta=yaml.safe_load((modelroot/'MLmodel').read_text())
        candidate_dir=modelroot/meta['flavors']['python_function']['artifacts']['adapter']['path']
    else:
        reference=json.loads((PROJECT/'evidence/candidate-manifest.json').read_text())
        candidate_dir=root/reference['local_artifact'];candidate_dir.mkdir(parents=True,exist_ok=True)
        for name,expected in reference['file_hashes'].items():
            target=root/name
            if target.exists() and file_digest(target)==expected:continue
            url='https://github.com/hixday28/mlops-homework/releases/download/hw8910-artifact/'+Path(name).name
            blob=download(url)
            if digest(blob)!=expected:raise ValueError('Подменён файл артефакта: '+name)
            target.write_bytes(blob)
    candidate=build_manifest(root,candidate_dir,candidate_spec['run_id'],candidate_spec['uri'],'adapter',b['protocol_sha256'],b)
    for role,manifest in [('baseline',baseline),('candidate',candidate)]:
        reference_path=PROJECT/'evidence'/f'{role}-manifest.json'
        if reference_path.exists() and json.loads(reference_path.read_text())!=manifest:
            raise ValueError('Артефакт/runtime изменён: нужна новая оценка, старый отчёт не наследуется')
        atomic_json(root/f'manifest-{role}.json',manifest);atomic_json(PROJECT/'evidence'/f'{role}-manifest.json',manifest)
    # Существующий закреплённый baseline импортируется как начальная точка,
    # это НЕ результат нового quality_gate и НЕ выпуск кандидата с passed=True.
    if not (root/'accepted.json').exists():
        atomic_json(root/'accepted.json',baseline)
        atomic_json(root/'baseline-anchor.json',dict(kind='imported-baseline-anchor',
            source='../hw67/evaluation/baseline.json',source_sha256=file_digest(OLD/'evaluation/baseline.json'),
            evaluation_run_id=a['evaluation_run_id'],model_id=baseline['model_id'],
            note='Точка отсчёта ДЗ 6+7; нового PASS кандидату не выдаётся'))
    import shutil
    shutil.copy2(OLD/'data/eval/golden.jsonl',root/'input.jsonl')
    config=dict(tokenizer_revision=baseline['runtime']['base_revision'],template_sha256=baseline['runtime']['template_sha256'],
                max_length=512,code_revision=file_digest(PROJECT/'prepare.py'))
    atomic_json(root/'config.json',config)
    print('READ EXISTING RUN',candidate_spec['run_id'],'baseline',baseline['model_id'],'source',args.artifact_source)


if __name__=='__main__':main()
