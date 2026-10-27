"""Копирует реальные отчёты в Git и прикладывает доказательства к eval run."""
import json
import re
import shutil
from pathlib import Path

from mlflow import MlflowClient
import mlflow
import yaml


def main():
    params = yaml.safe_load(Path('params.yaml').read_text())
    mlflow.set_tracking_uri(params['tracking']['uri'])
    client = MlflowClient()
    out = Path('reports'); out.mkdir(exist_ok=True)
    for p in Path('metrics').glob('*.json'):
        shutil.copy2(p, out / p.name)
    sweep = json.loads(Path('metrics/sweep.json').read_text())
    candidate = json.loads(Path('metrics/candidate-model.json').read_text())
    evaluation = json.loads(Path('metrics/candidate-eval.json').read_text())
    registry = client.get_registered_model(candidate['name'])
    ids = {'sweep': sweep['parent_id'], 'evaluation': evaluation['evaluation_run_id'],
           **{r['name']: r['run_id'] for r in sweep['rows']}}
    proof = json.loads(Path('metrics/storage-check.json').read_text())
    ids['storage'] = proof['run_id']
    links = {k: f"{params['tracking']['uri']}/#/experiments/{client.get_run(v).info.experiment_id}/runs/{v}" for k,v in ids.items()}
    result = dict(tracking_uri=params['tracking']['uri'], run_ids=ids, links=links,
                  aliases=dict(registry.aliases), candidate=candidate,
                  note='MLflow доступен на ноутбуке; JSON-отчёты доступны в репозитории без сервера.')
    (out/'run-links.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for name in ('quality-gate.json','negative-gate.json','storage-check.json','run-links.json'):
        client.log_artifact(evaluation['evaluation_run_id'],str(out/name),'evidence')
    if Path('.check-all.log').exists():
        text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',Path('.check-all.log').read_text())
        (out/'check-all.txt').write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n')
    print('Отчёты:',out,'; aliases:',dict(registry.aliases))


if __name__ == '__main__':
    main()
