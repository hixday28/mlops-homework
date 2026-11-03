"""Отдельные процессы Prefect: сбой, восстановление, новый вход и WAL."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lab
from release import file_digest

ROOT=Path(__file__).resolve().parents[1]


def run(label,workspace,*flags,success=False):
    before=file_digest(workspace/'accepted.json')
    log=ROOT/'evidence'/f'{label}.log'
    with log.open('w') as f:
        p=subprocess.run([sys.executable,str(ROOT/'flow.py'),'--workspace',str(workspace),*flags],cwd=ROOT,
                         env=dict(os.environ,HF_HUB_OFFLINE='1'),stdout=f,stderr=subprocess.STDOUT)
        f.write(f'\nExit code: {p.returncode}\n')
    after=file_digest(workspace/'accepted.json')
    assert (p.returncode==0)==success,(label,log)
    states=[json.loads(x) for x in (workspace/'task-states.jsonl').read_text().splitlines()]
    result=dict(label=label,exit_code=p.returncode,accepted_before=before,accepted_after=after,
                prepare=states[-1],log=str(log.relative_to(ROOT)))
    if '--fail-after-prepare' not in flags and (workspace/'release-report.json').exists():
        result['decision']=json.loads((workspace/'release-report.json').read_text())
    return result


def main():
    real=ROOT/'.real';original=(real/'input.jsonl').read_bytes();results=[]
    results.append(run('flow-fail-after-prepare',real,'--fail-after-prepare','--refresh-preparation'))
    results.append(run('flow-recovered-refusal',real))
    assert results[0]['prepare']['state']=='Completed' and results[1]['prepare']['state']=='Cached'
    extra=json.loads(original.splitlines()[0]);extra['id']='preparation-extra-only';extra['group_id']='preparation-extra-only'
    (real/'input.jsonl').write_bytes(original+json.dumps(extra,ensure_ascii=False).encode()+b'\n')
    try:results.append(run('flow-changed-input',real))
    finally:(real/'input.jsonl').write_bytes(original)
    assert results[-1]['prepare']['state']=='Completed' and results[-1]['prepare']['rows']==51
    results.append(run('flow-restored-input',real))
    assert results[-1]['prepare']['state']=='Cached'
    assert all(r['accepted_before']==r['accepted_after'] for r in results)
    import shutil
    shutil.copy2(real/'release-report.json',ROOT/'evidence/release-report.json')
    shutil.copy2(real/'baseline-anchor.json',ROOT/'evidence/baseline-anchor.json')
    import uuid
    batch=ROOT/'.flow-experiments'/uuid.uuid4().hex;batch.mkdir(parents=True)
    stub=batch/'success'
    lab.initialize(stub)
    results.append(run('flow-stub-success',stub,'--stub',success=True))
    results.append(run('flow-stub-repeat',stub,'--stub',success=True))
    assert len((stub/'publications.jsonl').read_text().splitlines())==1
    crash=batch/'crash';lab.initialize(crash)
    results.append(run('flow-stub-pointer-crash',crash,'--stub','--fail-after-pointer'))
    assert results[-1]['accepted_before']!=results[-1]['accepted_after']
    results.append(run('flow-stub-pointer-recovery',crash,'--stub',success=True))
    results.append(run('flow-stub-pointer-repeat',crash,'--stub',success=True))
    assert len((crash/'publications.jsonl').read_text().splitlines())==1
    (ROOT/'evidence/flow-experiments.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'evidence/stub-publications.jsonl').write_text((crash/'publications.jsonl').read_text())
    print('PASS: recovery, invalidation, rejection, successful STUB flow, publication WAL')


if __name__=='__main__':main()
