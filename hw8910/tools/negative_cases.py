"""Отказы metadata gate и ошибка os.replace на STUB; runtime/raw на реальном отчёте."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gate_bridge
import lab
from release import canonical,digest,file_digest,publish
ROOT=Path(__file__).resolve().parents[1]
results=[]
with tempfile.TemporaryDirectory(prefix='hw8910-negative-') as tmp:
    root=Path(tmp)/'stub';lab.initialize(root)
    m=lab.read(root/'manifest-B.json');r=lab.read(root/'report-B.json')
    before=file_digest(root/'accepted.json')
    for name,change in [('refused',{'passed':False}),('incomplete',{'status':'running'}),
                         ('truthy-string',{'passed':'true'}),('foreign',{'model_id':'wrong'}),
                         ('nan',{'metrics':{'accuracy':float('nan'),'format_rate':1.0}})]:
        try:publish(m,{**r,**change},root/'accepted.json');raise AssertionError(name+' was accepted')
        except ValueError as e:reason=str(e)
        after=file_digest(root/'accepted.json');assert before==after
        results.append(dict(case=name,synthetic=True,rejected=True,reason=reason,accepted_before=before,accepted_after=after))
    try:
        with patch('os.replace',side_effect=OSError('simulated disk error')):publish(m,r,root/'accepted.json')
        raise AssertionError('write error ignored')
    except OSError as e:reason=str(e)
    after=file_digest(root/'accepted.json');assert before==after
    results.append(dict(case='atomic-replace-error',synthetic=True,rejected=True,reason=reason,accepted_before=before,accepted_after=after))
    actual=ROOT/'.real';before=file_digest(actual/'accepted.json')
    manifest=lab.read(actual/'manifest-candidate.json');manifest=copy.deepcopy(manifest)
    manifest['runtime']['generation']['max_new_tokens']=64;manifest['runtime_sha256']=digest(canonical(manifest['runtime']))
    case_root=Path(tmp)/'runtime';case_root.mkdir()
    report=gate_bridge.decision(case_root,manifest)
    assert report['passed'] is False and report['status']=='failed' and 'runtime' in report['reason']
    results.append(dict(case='changed-runtime-with-recomputed-hash',synthetic=False,rejected=True,reason=report['reason'],
                        accepted_before=before,accepted_after=file_digest(actual/'accepted.json')))
    original=gate_bridge.OLD;alternate=Path(tmp)/'old';alternate.mkdir()
    for folder in ('evaluation','reports','data/eval'):
        shutil.copytree(original/folder,alternate/folder,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(original/'uv.lock',alternate/'uv.lock')
    p=alternate/'reports/candidate-outputs.json';outputs=lab.read(p);outputs[0]['raw']='TAMPERED';p.write_text(json.dumps(outputs))
    gate_bridge.OLD=alternate
    try:report=gate_bridge.decision(case_root,lab.read(actual/'manifest-candidate.json'))
    finally:gate_bridge.OLD=original
    assert report['passed'] is False and report['status']=='failed'
    results.append(dict(case='modified-saved-raw',synthetic=False,rejected=True,reason=report['reason'],
                        accepted_before=before,accepted_after=file_digest(actual/'accepted.json')))
(ROOT/'evidence/negative-cases.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print('PASS:',len(results),'negative cases, accepted bytes unchanged')
