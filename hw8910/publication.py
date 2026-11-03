"""Single-writer WAL: намерение → указатель → уникальное событие → завершение."""
import json
import os
from pathlib import Path
from release import atomic_json,canonical,digest,publish,validate_release,verify_files


def read(path):return json.loads(Path(path).read_text())


def append(path,value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def release_id(manifest,report):
    return digest(canonical({'manifest':manifest,'decision':report}))


def remember(root,manifest,report):
    atomic_json(root/'history'/f'{release_id(manifest,report)}.json',{'manifest':manifest,'report':report})


def _complete(root,intent,fail_after_pointer=False):
    manifest,report=intent['manifest'],intent['report']
    if release_id(manifest,report)!=intent['release_id']:raise ValueError('Повреждено намерение выпуска')
    if report.get('source_bindings'):
        from gate_bridge import verify_source_bindings
        verify_source_bindings(root,report)
    publish(manifest,report,root/'accepted.json')
    if fail_after_pointer:raise RuntimeError('Инъекция сбоя между указателем и журналом')
    journal=root/'publications.jsonl';events=[]
    if journal.exists():
        lines=journal.read_text().splitlines()
        for i,line in enumerate(lines):
            try:events.append(json.loads(line))
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise ValueError('Повреждена середина журнала')
                # Единственный незавершённый append защищён сохранённым intent.
                temporary=journal.with_suffix('.repair')
                temporary.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events));os.replace(temporary,journal)
    if not any(e['release_id']==intent['release_id'] for e in events):
        append(journal,{'release_id':intent['release_id'],'model_id':manifest['model_id'],
                        'runtime_sha256':manifest['runtime_sha256'],'evaluation_run_id':report['evaluation_run_id'],
                        'action':intent['action']})
    remember(root,manifest,report)
    (root/'publication-intent.json').unlink(missing_ok=True)


def publish_once(root,manifest,report,*,rollback=False,fail_after_pointer=False):
    root=Path(root);validate_release(manifest,report);verify_files(manifest,root)
    if rollback and not (root/'history'/f'{release_id(manifest,report)}.json').is_file():
        raise ValueError('Откат разрешён только к ранее принятому выпуску')
    pending=root/'publication-intent.json'
    if pending.exists():_complete(root,read(pending))
    intent={'manifest':manifest,'report':report,'release_id':release_id(manifest,report),'action':'rollback' if rollback else 'publish'}
    atomic_json(pending,intent)
    _complete(root,intent,fail_after_pointer)
    return intent['release_id']
