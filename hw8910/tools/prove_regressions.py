"""Тот же набор девяти интеграционных тестов против исходного commit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='e4904d7'


def main():
    with tempfile.TemporaryDirectory(prefix='hw8910-original-') as tmp:
        project=Path(tmp)
        for name in ('lab.py','service.py','release.py'):
            data=subprocess.check_output(['git','show',f'{ORIGINAL}:hw8910/{name}'],cwd=ROOT)
            (project/name).write_bytes(data)
        env=dict(os.environ,HW8910_PROJECT=str(project),HW8910_HTTP_DRIVER='stdlib',
                 HW8910_TRANSCRIPT=str(ROOT/'evidence/before-http-process.jsonl'))
        (ROOT/'evidence/before-http-process.jsonl').write_text('')
        cmd=[sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_integration.py','-v']
        with (ROOT/'evidence/before-integration.log').open('w') as log:
            result=subprocess.run(cmd,cwd=tmp,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.write(f'\nOriginal commit: {ORIGINAL}\nExit code: {result.returncode}\n')
        if result.returncode==0:raise SystemExit('ОШИБКА: исходный код неожиданно прошёл новые тесты')
    print('Original failed as expected. See evidence/before-integration.log')


if __name__=='__main__':main()
