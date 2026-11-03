"""Настоящие subprocess/HTTP; HW8910_PROJECT позволяет проверить исходный commit."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

PROJECT = Path(os.environ.get('HW8910_PROJECT', Path(__file__).resolve().parents[1])).resolve()
DRIVER = os.environ.get('HW8910_HTTP_DRIVER', 'stdlib')


def record(event):
    name = os.environ.get('HW8910_TRANSCRIPT')
    if name:
        with open(name, 'a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')


class ProcessIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'workspace'
        self.command('init', expected=0)

    def command(self, *args, expected=None):
        p = subprocess.run([sys.executable, str(PROJECT/'lab.py'), '--workspace', str(self.root), *args],
                           cwd=self.tmp.name, capture_output=True, text=True, timeout=20)
        record(dict(command=['lab.py',*args],exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
        if expected is not None:self.assertEqual(p.returncode,expected,p.stdout+p.stderr)
        return p

    def test_cache_recovery_and_changed_input(self):
        self.assertNotEqual(self.command('run','--fail-after-prepare').returncode,0)
        self.assertIn('CACHE_HIT',self.command('run',expected=0).stdout)
        (self.root/'input.txt').write_text('Новые байты по прежнему пути\n')
        self.assertIn('PREPARED',self.command('run',expected=0).stdout)
        config=json.loads((self.root/'config.json').read_text())
        for key,value in [('tokenizer_revision','r2'),('template_sha256','f'*64),('max_length',256),('code_revision','v2')]:
            config[key]=value;(self.root/'config.json').write_text(json.dumps(config))
            self.assertIn('PREPARED',self.command('run',expected=0).stdout)

    def test_refusal_incomplete_and_foreign_report_keep_pointer(self):
        before=(self.root/'accepted.json').read_bytes()
        for report in ('report-reject.json','report-incomplete.json','report-A.json'):
            with self.subTest(report=report):
                result=self.command('run','--report',report)
                self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertEqual((self.root/'accepted.json').read_bytes(),before)

    def test_repeat_publication_across_processes(self):
        self.command('run',expected=0); before=(self.root/'accepted.json').read_bytes()
        self.command('run',expected=0)
        events=(self.root/'publications.jsonl').read_text().splitlines()
        self.assertEqual(len(events),1,'Один выпуск B не должен создавать два события')
        self.assertEqual((self.root/'accepted.json').read_bytes(),before)

    def test_pointer_journal_failure_recovery(self):
        # Одна и та же операция append доступна и в исходном, и в исправленном lab.
        code='''import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import lab
original=Path.open
def broken(self,mode='r',*a,**kw):
 if self.name=='publications.jsonl' and mode=='a':raise OSError('simulated journal failure')
 return original(self,mode,*a,**kw)
with patch.object(Path,'open',broken):lab.run(Path(sys.argv[2]),'report-B.json')
'''
        p=subprocess.run([sys.executable,'-c',code,str(PROJECT),str(self.root)],capture_output=True,text=True,timeout=20)
        record(dict(experiment='journal failure',exit=p.returncode,stderr=p.stderr))
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(json.loads((self.root/'accepted.json').read_text())['model_id'],'fixture-B')
        self.command('run',expected=0);self.command('run',expected=0)
        self.assertEqual(len((self.root/'publications.jsonl').read_text().splitlines()),1)

    def test_atomic_write_failure_preserves_pointer(self):
        spec=importlib.util.spec_from_file_location('tested_release',PROJECT/'release.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        manifest=json.loads((self.root/'manifest-B.json').read_text());report=json.loads((self.root/'report-B.json').read_text())
        target=self.root/'accepted.json';before=target.read_bytes()
        with patch('os.replace',side_effect=OSError('simulated disk error')):
            with self.assertRaises(OSError):module.publish(manifest,report,target)
        self.assertEqual(target.read_bytes(),before)
        self.assertEqual(list(self.root.glob('*.tmp')),[])

    @contextlib.contextmanager
    def server(self,warmup=0):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        path=PROJECT/('api.py' if DRIVER=='fastapi' else 'service.py')
        log=tempfile.TemporaryFile(mode='w+')
        command=[sys.executable,str(path),'--workspace',str(self.root),'--port',str(port),'--warmup-seconds',str(warmup)]
        if DRIVER=='fastapi':command+=['--backend','stub','--debug']
        p=subprocess.Popen(command,cwd=self.tmp.name,stdout=log,stderr=subprocess.STDOUT)
        self.url=f'http://127.0.0.1:{port}'
        try:
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                if p.poll() is not None:
                    log.seek(0);self.fail('Сервер завершился: '+log.read())
                try:
                    if self.http('/health')[0]==200:break
                except OSError:pass
                time.sleep(.04)
            else: self.fail('HTTP health не появился')
            yield
        finally:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()

    def http(self,path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':'application/json'})
        try:r=urllib.request.urlopen(req,timeout=5)
        except urllib.error.HTTPError as e:r=e
        with r: result=(r.status,json.loads(r.read()))
        record(dict(http=path,payload=payload,status=result[0],response=result[1]))
        return result

    def wait_ready(self):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if self.http('/ready')[0]==200:return
            time.sleep(.05)
        self.fail('Сервис не прогрелся')

    def test_http_readiness_before_generation(self):
        with self.server(warmup=1.5):
            self.assertEqual(self.http('/health')[0],200)
            self.assertEqual(self.http('/ready')[0],503,'ready не может опережать прогрев')
            calls=self.http('/debug')[1]['calls']
            self.assertEqual(self.http('/generate',{'prompt':'тест','max_new_tokens':32})[0],503)
            self.assertEqual(self.http('/debug')[1]['calls'],calls)
            self.wait_ready();self.assertEqual(self.http('/generate',{'prompt':'тест','max_new_tokens':32})[0],200)

    def test_http_loaded_version_restart_and_rollback(self):
        with self.server():
            self.wait_ready()
            self.command('run',expected=0)
            status,answer=self.http('/generate',{'prompt':'тест','max_new_tokens':32})
            self.assertEqual(status,200);self.assertEqual(answer['model_id'],'fixture-A')
            self.assertIn('fixture-A',answer['answer'])
        with self.server():
            self.wait_ready();self.assertEqual(self.http('/generate',{'prompt':'тест','max_new_tokens':32})[1]['model_id'],'fixture-B')
        self.command('run','--manifest','manifest-A.json','--report','report-A.json','--rollback',expected=0)
        with self.server():
            self.wait_ready();answer=self.http('/generate',{'prompt':'тест','max_new_tokens':32})[1]
            self.assertEqual(answer['model_id'],'fixture-A');self.assertIn('fixture-A',answer['answer'])

    def test_http_invalid_requests_do_not_call_backend(self):
        with self.server():
            self.wait_ready();before=self.http('/debug')[1]['calls']
            for payload in ({'prompt':''},{'prompt':'тест','max_new_tokens':0},{'prompt':'тест','max_new_tokens':100000},
                            {'prompt':'тест','max_new_tokens':'32'},{'prompt':'тест','max_new_tokens':True}):
                with self.subTest(payload=payload):
                    status,_=self.http('/generate',payload)
                    self.assertTrue(400<=status<500)
                    self.assertEqual(self.http('/debug')[1]['calls'],before)
            self.assertEqual(self.http('/generate',{'prompt':'тест','max_new_tokens':32})[0],200)
            self.assertEqual(self.http('/debug')[1]['calls'],before+1)

    def test_http_tampered_artifact_never_ready(self):
        (self.root/'model-A.txt').write_text('подменённый артефакт')
        with self.server():
            self.assertEqual(self.http('/ready')[0],503)
            self.assertEqual(self.http('/generate',{'prompt':'тест','max_new_tokens':32})[0],503)


if __name__=='__main__':unittest.main(verbosity=2)
