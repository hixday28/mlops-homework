"""Локальный процесс FastAPI; PID проверяется перед остановкой."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[1]


def get(path):
    import urllib.error
    try:r=urllib.request.urlopen('http://127.0.0.1:8091'+path,timeout=2)
    except urllib.error.HTTPError as exc:r=exc
    with r:return r.status,json.loads(r.read())


def stop():
    pidfile=ROOT/'.real/api.pid'
    if not pidfile.exists():return
    pid=int(pidfile.read_text())
    p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    if str(ROOT/'api.py') in p.stdout:
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            p=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
            if not p.stdout.strip() or p.stdout.strip().startswith('Z'):break
            time.sleep(.1)
        else:raise RuntimeError('API не завершился; автоматический SIGKILL не применяется')
    pidfile.unlink(missing_ok=True)


def start():
    try:
        if get('/health')[0]==200:print('API already alive: http://127.0.0.1:8091');return
    except OSError:pass
    root=ROOT/'.real';root.mkdir(exist_ok=True)
    env=dict(os.environ,HF_HUB_OFFLINE='1')
    log=(root/'api.log').open('a')
    p=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'api.py'),'--workspace',str(root),
                        '--port','8091','--backend','transformers','--debug'],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'api.pid').write_text(str(p.pid));log.close()
    observations=[]
    for _ in range(600):
        if p.poll() is not None:raise RuntimeError('API завершился, см. .real/api.log')
        try:
            health=get('/health');ready=get('/ready')
            observations.append({'health':health[0],'ready':ready[0],'response':ready[1]})
            if ready[0]==200:
                (ROOT/'evidence/ready-observations.json').write_text(json.dumps(observations,ensure_ascii=False,indent=2)+'\n')
                print('API ready:',ready[1]['model_id']);return
            if ready[1].get('error'):raise RuntimeError(ready[1]['error'])
        except (ConnectionError,urllib.error.URLError,TimeoutError):pass
        time.sleep(.1)
    raise RuntimeError('API load timeout')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','restart']);a=p.parse_args()
    if a.action in ('stop','restart'):stop()
    if a.action in ('start','restart'):start()


if __name__=='__main__':main()
