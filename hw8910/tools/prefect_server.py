"""Управление только локальным сервером этого ДЗ на порту 4208."""
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    try:
        if urllib.request.urlopen('http://127.0.0.1:4208/api/health',timeout=2).status==200:
            print('Prefect already healthy: http://127.0.0.1:4208');return
    except OSError:pass
    home=ROOT/'.prefect';home.mkdir(exist_ok=True)
    env=dict(os.environ,PREFECT_HOME=str(home),PREFECT_SERVER_ANALYTICS_ENABLED='false')
    log=(home/'server.log').open('a')
    p=subprocess.Popen([str(ROOT/'.venv/bin/prefect'),'server','start','--host','127.0.0.1','--port','4208'],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (home/'server.pid').write_text(str(p.pid));log.close()
    for _ in range(100):
        if p.poll() is not None:raise SystemExit('Prefect завершился, см. .prefect/server.log')
        try:
            if urllib.request.urlopen('http://127.0.0.1:4208/api/health',timeout=1).status==200:
                print('Prefect ready: http://127.0.0.1:4208');return
        except OSError:pass
        time.sleep(.4)
    raise SystemExit('Prefect startup timeout')


if __name__=='__main__':main()
