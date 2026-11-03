"""Управление только локальным сервером этого ДЗ на порту 4208."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
BASE_URL='http://127.0.0.1:4208'
API_URL=BASE_URL+'/api'


def get_json(path):
    with urllib.request.urlopen(BASE_URL+path,timeout=2) as response:
        return json.load(response)


def stop_owned(home):
    pidfile=home/'server.pid'
    if not pidfile.exists():
        raise SystemExit('UI API URL неверен; PID сервера этого ДЗ не найден')
    pid=int(pidfile.read_text())
    command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
    if str(ROOT/'.venv/bin/prefect') not in command or '--port 4208' not in command:
        raise SystemExit('Порт 4208 занят чужим процессом; автоматическая остановка отменена')
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        state=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip()
        if not state or state.startswith('Z'):
            pidfile.unlink(missing_ok=True);return
        time.sleep(.1)
    raise SystemExit('Prefect не завершился; автоматический SIGKILL не применяется')


def main():
    home=ROOT/'.prefect';home.mkdir(exist_ok=True)
    healthy=False
    try:
        healthy=get_json('/api/health') is True
    except OSError:pass
    if healthy:
        if get_json('/ui-settings').get('api_url')==API_URL:
            print('Prefect API and UI already healthy: '+BASE_URL);return
        print('Исправление UI API URL: перезапуск сервера этого ДЗ без удаления базы')
        stop_owned(home)
    env=dict(os.environ,PREFECT_HOME=str(home),PREFECT_API_URL=API_URL,
             PREFECT_SERVER_UI_API_URL=API_URL,PREFECT_UI_API_URL=API_URL,
             PREFECT_SERVER_ANALYTICS_ENABLED='false')
    log=(home/'server.log').open('a')
    p=subprocess.Popen([str(ROOT/'.venv/bin/prefect'),'server','start','--host','127.0.0.1','--port','4208'],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (home/'server.pid').write_text(str(p.pid));log.close()
    for _ in range(100):
        if p.poll() is not None:raise SystemExit('Prefect завершился, см. .prefect/server.log')
        try:
            if get_json('/api/health') is True and get_json('/ui-settings').get('api_url')==API_URL:
                print('Prefect API and UI ready: '+BASE_URL);return
        except OSError:pass
        time.sleep(.4)
    raise SystemExit('Prefect startup timeout')


if __name__=='__main__':main()
