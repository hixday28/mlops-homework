"""Один и тот же набор 8+8 HTTP-запросов; сохраняются также ошибки."""
from concurrent.futures import ThreadPoolExecutor
import json
import csv
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request
from api_server import ROOT,get,start,stop


def call(prompt):
    request=urllib.request.Request('http://127.0.0.1:8091/generate',
        data=json.dumps({'prompt':prompt,'max_new_tokens':32}).encode(),headers={'Content-Type':'application/json'})
    started=time.perf_counter()
    try:
        try:r=urllib.request.urlopen(request,timeout=60)
        except urllib.error.HTTPError as exc:r=exc
        with r:status=r.status;body=json.loads(r.read())
        error=None if status==200 else body.get('error',str(body))
    except Exception as exc:status=None;body={};error=f'{type(exc).__name__}: {exc}'
    return {'status':status,'error':error,'client_seconds':time.perf_counter()-started,
            'model_id':body.get('model_id'),'request_id':body.get('request_id'),
            'output_tokens':body.get('output_tokens'),'generation_seconds':body.get('generation_seconds'),
            'queue_seconds':body.get('queue_seconds'),'service_seconds':body.get('service_seconds'),'body':body}


def main():
    start()
    rows=[json.loads(l) for l in (ROOT.parent/'hw67/data/eval/golden.jsonl').read_text().splitlines()]
    prompts=rows[:8]
    ready=get('/ready')[1];smoke=call(prompts[0]['input'])
    if smoke['status']!=200:raise RuntimeError(smoke)
    ref=json.loads((ROOT.parent/'hw67/reports/baseline-outputs.json').read_text())
    expected=next(o['raw'] for o in ref if o['id']==prompts[0]['id'])
    proof=dict(health=get('/health'),ready=ready,response=smoke,baseline_raw=expected,
               same_answer_as_saved_baseline=smoke['body']['answer']==expected)
    (ROOT/'evidence/real-smoke.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    before=get('/debug')[1]['calls'];bad=[]
    for payload in ({'prompt':''},{'prompt':'тест','max_new_tokens':0},{'prompt':'тест','max_new_tokens':100000},
                    {'prompt':'тест','max_new_tokens':'32'},{'prompt':'тест','max_new_tokens':31},
                    {'prompt':'й '*1000,'max_new_tokens':32}):
        request=urllib.request.Request('http://127.0.0.1:8091/generate',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        try:r=urllib.request.urlopen(request,timeout=10)
        except urllib.error.HTTPError as exc:r=exc
        with r:bad.append({'status':r.status,'response':json.loads(r.read())})
    after=get('/debug')[1]['calls']
    assert before==after and all(400<=x['status']<500 for x in bad)
    (ROOT/'evidence/real-invalid.json').write_text(json.dumps(dict(calls_before=before,calls_after=after,requests=bad),ensure_ascii=False,indent=2)+'\n')
    attempts=[];summary=[]
    for concurrency in (1,2):
        start_time=time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:results=list(pool.map(lambda p:call(p['input']),prompts))
        elapsed=time.perf_counter()-start_time
        for row,result in zip(prompts,results):attempts.append(dict(input_id=row['id'],concurrency=concurrency,max_new_tokens=32,**result))
        ok=[x for x in results if x['status']==200];errors=[x for x in results if x['status']!=200]
        summary.append(dict(concurrency=concurrency,n=len(results),success=len(ok),errors=len(errors),batch_seconds=elapsed,
            successful_requests_per_second=len(ok)/elapsed,
            mean_success_seconds=statistics.mean(x['client_seconds'] for x in ok) if ok else None,
            max_success_seconds=max((x['client_seconds'] for x in ok),default=None),
            mean_error_seconds=statistics.mean(x['client_seconds'] for x in errors) if errors else None,
            max_error_seconds=max((x['client_seconds'] for x in errors),default=None),
            mean_queue_seconds=statistics.mean(x['queue_seconds'] for x in ok) if ok else None,
            total_output_tokens=sum(x['output_tokens'] for x in ok)))
    (ROOT/'evidence/requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in attempts))
    (ROOT/'evidence/benchmark.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'docs/benchmark.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summary)
    before_restart=ready
    stop();start();after_restart=get('/ready')[1];answer_after=call(prompts[0]['input'])
    assert before_restart['model_id']==after_restart['model_id'] and before_restart['runtime_sha256']==after_restart['runtime_sha256']
    (ROOT/'evidence/real-restart.json').write_text(json.dumps(dict(before=before_restart,after=after_restart,response=answer_after),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
