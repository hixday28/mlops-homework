"""Общий контроллер для stdlib-стенда и FastAPI; backend внедряется отдельно."""
import copy
import json
from pathlib import Path
import threading
import time
import uuid
from release import verify_files


class NotReady(Exception):pass


class StubBackend:
    label='STUB ONLY'
    def __init__(self,manifest,root):
        self.loaded_id=manifest['model_id'];self.runtime_sha256=manifest['runtime_sha256'];self.calls=0
    def warmup(self):pass
    def validate_prompt(self,prompt):pass
    def generate(self,prompt,limit):
        self.calls+=1;time.sleep(.01)
        return {'answer':f'STUB ONLY: {self.loaded_id}','output_tokens':1}


class ServiceState:
    def __init__(self,root,warmup_seconds=0,backend_factory=StubBackend,strict_limit=False):
        self.root=Path(root);self.ready=False;self.error=None;self.backend=None
        self.warmup_finished=threading.Event();self._lock=threading.Lock();self.strict_limit=strict_limit
        try:
            self.manifest=copy.deepcopy(json.loads((self.root/'accepted.json').read_text()))
            verify_files(self.manifest,self.root)
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}';return
        def load():
            try:
                self.backend=backend_factory(self.manifest,self.root)
                self.backend.warmup();time.sleep(warmup_seconds)
                self.warmup_finished.set();self.ready=True
            except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'
        threading.Thread(target=load,daemon=True).start()

    def validate(self,request):
        if not isinstance(request,dict) or set(request)-{'prompt','max_new_tokens'}:raise ValueError('Неверные поля запроса')
        prompt=request.get('prompt')
        if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>2048:raise ValueError('prompt: 1–2048 символов')
        limit=request.get('max_new_tokens',32)
        maximum=self.manifest['runtime']['generation']['max_new_tokens']
        if type(limit) is not int or not 1<=limit<=maximum or self.strict_limit and limit!=maximum:
            raise ValueError(f'max_new_tokens: {maximum} (фиксированный)' if self.strict_limit else f'max_new_tokens: 1–{maximum}')
        return prompt,limit

    def generate(self,request):
        if not self.ready:raise NotReady('Модель ещё не загружена и не прогрета')
        started=time.perf_counter();prompt,limit=self.validate(request)
        with self._lock:
            acquired=time.perf_counter();self.backend.validate_prompt(prompt)
            generated=time.perf_counter();result=self.backend.generate(prompt,limit)
            finished=time.perf_counter()
        result.update(model_id=self.backend.loaded_id,runtime_sha256=self.backend.runtime_sha256,
                      request_id=str(uuid.uuid4()),queue_seconds=acquired-started,
                      generation_seconds=finished-generated,service_seconds=finished-started)
        return result

    def readiness(self):
        return {'ready':self.ready,'model_id':self.backend.loaded_id if self.ready else None,
                'runtime_sha256':self.backend.runtime_sha256 if self.ready else None,'error':self.error}

    def debug(self):
        return {'loaded_id':self.backend.loaded_id if self.backend else None,'calls':self.backend.calls if self.backend else 0,
                'warmup_finished':self.warmup_finished.is_set()}
