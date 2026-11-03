"""FastAPI с фоновой загрузкой в lifespan и одним вычислением одновременно."""
import argparse
from contextlib import asynccontextmanager
import json
from pathlib import Path
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from serving import ServiceState,StubBackend,NotReady


def create_app(root,backend_factory=StubBackend,warmup_seconds=0,debug=False,strict_limit=False):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model=ServiceState(root,warmup_seconds,backend_factory,strict_limit)
        yield
        app.state.model.ready=False
    app=FastAPI(title='MLOps: serving принятой версии',lifespan=lifespan)
    @app.get('/health')
    async def health():return {'alive':True,'backend':getattr(backend_factory,'label','transformers')}
    @app.get('/ready')
    async def ready():
        state=app.state.model
        return JSONResponse(state.readiness(),status_code=200 if state.ready else 503)
    if debug:
        @app.get('/debug')
        async def diagnostics():return app.state.model.debug()
    @app.post('/generate')
    async def generate(request:Request):
        try:
            raw=await request.body()
            if len(raw)>65536:return JSONResponse({'error':'body too large'},status_code=413)
            payload=json.loads(raw)
            result=await run_in_threadpool(app.state.model.generate,payload)
            return result
        except NotReady as exc:return JSONResponse({'error':str(exc)},status_code=503)
        except (ValueError,TypeError,KeyError) as exc:return JSONResponse({'error':str(exc)},status_code=422)
    return app


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('.real'))
    p.add_argument('--port',type=int,default=8091);p.add_argument('--warmup-seconds',type=float,default=0)
    p.add_argument('--backend',choices=['stub','transformers'],default='transformers');p.add_argument('--debug',action='store_true')
    a=p.parse_args();factory=StubBackend
    if a.backend=='transformers':
        from model_backend import TransformersBackend
        factory=TransformersBackend
    import uvicorn
    uvicorn.run(create_app(a.workspace,factory,a.warmup_seconds,a.debug,a.backend=='transformers'),host='127.0.0.1',port=a.port,log_level='warning')


if __name__=='__main__':main()
