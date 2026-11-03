"""Совместимый с исходником HTTP-адаптер для быстрых опытов STUB ONLY."""
import argparse
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from serving import ServiceState,StubBackend,NotReady


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,status,payload):
            data=json.dumps(payload,ensure_ascii=False).encode();self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':self.send_json(200,{'alive':True,'backend':'STUB ONLY'})
            elif self.path=='/ready':self.send_json(200 if state.ready else 503,state.readiness())
            elif self.path=='/debug':self.send_json(200,state.debug())
            else:self.send_json(404,{'error':'not found'})
        def do_POST(self):
            if self.path!='/generate':self.send_json(404,{'error':'not found'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=65536:self.send_json(413,{'error':'body too large/empty'});return
                request=json.loads(self.rfile.read(size))
                self.send_json(200,state.generate(request))
            except NotReady as exc:self.send_json(503,{'error':str(exc)})
            except (ValueError,TypeError,KeyError) as exc:self.send_json(422,{'error':str(exc)})
    return Handler


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('.lab'))
    p.add_argument('--port',type=int,default=8091);p.add_argument('--warmup-seconds',type=float,default=5)
    args=p.parse_args();state=ServiceState(args.workspace,args.warmup_seconds)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(state));print(f'STUB ONLY: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
