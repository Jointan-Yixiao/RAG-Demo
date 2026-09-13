"""Double-click launcher for the local workbench (no administrator rights)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import webbrowser

ROOT=Path(__file__).resolve().parents[1]


def healthy(url):
    try:
        with urlopen(url+'/api/health',timeout=1) as response:
            return json.loads(response.read()).get('service')=='rag-workbench'
    except Exception:
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--no-open',action='store_true')
    args=parser.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    if args.stop:
        if healthy(url):
            request=Request(url+'/api/shutdown',data=b'{}',headers={'Content-Type':'application/json'},method='POST')
            with urlopen(request,timeout=5) as response:response.read()
            print('RAG Workbench is stopping.')
        else:
            print('RAG Workbench is not running on this port.')
        return 0
    if not healthy(url):
        directory=ROOT/'data/ui'
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'server.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'scripts/_rag_web.py'),'--port',str(args.port)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        for _ in range(60):
            if healthy(url):break
            if child.poll() is not None:
                print('Unable to start. Check data/ui/server.log or try another port.')
                return 1
            time.sleep(.25)
        else:
            print('Startup timed out. Check data/ui/server.log.')
            return 1
    print('RAG Workbench: '+url)
    if not args.no_open:webbrowser.open(url)
    return 0


if __name__=='__main__':sys.exit(main())
