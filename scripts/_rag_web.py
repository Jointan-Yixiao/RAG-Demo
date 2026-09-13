"""Local RAG workbench HTTP adapter. No changes to the accepted retrieval pipeline.

Run with the project Python 3.12 environment. Only binds loopback; serves an
explicit asset allowlist, serializes GPU work, and never returns credentials.
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, unquote
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import _openai_compatible_generate as adapter

PUBLIC_SETTINGS = ('provider','base_url','model','language','detail')
PENDING = {'queued','running'}
PUBLIC_JOB = ('id','query','status','stage','created_at','answer_markdown','evidence','warning','error','usage')


class UIError(Exception):
    def __init__(self, message, status=400):
        self.status = status
        super().__init__(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def within(path, root):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise UIError('无法访问此文件。', 404)
    return resolved


def safe_url(url):
    if not isinstance(url, str):
        return ''
    p = urlsplit(url)
    return url if p.scheme in ('http','https') and p.hostname and not p.username and not p.password else ''


def resolve_bge_model_dir(root):
    root = Path(root).resolve()
    configured = os.environ.get('RAG_BGE_MODEL_DIR')
    if configured:
        path = Path(configured)
        return (path if path.is_absolute() else root/path).resolve()
    downloaded = root/'models/bge-reranker-v2-m3'
    if (downloaded/'model.safetensors').is_file():
        return downloaded
    return root/'data/metadata/retrieval-eval/experiment-29-bge-v2-m3/model/bge-reranker-v2-m3'


class WindowsVault:
    """Current-user Windows DPAPI; ciphertext alone cannot recover the API key."""
    class Blob(ctypes.Structure):
        _fields_ = [('length',ctypes.c_uint32),('data',ctypes.POINTER(ctypes.c_ubyte))]

    def crypt(self, data, decrypt=False):
        if os.name != 'nt':
            raise UIError('保存密钥需要 Windows；其他系统请通过环境变量配置。')
        raw = ctypes.create_string_buffer(data)
        src = self.Blob(len(data), ctypes.cast(raw,ctypes.POINTER(ctypes.c_ubyte)))
        dst = self.Blob()
        crypt = ctypes.WinDLL('crypt32', use_last_error=True)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        if decrypt:
            ok = crypt.CryptUnprotectData(ctypes.byref(src),None,None,None,None,1,ctypes.byref(dst))
        else:
            ok = crypt.CryptProtectData(ctypes.byref(src),'RAG Workbench',None,None,None,1,ctypes.byref(dst))
        if not ok:
            raise UIError('无法读取或保存本机密钥，请重新填写。')
        try:
            return ctypes.string_at(dst.data,dst.length)
        finally:
            kernel.LocalFree(ctypes.cast(dst.data,ctypes.c_void_p))

    def encrypt(self, value):
        return base64.b64encode(self.crypt(value.encode('utf-8'))).decode('ascii')

    def decrypt(self, value):
        return self.crypt(base64.b64decode(value), True).decode('utf-8')


class Settings:
    def __init__(self, root, directory, vault=None, key_resolver=None):
        self.root = Path(root)
        self.path = Path(directory)/'settings.json'
        self.base = read_json(self.root/'config/generation.deepseek-rag.json')
        self.defaults = dict(provider='openai-compatible',base_url=self.base['base_url'],
                             model=self.base['model'],language='auto',detail='balanced')
        self.vault = vault or WindowsVault()
        self.lock = threading.RLock()
        self.key_resolver = key_resolver or (lambda: adapter.resolve_api_key(adapter.validate_config(self.base))[1])

    def stored(self):
        return {**self.defaults, **(read_json(self.path,{}) or {})}

    def validate(self, body):
        if not isinstance(body,dict) or set(body)-set(PUBLIC_SETTINGS)-{'api_key'}:
            raise UIError('设置格式不正确。')
        result = self.stored()
        result.update({k:v for k,v in body.items() if k in PUBLIC_SETTINGS})
        if result['provider'] != 'openai-compatible':
            raise UIError('目前支持 OpenAI 兼容接口。')
        if result['language'] not in ('auto','zh','en') or result['detail'] not in ('concise','balanced','detailed'):
            raise UIError('请选择有效的回答语言和详细程度。')
        if not isinstance(result['model'],str) or not 1 <= len(result['model'].strip()) <= 200 or any(ord(c)<32 for c in result['model']):
            raise UIError('请填写有效的模型名称。')
        try:
            if not isinstance(result['base_url'],str) or len(result['base_url'])>1000:
                raise ValueError()
            adapter.chat_completions_url(result['base_url'])
        except (ValueError,TypeError):
            raise UIError('接口地址需使用 HTTPS，且不能带账号、查询参数或片段。') from None
        result['model'] = result['model'].strip()
        result['base_url'] = result['base_url'].strip().rstrip('/')
        key = body.get('api_key','')
        if not isinstance(key,str) or len(key)>4096 or any(c.isspace() for c in key):
            raise UIError('密钥格式不正确，请检查空格或换行。')
        return result, key

    def credential(self, settings):
        endpoint = adapter.chat_completions_url(settings['base_url'])
        if settings.get('encrypted_key'):
            if settings.get('key_endpoint') != endpoint:
                raise UIError('更换接口地址时，请填写该接口的密钥。')
            return self.vault.decrypt(settings['encrypted_key'])
        if endpoint == adapter.chat_completions_url(self.defaults['base_url']):
            try:
                return self.key_resolver()
            except adapter.MissingApiKeyError:
                pass
        raise UIError('尚未配置这个接口的密钥，请先在设置中填写。')

    def public(self):
        with self.lock:
            settings = self.stored()
            try:
                configured = bool(self.credential(settings))
            except UIError:
                configured = False
            return {**{k:settings[k] for k in PUBLIC_SETTINGS},'key_configured':configured}

    def snapshot(self, overrides=None):
        with self.lock:
            settings,key = self.validate(overrides or {})
            key = key or self.credential(settings)
            return {k:settings[k] for k in PUBLIC_SETTINGS}, key

    def save(self, body):
        with self.lock:
            result,key = self.validate(body)
            old = self.stored()
            if adapter.chat_completions_url(result['base_url']) != adapter.chat_completions_url(old['base_url']) and not key:
                raise UIError('更换接口地址时，请同时填写该接口的密钥。')
            if key:
                result['encrypted_key'] = self.vault.encrypt(key)
                result['key_endpoint'] = adapter.chat_completions_url(result['base_url'])
            write_json(self.path,result)
            return self.public()

    def generation_config(self, settings):
        raw = {k:self.base[k] for k in ('provider','base_url','model','max_output_tokens','max_input_utf8_bytes','timeout_seconds')}
        raw.update(provider='openai-compatible',base_url=settings['base_url'],model=settings['model'],api_key_env='RAG_UI_RUN_KEY')
        if urlsplit(settings['base_url']).hostname == 'api.deepseek.com' and settings['model'].startswith('deepseek-'):
            raw['provider'] = 'deepseek'
            raw['extra_body'] = self.base.get('extra_body',{})
        return raw

    def test(self, body):
        settings,key = self.snapshot(body)
        config = adapter.validate_config(self.generation_config(settings))
        payload = {'model':settings['model'],'messages':[{'role':'user','content':'Reply with OK.'}],
                   'max_tokens':32,'stream':False}
        if config['provider']=='deepseek':
            payload['thinking']={'type':'disabled'}
        try:
            status,raw = adapter.HttpsTransport(config['endpoint'],45)(config['endpoint'],
                json.dumps(payload).encode('utf-8'),{'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            if status==200 and isinstance(json.loads(raw).get('choices'),list) and json.loads(raw)['choices']:
                return {'ok':True,'message':'连接成功，模型已响应测试请求。'}
            messages = {401:'密钥无效或已失效。',403:'接口拒绝访问，请检查密钥权限。',404:'接口地址或模型名称不存在。',429:'服务商暂时限流或额度不足。'}
            return {'ok':False,'message':messages.get(status,'接口未返回有效的模型响应，请检查配置。')}
        except Exception:
            return {'ok':False,'message':'连接未完成，请检查网络、接口地址和服务商状态。'}


def display_answer(answer):
    answer = re.sub(r'!\[[^\]]*\]\([^\n]*?\)', '', answer or '')
    answer = re.sub(r'(?:\[S\d+\]\s*)+(?=的(?:原图|图表|正文|内容|描述))','对应证据',answer)
    answer = re.sub(r'\[S\d+\]', '', answer)
    return re.sub(r'[ \t]+([，。；：,.!?])',r'\1',answer).strip()


def result_from_run(job_id, run_dir, root):
    """Resolve only citations used in the delivered, reviewed answer for this job."""
    exported = read_json(Path(run_dir)/'07-answers/answers.json',{})
    row = next((r for r in exported.get('answers',[]) if r.get('case_id')=='q1'),None)
    if not row:
        return dict(status='failed',stage='处理未完成',answer_markdown='',evidence=[],error='本次处理未完成，请检查模型配置或重新提问。',warning=None,usage=None)
    status = row.get('status','failed')
    if status not in ('answered','answered_with_frontend_gaps'):
        no_answer = status.startswith('no_answer')
        return dict(status='no_answer' if no_answer else 'failed',stage='未找到可回答的资料' if no_answer else '处理未完成',
            answer_markdown=row.get('user_message_zh') or ('当前资料无法回答这个问题，请补充来源或换一个更明确的问法。' if no_answer else ''),
            evidence=[],error=None if no_answer else '本次回答未通过完整性检查，请稍后重试。',warning=None,usage=None)
    answer = row.get('answer_text') or ''
    if not answer.strip() or row.get('invalid_citation_ids'):
        raise UIError('本次回答引用校验未通过。',500)
    contexts = read_json(Path(run_dir)/'05-context/contexts.json',{})
    case = next((c for c in contexts.get('cases',[]) if c.get('case_id')=='q1'),{})
    payload = json.loads(next(m['content'] for m in case.get('messages',[]) if m['role']=='user'))
    texts = {e['citation_id']:e for e in payload.get('evidence',[])}
    mapping = {e['citation_id']:e for e in case.get('citation_map',[])}
    used = set(re.findall(r'\[(S\d+)\]',answer))
    citations = {c['citation_id']:c for c in row.get('citations',[])}
    if not used.issubset(citations.keys() & texts.keys() & mapping.keys()):
        raise UIError('本次回答的证据记录不完整。',500)
    evidence,assets = [],{}
    for cid,citation in citations.items():
        if cid not in used:
            continue
        text = texts[cid]['text']
        meta = mapping[cid]['metadata']
        if hashlib.sha256(text.encode()).hexdigest()!=mapping[cid]['text_sha256']:
            raise UIError('本次证据文本校验未通过。',500)
        if citation.get('chunk_ids')!=mapping[cid].get('chunk_ids'):
            raise UIError('本次证据关联校验未通过。',500)
        image = None
        if meta.get('image_path'):
            path = within(Path(root)/meta['image_path'],Path(root)/'原始资料')
            if path.is_file() and path.suffix.lower() in ('.png','.jpg','.jpeg'):
                assets[cid] = str(path)
                image = f'/api/questions/{job_id}/evidence/{cid}/image'
        related = '\n\n'.join(display_answer(p) for p in re.split(r'\n\s*\n',answer) if f'[{cid}]' in p)
        evidence.append(dict(citation_id=cid,type='figure' if meta.get('image_path') else 'text',
            document_id=meta.get('document_id',''),title=meta.get('document_title') or '未命名资料',
            source_url=safe_url(meta.get('source_url')),section=' / '.join(meta.get('section_path') or []),
            page=meta.get('page',meta.get('page_number')),label=meta.get('label'),text=text,
            image_url=image,related_answer=related))
    report = read_json(Path(run_dir)/'run-report.json',{})
    front = (report.get('frontend_calls') or {}).get('usage_totals',{})
    generation = (report.get('stages',{}).get('generation') or {})
    gen = generation.get('usage_totals_including_answer_review') or generation.get('usage_totals') or {}
    usage = {k:(front.get(k,0) or 0)+(gen.get(k,0) or 0) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    warning = '部分检索条件未能完整处理，核对证据时请留意问题覆盖范围。' if status=='answered_with_frontend_gaps' else None
    if not used:
        warning = '本次回答没有可定位的引用，请谨慎核对。'
    return dict(status='answered',stage='已完成',answer_markdown=display_answer(answer),evidence=evidence,
                warning=warning,error=None,usage=usage,_assets=assets)


class Workbench:
    def __init__(self, root=ROOT, data_dir=None, index_dir=None, start_worker=True, settings=None):
        self.root = Path(root).resolve()
        self.directory = Path(data_dir or self.root/'data/ui').resolve()
        self.jobs = self.directory/'jobs'
        self.jobs.mkdir(parents=True,exist_ok=True)
        self.settings = settings or Settings(self.root,self.directory)
        runtime = self.root/'data/runtime/index'
        self.index_dir = Path(index_dir or (runtime if (runtime/'records.jsonl').exists() else self.root/'data/index/description-v1')).resolve()
        self.bge_model_dir = resolve_bge_model_dir(self.root)
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.process = None
        self.testing = threading.Lock()
        self.catalog = {d['document_id']:d for d in read_json(self.root/'data/metadata/documents.json',{}).get('documents',[])
                        if d.get('include_in_index',True)}
        for path in self.jobs.glob('*/job.json'):
            job = read_json(path)
            if job.get('status') in PENDING:
                job.update(status='interrupted',stage='上次运行已中断',error='上次运行中断，未自动重新调用模型。请重新提问。')
                write_json(path,job)
        if start_worker:
            self.thread = threading.Thread(target=self.work,daemon=True)
            self.thread.start()

    def health(self):
        model = self.bge_model_dir
        ready = (self.index_dir/'manifest.json').is_file() and (model/'model.safetensors').is_file() and bool(shutil.which('node'))
        return dict(service='rag-workbench',ready=ready,busy=self.process is not None or not self.queue.empty(),
                    message='本地资料已就绪' if ready else '缺少本地索引、重排模型或 Node.js，请查看启动说明。')

    def job_dir(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}',job_id):
            raise UIError('未找到这次问答。',404)
        return self.jobs/job_id

    def get_job(self, job_id, public=True):
        with self.lock:
            directory = self.job_dir(job_id)
            job = read_json(directory/'job.json')
            if job is None:
                raise UIError('未找到这次问答。',404)
            if job['status']=='running':
                progress = read_json(directory/'progress.json',{})
                job['stage'] = progress.get('stage',job['stage'])
            return {k:job.get(k) for k in PUBLIC_JOB} if public else job

    def history(self):
        items = []
        for path in self.jobs.glob('*/job.json'):
            job = self.get_job(path.parent.name)
            items.append({k:job.get(k) for k in ('id','query','status','stage','created_at','warning')})
        return {'items':sorted(items,key=lambda r:r['created_at'],reverse=True)}

    def update(self, job_id, **changes):
        with self.lock:
            path = self.job_dir(job_id)/'job.json'
            job = read_json(path)
            job.update(changes)
            write_json(path,job)

    def submit(self, query, request_id=None):
        if not isinstance(query,str) or not query.strip() or len(query)>2000 or '\x00' in query:
            raise UIError('请输入 1 到 2000 字的问题。')
        if request_id is not None and not re.fullmatch(r'[0-9a-f]{32}',request_id):
            raise UIError('提交标识无效，请刷新页面后重试。')
        with self.lock:
            if request_id:
                for path in self.jobs.glob('*/job.json'):
                    previous = read_json(path)
                    if previous.get('request_id')==request_id:
                        if previous['query']!=query.strip():
                            raise UIError('同一次提交不能改成不同的问题。',409)
                        return self.get_job(previous['id'])
            if not self.health()['ready']:
                raise UIError(self.health()['message'],503)
            settings,key = self.settings.snapshot()
            if self.queue.full():
                raise UIError('等待处理的问题较多，请稍后再试。',429)
            job_id = uuid.uuid4().hex
            directory = self.job_dir(job_id)
            directory.mkdir()
            job = dict(id=job_id,query=query.strip(),status='queued',stage='等待处理',created_at=now(),
                       answer_markdown='',evidence=[],warning=None,error=None,usage=None,settings=settings,request_id=request_id)
            write_json(directory/'job.json',job)
            write_json(directory/'query.json',{'queries':[{'id':'q1','original_query':query.strip()}]})
            write_json(directory/'generation.json',self.settings.generation_config(settings))
            rules = (self.root/'config/deepseek-rag-rules.txt').read_text(encoding='utf-8')
            language = {'auto':'Use the same language as the original user question.','zh':'Answer in Simplified Chinese.','en':'Answer in English.'}[settings['language']]
            detail = {'concise':'Be concise while covering each requested item and keeping necessary limitations.',
                      'balanced':'Use a moderate level of explanation, enough to directly answer the question.',
                      'detailed':'Explain the directly relevant evidence in detail without adding unrelated topics.'}[settings['detail']]
            (directory/'rules.txt').write_text(rules+'\n\nUSER DISPLAY PREFERENCES\n'+language+'\n'+detail+
                '\nKeep all grounding rules and nearby [Snumber] citations. Citation markers are hidden on the main answer screen; '
                'use them as nearby annotations, never as grammatical subjects or possessives. '
                'Original figures are shown in a separate evidence view, accessible through the citation/evidence button. '
                'Do not claim you attached or sent an image directly to the user in the answer.\n',encoding='utf-8')
            self.queue.put_nowait((job_id,key))
            return self.get_job(job_id)

    def work(self):
        while not self.stop.is_set():
            try:
                job_id,key = self.queue.get(timeout=.5)
            except queue.Empty:
                continue
            directory = self.job_dir(job_id)
            try:
                self.update(job_id,status='running',stage='正在准备')
                env = dict(os.environ,RAG_UI_RUN_KEY=key,PYTHONUTF8='1',RAG_BGE_MODEL_DIR=str(self.bge_model_dir))
                command = [sys.executable,'-X','utf8',str(self.root/'scripts/_rag_web_worker.py'),
                           '--job-dir',str(directory),'--index-dir',str(self.index_dir)]
                with (directory/'worker.log').open('w',encoding='utf-8') as log:
                    self.process = subprocess.Popen(command,cwd=str(self.root),env=env,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    del env
                    key = None
                    deadline = time.monotonic()+1800
                    while self.process.poll() is None:
                        if self.stop.wait(.5) or time.monotonic()>deadline:
                            self.terminate_process()
                            raise UIError('本次处理已停止或超时，未自动重试。')
                result = result_from_run(job_id,directory/'run',self.root)
                self.update(job_id,**result)
            except Exception as error:
                message = str(error) if isinstance(error,UIError) else '本次处理未完成，请检查本地运行环境后重新提问。'
                self.update(job_id,status='interrupted' if self.stop.is_set() else 'failed',stage='处理未完成',error=message)
            finally:
                key = None
                self.process = None
                self.queue.task_done()

    def terminate_process(self):
        if self.process is not None and self.process.poll() is None:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.process.terminate()
            self.process.wait(timeout=20)

    def close(self):
        self.stop.set()
        if hasattr(self,'thread'):
            self.thread.join(timeout=25)

    def documents(self):
        return {'documents':[dict(id=d['document_id'],title=d['document_title'],source_url=safe_url(d.get('source_url'))) for d in self.catalog.values()]}

    def document(self, document_id):
        d = self.catalog.get(document_id)
        if not d:
            raise UIError('未找到这份资料。',404)
        path = within(self.root/d['source_path'],self.root/'原始资料')
        if not path.is_file():
            raise UIError('本地原文文件不存在。',404)
        return dict(id=document_id,title=d['document_title'],source_url=safe_url(d.get('source_url')),text=path.read_text(encoding='utf-8'))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, app):
        self.app = app
        super().__init__(address,Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self,*args):
        pass  # Query strings, request bodies and credentials never enter HTTP logs.

    def send(self, data, status=200, content_type='application/json; charset=utf-8'):
        payload = json.dumps(data,ensure_ascii=False).encode('utf-8') if isinstance(data,(dict,list)) else data
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(payload)))
        if self.close_connection:
            self.send_header('Connection','close')
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cross-Origin-Resource-Policy','same-origin')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def guard(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}',f'localhost:{port}'}
        if self.headers.get('Host','').lower() not in hosts:
            raise UIError('仅允许本机访问。',403)
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://'+h for h in hosts}:
            raise UIError('不允许跨站访问本机设置。',403)
        if self.headers.get('Sec-Fetch-Site')=='cross-site':
            raise UIError('不允许跨站访问。',403)

    def body(self):
        if self.headers.get('Transfer-Encoding'):
            raise UIError('请求需要 JSON 格式。',415)
        try:
            size = int(self.headers.get('Content-Length','0'))
        except ValueError:
            raise UIError('请求长度不正确。') from None
        if not 0 < size <= 16384:
            raise UIError('请求过大或为空。',413)
        self.connection.settimeout(15)
        try:
            raw = self.rfile.read(size)
            if len(raw)!=size:
                raise UIError('请求内容不完整。')
            # Consume a bounded body before rejecting its media type, otherwise
            # closing an unread Windows socket can discard the error response.
            if self.headers.get_content_type()!='application/json':
                raise UIError('请求需要 JSON 格式。',415)
            body = json.loads(raw)
        except (ValueError,UnicodeError,TimeoutError):
            raise UIError('请求内容无效。') from None
        if not isinstance(body,dict):
            raise UIError('请求内容需要是对象。')
        return body

    def handle_request(self):
        try:
            self.guard()
            path = unquote(urlsplit(self.path).path)
            app = self.server.app
            if self.command=='GET':
                if path=='/api/health':return self.send(app.health())
                if path=='/api/history':return self.send(app.history())
                if path=='/api/settings':return self.send(app.settings.public())
                if path=='/api/documents':return self.send(app.documents())
                if path.startswith('/api/documents/'):
                    return self.send(app.document(path.removeprefix('/api/documents/')))
                match = re.fullmatch(r'/api/questions/([0-9a-f]{32})(?:/evidence/(S\d+)/image)?',path)
                if match:
                    job_id,cid = match.groups()
                    if not cid:return self.send(app.get_job(job_id))
                    asset = app.get_job(job_id,False).get('_assets',{}).get(cid)
                    if not asset:raise UIError('这条证据没有原图。',404)
                    file = within(asset,app.root/'原始资料')
                    if not file.is_file():raise UIError('原图文件不存在。',404)
                    return self.send(file.read_bytes(),content_type=mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
                static = {'/':'index.html','/index.html':'index.html','/app.js':'app.js','/styles.css':'styles.css'}
                if path in static:
                    file = app.root/'web'/static[path]
                    if not file.is_file():raise UIError('界面文件尚未安装。',503)
                    kind = {'html':'text/html','js':'application/javascript','css':'text/css'}[file.suffix[1:]]
                    return self.send(file.read_bytes(),content_type=kind+'; charset=utf-8')
                raise UIError('未找到这个页面。',404)
            if self.command in ('POST','PUT'):
                body = self.body()
                if self.command=='POST' and path=='/api/questions':
                    if set(body)!={'query'}:raise UIError('问题请求格式不正确。')
                    return self.send(app.submit(body['query'],self.headers.get('Idempotency-Key')),202)
                if self.command=='PUT' and path=='/api/settings':
                    return self.send(app.settings.save(body))
                if self.command=='POST' and path=='/api/settings/test':
                    if not app.testing.acquire(blocking=False):raise UIError('连接测试进行中，请稍候。',429)
                    try:return self.send(app.settings.test(body))
                    finally:app.testing.release()
                if self.command=='POST' and path=='/api/shutdown':
                    if body:raise UIError('停止请求格式不正确。')
                    self.send({'ok':True,'message':'正在停止本机工作台。'})
                    threading.Thread(target=self.server.shutdown,daemon=True).start()
                    return
            raise UIError('不支持这个操作。',405)
        except UIError as error:
            self.close_connection=True
            self.send({'error':str(error)},error.status)
        except (BrokenPipeError,ConnectionResetError):
            pass
        except Exception:
            self.close_connection=True
            self.send({'error':'本机服务处理失败，请检查配置或重新启动。'},500)

    do_GET = handle_request
    do_POST = handle_request
    do_PUT = handle_request
    do_OPTIONS = handle_request


def main():
    parser = argparse.ArgumentParser(description='本地 RAG 资料问答工作台')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/ui')
    parser.add_argument('--index-dir',type=Path)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    lockfile = (args.data_dir/'server.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        if lockfile.tell()==0:lockfile.write(b'1');lockfile.flush()
        lockfile.seek(0)
        try:msvcrt.locking(lockfile.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise SystemExit('这个资料目录已有工作台运行，请使用已打开的窗口。')
    app = Workbench(data_dir=args.data_dir,index_dir=args.index_dir)
    try:
        server = Server(('127.0.0.1',args.port),app)
        print(f'RAG Workbench: http://127.0.0.1:{server.server_port}',flush=True)
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        if 'server' in locals():server.server_close()
        lockfile.close()


if __name__=='__main__':
    main()
