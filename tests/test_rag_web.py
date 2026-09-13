"""HTTP, credential, queue and citation boundaries; all model calls are mocked."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import _rag_web as web


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        for p in ('config','data/metadata','原始资料','web'):
            (self.root/p).mkdir(parents=True)
        self.config={'provider':'deepseek','base_url':'https://api.deepseek.com','model':'deepseek-flash',
                     'api_key_env':'DEEPSEEK_API_KEY','max_output_tokens':16384,'max_input_utf8_bytes':131072,'timeout_seconds':120}
        web.write_json(self.root/'config/generation.deepseek-rag.json',self.config)
        (self.root/'config/deepseek-rag-rules.txt').write_text('Preserve grounding and citations.',encoding='utf-8')
        (self.root/'原始资料/paper.md').write_text('# Paper\n\nOriginal document.',encoding='utf-8')
        (self.root/'原始资料/figure.png').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
        (self.root/'.env').write_text('private-file-sentinel',encoding='utf-8')
        (self.root/'web/index.html').write_text('<p>Workbench</p>',encoding='utf-8')
        web.write_json(self.root/'data/metadata/documents.json',{'documents':[{'document_id':'paper',
            'document_title':'Paper','source_url':'https://example.com/paper','source_path':'原始资料/paper.md'}]})
        self.settings=web.Settings(self.root,self.root/'data/ui',key_resolver=lambda:'synthetic-test-key')
        self.app=web.Workbench(root=self.root,settings=self.settings,start_worker=False)

    def tearDown(self):
        self.app.close()
        self.tmp.cleanup()

    def run_fixture(self, answer='Answer supported by the figure [S2].', status='answered'):
        run=self.root/'run'
        meta={'document_id':'paper','document_title':'Paper','source_url':'https://example.com/paper',
              'source_path':'原始资料/paper.md','section_path':['Methods'],'page':2}
        entries=[dict(citation_id='S1',chunk_ids=['body'],text='Body evidence.',source=meta),
                 dict(citation_id='S2',chunk_ids=['figure'],text='Figure evidence.',source={**meta,'image_path':'原始资料/figure.png','label':'Figure 1'})]
        mapping=[dict(citation_id=e['citation_id'],chunk_ids=e['chunk_ids'],metadata=e['source'],
                      text_sha256=hashlib.sha256(e['text'].encode()).hexdigest()) for e in entries]
        web.write_json(run/'05-context/contexts.json',{'cases':[{'case_id':'q1','messages':[{'role':'user','content':json.dumps({'evidence':entries})}],
                                                              'citation_map':mapping}]})
        row={'case_id':'q1','status':status,'answer_text':answer,'invalid_citation_ids':[],
             'citations':[{'citation_id':e['citation_id'],'chunk_ids':e['chunk_ids']} for e in entries]}
        web.write_json(run/'07-answers/answers.json',{'answers':[row]})
        return run


class Credentials(Fixture):
    def test_public_never_returns_key(self):
        public=self.settings.public()
        self.assertTrue(public['key_configured'])
        self.assertNotIn('synthetic-test-key',json.dumps(public))
        self.assertEqual(self.settings.snapshot()[1],'synthetic-test-key')

    @unittest.skipUnless(os.name=='nt','Windows DPAPI')
    def test_save_encrypts_and_survives_new_settings_instance(self):
        public=self.settings.save({'api_key':'new-synthetic-key','detail':'detailed','language':'en'})
        self.assertTrue(public['key_configured'])
        self.assertNotIn('new-synthetic-key',self.settings.path.read_text())
        fresh=web.Settings(self.root,self.root/'data/ui')
        value,key=fresh.snapshot()
        self.assertEqual(key,'new-synthetic-key')
        self.assertEqual(value['language'],'en')
        self.assertEqual(value['detail'],'detailed')
        fresh.save({'model':'another-model'})
        self.assertEqual(fresh.snapshot()[1],'new-synthetic-key')

    def test_new_endpoint_cannot_borrow_existing_key(self):
        with self.assertRaises(web.UIError):self.settings.save({'base_url':'https://other.example/v1'})
        with self.assertRaises(web.UIError):self.settings.snapshot({'base_url':'https://other.example/v1'})
        self.assertFalse(self.settings.path.exists())

    def test_bad_url_or_unknown_preferences_never_persist(self):
        for body in ({'base_url':'http://bad.example'},{'base_url':'https://key@bad.example'},
                     {'base_url':'https://bad.example/?secret=x'},{'language':'bad'},
                     {'detail':'bad'},{'api_key':'has whitespace'},{'unexpected':'value'}):
            with self.subTest(body=body),self.assertRaises(web.UIError):self.settings.save(body)
        self.assertFalse(self.settings.path.exists())

    def test_other_provider_has_no_deepseek_specific_body(self):
        settings=dict(self.settings.defaults,base_url='https://example.com/v1',model='model')
        config=self.settings.generation_config(settings)
        self.assertNotIn('extra_body',config)
        self.assertEqual(config['api_key_env'],'RAG_UI_RUN_KEY')

    def test_connection_probe_bounded_no_echo_and_no_settings_save(self):
        calls=[]
        class Transport:
            def __init__(self,endpoint,timeout):self.endpoint=endpoint
            def __call__(self,url,body,headers):
                calls.append(json.loads(body))
                return 401,'synthetic-test-key provider-private-error'
        with patch.object(web.adapter,'HttpsTransport',Transport):result=self.settings.test({})
        self.assertFalse(result['ok'])
        self.assertNotIn('synthetic-test-key',json.dumps(result))
        self.assertEqual(calls[0]['max_tokens'],32)
        self.assertEqual(calls[0]['thinking'],{'type':'disabled'})
        self.assertFalse(self.settings.path.exists())


class ResultsAndJobs(Fixture):
    def test_only_used_citations_and_same_run_image_exposed(self):
        result=web.result_from_run('a'*32,self.run_fixture(),self.root)
        self.assertEqual(result['status'],'answered')
        self.assertEqual([e['citation_id'] for e in result['evidence']],['S2'])
        self.assertNotIn('[S2]',result['answer_markdown'])
        self.assertEqual(result['evidence'][0]['text'],'Figure evidence.')
        self.assertIn('evidence/S2/image',result['evidence'][0]['image_url'])
        self.assertEqual(result['evidence'][0]['related_answer'],'Answer supported by the figure.')

    def test_unknown_citation_fails_closed(self):
        with self.assertRaises(web.UIError):web.result_from_run('a'*32,self.run_fixture('Unsupported [S99].'),self.root)

    def test_citation_possessive_stays_grammatical_on_main_view(self):
        self.assertEqual(web.display_answer('这是 [S2] 的原图。'), '这是 对应证据的原图。')

    def test_evidence_tampering_fails_closed(self):
        run=self.run_fixture()
        path=run/'05-context/contexts.json'
        contexts=web.read_json(path)
        contexts['cases'][0]['citation_map'][1]['text_sha256']='tampered'
        web.write_json(path,contexts)
        with self.assertRaises(web.UIError):web.result_from_run('a'*32,run,self.root)

    def test_no_answer_is_not_failed(self):
        result=web.result_from_run('a'*32,self.run_fixture(None,'no_answer_no_evidence'),self.root)
        self.assertEqual(result['status'],'no_answer')
        self.assertIsNone(result['error'])
        self.assertTrue(result['answer_markdown'])
        self.assertEqual(result['evidence'],[])

    def test_failed_answer_is_not_displayed(self):
        result=web.result_from_run('a'*32,self.run_fixture('Unsafe incomplete answer','failed'),self.root)
        self.assertEqual(result['answer_markdown'],'')
        self.assertEqual(result['status'],'failed')

    def test_unindexed_catalog_notes_are_not_listed_as_searchable(self):
        path=self.root/'data/metadata/documents.json'
        catalog=web.read_json(path)
        catalog['documents'].append({'document_id':'SOURCES','document_title':'Sources list',
                                     'source_path':'原始资料/paper.md','include_in_index':False})
        web.write_json(path,catalog)
        app=web.Workbench(root=self.root,settings=self.settings,start_worker=False)
        self.assertEqual([d['id'] for d in app.documents()['documents']],['paper'])
        with self.assertRaises(web.UIError):app.document('SOURCES')

    def test_job_settings_snapshot_and_persistent_interruption(self):
        with patch.object(self.app,'health',return_value={'ready':True}):
            job=self.app.submit('Question one')
        directory=self.app.job_dir(job['id'])
        self.settings.save({'language':'en','detail':'detailed'})
        self.assertEqual(web.read_json(directory/'job.json')['settings']['language'],'auto')
        self.assertNotIn('synthetic-test-key',''.join(p.read_text(encoding='utf-8') for p in directory.iterdir()))
        self.assertNotIn('settings',self.app.get_job(job['id']))
        second=web.Workbench(root=self.root,settings=self.settings,start_worker=False)
        self.assertEqual(second.get_job(job['id'])['status'],'interrupted')
        self.assertTrue(second.queue.empty())
        self.assertEqual(len(second.history()['items']),1)

    def test_queue_backpressure_and_question_validation(self):
        with patch.object(self.app,'health',return_value={'ready':True}):
            for _ in range(8):self.app.submit('Question')
            with self.assertRaises(web.UIError) as ctx:self.app.submit('Extra')
            self.assertEqual(ctx.exception.status,429)
            for query in ('','x'*2001,123,'\x00'):
                with self.assertRaises(web.UIError):self.app.submit(query)

    def test_same_submission_reuses_job_even_after_settings_change(self):
        with patch.object(self.app,'health',return_value={'ready':True}):
            first=self.app.submit('Question','b'*32)
            self.settings.save({'language':'en'})
            replay=self.app.submit('Question','b'*32)
            self.assertEqual(first['id'],replay['id'])
            self.assertEqual(self.app.queue.qsize(),1)
            with self.assertRaises(web.UIError):self.app.submit('Changed','b'*32)
            other=self.app.submit('Question','c'*32)
            self.assertNotEqual(first['id'],other['id'])
        with self.assertRaises(web.UIError):self.app.submit('Question','invalid')

    def test_gpu_jobs_run_serially(self):
        started=[]
        worker_environments=[]
        release=threading.Event()
        done=threading.Event()
        class Process:
            def __init__(self,*a,**kw):
                self.n=len(started)
                started.append(self.n)
                worker_environments.append(kw['env'].copy())
            def poll(self):return 0 if release.is_set() or self.n>0 else None
        with patch.object(self.app,'health',return_value={'ready':True}),patch.object(web.subprocess,'Popen',Process),\
             patch.object(web,'result_from_run',return_value={'status':'answered','stage':'已完成','answer_markdown':'Answer','evidence':[]}):
            first=self.app.submit('One')
            second=self.app.submit('Two')
            self.app.thread=threading.Thread(target=self.app.work,daemon=True)
            self.app.thread.start()
            deadline=time.monotonic()+3
            while not started and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(len(started),1)
            self.assertEqual(self.app.get_job(second['id'])['status'],'queued')
            release.set()
            self.app.queue.join()
            self.assertEqual(len(started),2)
            self.assertTrue(all(env['RAG_BGE_MODEL_DIR']==str(self.app.bge_model_dir)
                                for env in worker_environments))
            self.assertEqual(self.app.get_job(first['id'])['status'],'answered')
            self.app.close()


class ModelDiscovery(Fixture):
    def test_standard_download_is_ready_without_shell_environment(self):
        model=self.root/'models/bge-reranker-v2-m3'
        model.mkdir(parents=True)
        (model/'model.safetensors').write_bytes(b'fixture')
        index=self.root/'data/runtime/index'
        index.mkdir(parents=True)
        (index/'records.jsonl').write_text('',encoding='utf-8')
        web.write_json(index/'manifest.json',{})
        with patch.dict(os.environ,{},clear=True),patch.object(web.shutil,'which',return_value='node'):
            app=web.Workbench(root=self.root,settings=self.settings,start_worker=False)
            self.assertEqual(app.bge_model_dir,model)
            self.assertTrue(app.health()['ready'])
            app.close()

    def test_explicit_model_directory_has_priority(self):
        with patch.dict(os.environ,{'RAG_BGE_MODEL_DIR':'models/custom-bge'}):
            self.assertEqual(web.resolve_bge_model_dir(self.root),self.root/'models/custom-bge')

    def test_development_model_fallback_is_preserved(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(web.resolve_bge_model_dir(self.root),
                self.root/'data/metadata/retrieval-eval/experiment-29-bge-v2-m3/model/bge-reranker-v2-m3')


class HTTPBoundaries(Fixture):
    def setUp(self):
        super().setUp()
        self.server=web.Server(('127.0.0.1',0),self.app)
        self.server_thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.server_thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.server_thread.join()
        super().tearDown()

    def request(self,path,body=None,method='GET',headers=None):
        data=json.dumps(body).encode() if body is not None else None
        headers={'Content-Type':'application/json',**(headers or {})}
        try:
            response=urlopen(Request(self.base+path,data=data,method=method,headers=headers),timeout=4)
        except HTTPError as error:response=error
        with response:return response.status,response.read(),dict(response.headers)

    def test_static_allowlist_and_host_origin_controls(self):
        for path in ('/.env','/../.env','/%2e%2e/.env','/scripts/_rag_web.py','/data/ui/settings.json'):
            status,body,_=self.request(path)
            self.assertEqual(status,404)
            self.assertNotIn(b'private-file-sentinel',body)
        self.assertEqual(self.request('/api/settings',headers={'Origin':'https://evil.example'})[0],403)
        self.assertEqual(self.request('/api/settings',headers={'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('/api/settings',headers={'Sec-Fetch-Site':'cross-site'})[0],403)
        self.assertEqual(self.request('/api/settings',{},'PUT',{'Content-Type':'text/plain'})[0],415)

    def test_public_settings_documents_and_security_headers(self):
        status,body,headers=self.request('/api/settings')
        self.assertEqual(status,200)
        self.assertNotIn(b'synthetic-test-key',body)
        self.assertNotIn('Access-Control-Allow-Origin',headers)
        self.assertIn("script-src 'self'",headers['Content-Security-Policy'])
        self.assertEqual(self.request('/api/documents/paper')[0],200)
        self.assertEqual(self.request('/api/documents/../../.env')[0],404)

    def test_http_submit_poll_preferences_and_missing_resources(self):
        with patch.object(self.app,'health',return_value={'ready':True}):
            status,body,_=self.request('/api/questions',{'query':'A new question'},'POST')
        self.assertEqual(status,202)
        job=json.loads(body)
        self.assertEqual(self.request('/api/questions/'+job['id'])[0],200)
        self.assertEqual(self.request('/api/questions/'+job['id']+'/evidence/S99/image')[0],404)
        status,body,_=self.request('/api/settings',{'language':'zh','detail':'concise'},'PUT')
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['language'],'zh')
        self.assertEqual(self.request('/api/questions/'+'a'*32)[0],404)


if __name__=='__main__':unittest.main()
