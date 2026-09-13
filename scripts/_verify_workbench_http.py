"""Check a completed real job through public HTTP APIs; never re-run that job."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from _rag_web import ROOT, read_json, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--test-connection',action='store_true',help='Make one 32-output-token API probe (may cost money).')
    args=parser.parse_args()
    def get(path):
        with urlopen(args.url+path,timeout=10) as response:return json.load(response)
    health=get('/api/health')
    assert health['service']=='rag-workbench' and health['ready']
    with urlopen(args.url,timeout=10) as response:
        html=response.read().decode('utf-8')
        assert '/app.js' in html and '/styles.css' in html
    for path in ('/app.js','/styles.css'):
        with urlopen(args.url+path,timeout=10) as response:assert len(response.read())>100
    history=get('/api/history')['items']
    assert any(item['id']==args.job_id for item in history)
    job=get('/api/questions/'+args.job_id)
    assert job['status']=='answered' and job['answer_markdown']
    assert not re.search(r'\[S\d+\]',job['answer_markdown'])
    assert '_assets' not in job and 'settings' not in job
    persisted=read_json(ROOT/'data/ui/jobs'/args.job_id/'job.json')
    image_checks=[]
    for e in job['evidence']:
        assert e['text'] and e['related_answer']
        if e['image_url']:
            with urlopen(args.url+e['image_url'],timeout=10) as response:
                data=response.read()
                assert response.headers['Content-Type'] in ('image/png','image/jpeg')
            source=Path(persisted['_assets'][e['citation_id']]).read_bytes()
            assert hashlib.sha256(source).digest()==hashlib.sha256(data).digest()
            image_checks.append({'citation_id':e['citation_id'],'label':e['label'],'bytes':len(data),'original_identical':True})
    documents=get('/api/documents')['documents']
    for document in documents:
        assert get('/api/documents/'+document['id'])['text']
    settings=get('/api/settings')
    assert set(settings)=={'provider','base_url','model','language','detail','key_configured'}
    saved={k:v for k,v in settings.items() if k!='key_configured'}
    req=Request(args.url+'/api/settings',data=json.dumps(saved).encode(),headers={'Content-Type':'application/json'},method='PUT')
    with urlopen(req,timeout=10) as response:assert json.load(response)==settings
    assert get('/api/settings')==settings
    probe=None
    if args.test_connection:
        req=Request(args.url+'/api/settings/test',data=json.dumps(saved).encode(),headers={'Content-Type':'application/json'},method='POST')
        with urlopen(req,timeout=55) as response:probe=json.load(response)
        assert probe['ok'],probe['message']
    report={'job_id':args.job_id,'question':job['query'],'status':job['status'],
        'answer_chars':len(job['answer_markdown']),'inline_citations_removed':True,'used_evidence_count':len(job['evidence']),
        'images':image_checks,'documents_accessible':len(documents),'settings_roundtrip':True,'credential_never_returned':True,
        'history_readable':True,'usage':job['usage'],'connection_probe':probe,
        'browser_click_verification':'pending Chrome ERR_BLOCKED_BY_CLIENT'}
    write_json(ROOT/'data/ui/acceptance/http-verification.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
