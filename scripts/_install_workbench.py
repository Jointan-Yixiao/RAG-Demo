"""First-run preparation for the Windows launcher. Never reads API credentials.

--check is offline. --install downloads only the pinned dependencies/models and
builds the local index when necessary; it never calls a generation API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
import venv
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


class SetupError(Exception):
    pass


def safe_path(path, root=ROOT):
    path, root = Path(path).absolute(), Path(root).resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise SetupError('安装目录超出项目范围。')
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink() or item.is_junction():
            raise SetupError('安装目录包含链接，请使用普通项目目录。')
    return resolved


def pinned_requirements(root=ROOT):
    pins = {'torch':'2.11.0+cu128','torchvision':'0.26.0+cu128'}
    for line in (Path(root)/'requirements-runtime-lock.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            name, version = line.split('==',1)
            pins[name] = version
    return pins


def requirements_match(root=ROOT):
    try:
        return all(importlib.metadata.version(name) == version
                   for name, version in pinned_requirements(root).items())
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        return False


def capture(command, root=ROOT, timeout=30):
    try:
        p = subprocess.run(command, cwd=root, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=timeout,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        return p.returncode, p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return 1, ''


def check_state(root=ROOT):
    root = Path(root).resolve()
    python = root/'.venv/Scripts/python.exe'
    state = dict(python_ready=sys.version_info[:2]==(3,12), node_ready=False,
                 requirements_ready=False, models_ready=False, index_ready=False,
                 ready=False, message='等待初始化运行环境。')
    node = shutil.which('node')
    if node:
        code, version = capture([node,'--version'],root,10)
        state['node_ready'] = code==0 and bool(re.match(r'^v24\.',version))
    if not state['python_ready']:
        state['message']='请安装 Python 3.12，再运行初始化。'
        return state
    if not state['node_ready']:
        state['message']='请安装 Node.js 24.x，再运行初始化。'
    if not python.is_file():
        return state
    if python.resolve() != Path(sys.executable).resolve():
        code, output = capture([str(python),'-B','-X','utf8',str(root/'scripts/_install_workbench.py'),'--check'],root,90)
        if code==0:
            try:
                result=json.loads(output)
                if isinstance(result,dict) and 'ready' in result:
                    return result
            except ValueError:
                pass
        state['message']='项目 Python 环境无法读取，请查看安装说明。'
        return state
    state['requirements_ready'] = requirements_match(root)
    if not state['requirements_ready']:
        state['message']='点击“初始化环境”安装项目依赖。'
        return state
    try:
        from huggingface_hub import snapshot_download
        from _rag_web import resolve_bge_model_dir
        cfg=json.loads((root/'config/runtime.json').read_text(encoding='utf-8'))
        embedding=cfg['models']['embedding']
        snapshot=Path(snapshot_download(embedding['repo'],revision=embedding['revision'],local_files_only=True))
        shards=list(snapshot.glob('*.safetensors'))
        shard_index=snapshot/'model.safetensors.index.json'
        if shard_index.is_file():
            weight_map=json.loads(shard_index.read_text(encoding='utf-8'))['weight_map']
            shards=[snapshot/name for name in set(weight_map.values())]
        gme=bool(shards) and all(p.is_file() and p.stat().st_size>0 for p in shards)
        gme=gme and (snapshot/'config.json').is_file() and (snapshot/'tokenizer.json').is_file()
        bge=resolve_bge_model_dir(root)
        bge_ok=all((bge/name).is_file() and (bge/name).stat().st_size==spec['size']
                   for name,spec in cfg['models']['reranker']['files'].items())
        state['models_ready']=gme and bge_ok
    except (OSError,ValueError,KeyError,ImportError):
        pass
    try:
        import _rag_e2e as runner
        runtime=root/'data/runtime/index'
        index=runtime if (runtime/'records.jsonl').is_file() else root/'data/index/description-v1'
        bundle=runner.load_selected_bundle(index)
        state['index_ready']=bool(bundle['chunks'])
    except Exception:
        pass
    state['ready']=all(state[k] for k in ('python_ready','node_ready','requirements_ready','models_ready','index_ready'))
    if state['ready']:
        state['message']='本地环境已就绪，可打开工作台。API 密钥在网页设置中配置。'
    elif state['node_ready']:
        state['message']='点击“初始化环境”准备缺少的模型或资料索引。'
    return state


def stage(key, message):
    print('STAGE|'+key+'|'+message,flush=True)


def run(command, root=ROOT):
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PIP_DISABLE_PIP_VERSION_CHECK='1')
    child=subprocess.Popen(command,cwd=root,env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if child.wait()!=0:
        raise SetupError('此步骤未完成，请查看详细日志后重试。已有资料和索引已保留。')


def promote_index(staged, destination, root=ROOT):
    staged=safe_path(staged,root);destination=safe_path(destination,root)
    if not staged.is_dir():
        raise SetupError('新的索引尚未生成。')
    backup=None
    if destination.exists():
        backup=safe_path(destination.with_name(destination.name+'-before-'+uuid.uuid4().hex[:12]),root)
        destination.rename(backup)
    try:
        staged.rename(destination)
    except Exception:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    return backup


def install(root=ROOT):
    root=Path(root).resolve()
    if os.name!='nt' or sys.version_info[:2]!=(3,12):
        raise SetupError('初始化需要 Windows x64 和 Python 3.12。')
    try:
        with urlopen('http://127.0.0.1:8765/api/health',timeout=2) as response:
            health=json.loads(response.read())
    except (OSError,ValueError):
        health={}
    if health.get('service')=='rag-workbench' and health.get('busy'):
        raise SetupError('工作台正在处理问题，请完成后再初始化。')
    state=check_state(root)
    if not state['node_ready']:
        raise SetupError('请先安装 Node.js 24.x，然后重新打开启动器。')
    if state['ready']:
        stage('done','运行环境、模型和资料均已就绪，无需重新下载。')
        return state
    python=safe_path(root/'.venv/Scripts/python.exe',root)
    if not python.is_file():
        environment=safe_path(root/'.venv',root)
        if environment.exists() and any(environment.iterdir()):
            raise SetupError('现有 .venv 不完整，已保留。请按安装说明检查后重试。')
        stage('environment','正在创建项目 Python 环境。')
        venv.EnvBuilder(with_pip=True).create(environment)
    code,version=capture([str(python),'-I','-c','import sys; print("%d.%d" % sys.version_info[:2])'],root)
    if code!=0 or version!='3.12':
        raise SetupError('项目环境需要 Python 3.12；现有环境已保留。')
    if not state['requirements_ready']:
        stage('dependencies','正在安装固定版本依赖，首次运行需要下载较大文件。')
        run([str(python),'-m','pip','install','torch==2.11.0+cu128','torchvision==0.26.0+cu128',
             '--index-url','https://download.pytorch.org/whl/cu128'],root)
        run([str(python),'-m','pip','install','-r',str(root/'requirements-runtime-lock.txt')],root)
        run([str(python),'-m','pip','check'],root)
    # All remaining steps must use the freshly prepared virtual environment.
    if python.resolve()!=Path(sys.executable).resolve():
        run([str(python),'-B','-X','utf8',str(root/'scripts/_install_workbench.py'),'--finish-install'],root)
        return check_state(root)
    return finish_install(root)


def finish_install(root=ROOT):
    root=Path(root).resolve()
    from _rag_web import resolve_bge_model_dir
    python=root/'.venv/Scripts/python.exe'
    rebuild=root/'scripts/_rebuild_runtime.py'
    state=check_state(root)
    if not state['requirements_ready']:
        raise SetupError('固定版本依赖尚未安装完整，请查看日志。')
    bge=resolve_bge_model_dir(root)
    if not bge.is_dir():
        bge=root/'models/bge-reranker-v2-m3'
    safe_path(bge,root)
    stage('models','正在核对模型缓存；缺少的固定版本模型将自动下载。')
    run([str(python),'-B','-X','utf8',str(rebuild),'models','--download','--bge-dir',str(bge)],root)
    state=check_state(root)
    if not state['index_ready']:
        stage('index','正在从原始资料建立索引，此步骤在本机运行。')
        session=safe_path(root/'data/runtime'/('setup-'+uuid.uuid4().hex[:12]),root)
        session.mkdir(parents=True)
        run([str(python),'-B','-X','utf8',str(rebuild),'records','--out',str(session/'records')],root)
        run([str(python),'-B','-X','utf8',str(rebuild),'encode','--records',str(session/'records'),'--out',str(session/'index')],root)
        run([str(python),'-B','-X','utf8',str(rebuild),'verify','--index',str(session/'index')],root)
        promote_index(session/'index',root/'data/runtime/index',root)
    final=check_state(root)
    if not final['ready']:
        raise SetupError(final['message'])
    stage('done','初始化完成。请打开工作台，在设置页填写 API 密钥。')
    return final


def main(argv=None):
    parser=argparse.ArgumentParser(description='Prepare the local RAG workbench without paid API calls')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--install',action='store_true')
    group.add_argument('--finish-install',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.check:
        print(json.dumps(check_state(),ensure_ascii=False),flush=True)
        return 0
    directory=safe_path(ROOT/'data/ui')
    directory.mkdir(parents=True,exist_ok=True)
    lockfile=(directory/'setup.lock').open('a+b')
    try:
        if os.name=='nt' and not args.finish_install:
            import msvcrt
            if lockfile.tell()==0:
                lockfile.write(b'1');lockfile.flush()
            lockfile.seek(0)
            try:
                msvcrt.locking(lockfile.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:
                raise SetupError('另一个初始化正在运行，请等待它完成。') from None
        result=finish_install() if args.finish_install else install()
        (directory/'setup-result.json').write_text(json.dumps({'completed_at':datetime.now(timezone.utc).isoformat(),
            'status':result,'paid_api_calls':0},ensure_ascii=False,indent=2),encoding='utf-8')
        return 0
    except SetupError as error:
        stage('error',str(error))
        return 1
    except Exception:
        stage('error','初始化未完成，请查看日志和安装说明。已有文件已保留。')
        return 1
    finally:
        lockfile.close()


if __name__=='__main__':
    raise SystemExit(main())
