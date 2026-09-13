"""Offline first-run safety tests; no package install, downloads or model calls."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import _install_workbench as setup
import _launch_workbench as launch


class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.local_service=patch.object(setup,'urlopen',side_effect=OSError('offline fixture'))
        self.local_service.start()

    def tearDown(self):
        self.local_service.stop()
        self.temp.cleanup()

    def test_output_path_cannot_escape_root(self):
        with self.assertRaises(setup.SetupError):setup.safe_path(self.root.parent/'outside',self.root)
        with self.assertRaises(setup.SetupError):setup.safe_path(self.root,self.root)
        self.assertEqual(setup.safe_path(self.root/'data/runtime/index',self.root),self.root/'data/runtime/index')

    def test_index_promotion_retains_previous_files(self):
        old=self.root/'data/runtime/index';old.mkdir(parents=True)
        (old/'old.json').write_text('previous')
        new=self.root/'data/runtime/setup-1/index';new.mkdir(parents=True)
        (new/'new.json').write_text('new')
        backup=setup.promote_index(new,old,self.root)
        self.assertEqual((backup/'old.json').read_text(),'previous')
        self.assertEqual((old/'new.json').read_text(),'new')

    def test_failed_promotion_restores_previous_index(self):
        old=self.root/'index';old.mkdir();(old/'proof').write_text('keep')
        new=self.root/'stage';new.mkdir()
        original=Path.rename
        def rename(path,target):
            if path==new:raise OSError('fixture failure')
            return original(path,target)
        with patch.object(Path,'rename',rename),self.assertRaises(OSError):setup.promote_index(new,old,self.root)
        self.assertEqual((old/'proof').read_text(),'keep')
        self.assertTrue(new.exists())

    def test_pinned_requirements_include_cuda_torch_and_lockfile(self):
        (self.root/'requirements-runtime-lock.txt').write_text('# pinned\nnumpy==2.5.2\n',encoding='utf-8')
        pins=setup.pinned_requirements(self.root)
        self.assertEqual(pins,{'torch':'2.11.0+cu128','torchvision':'0.26.0+cu128','numpy':'2.5.2'})

    def test_mismatched_dependency_is_not_reported_ready(self):
        with patch.object(setup,'pinned_requirements',return_value={'numpy':'2.5.2'}),\
             patch.object(setup.importlib.metadata,'version',return_value='1.0'):
            self.assertFalse(setup.requirements_match(self.root))

    @unittest.skipUnless(os.name=='nt' and sys.version_info[:2]==(3,12),'Windows Python 3.12 preparation')
    def test_ready_install_does_not_download_or_rebuild(self):
        with patch.object(setup,'check_state',return_value={'node_ready':True,'ready':True}),\
             patch.object(setup,'run') as run:
            self.assertTrue(setup.install(self.root)['ready'])
            run.assert_not_called()

    @unittest.skipUnless(os.name=='nt' and sys.version_info[:2]==(3,12),'Windows Python 3.12 preparation')
    def test_missing_node_stops_before_creating_environment(self):
        with patch.object(setup,'check_state',return_value={'node_ready':False,'ready':False}),\
             patch.object(setup,'run') as run,self.assertRaises(setup.SetupError):
            setup.install(self.root)
        run.assert_not_called()
        self.assertFalse((self.root/'.venv').exists())

    @unittest.skipUnless(os.name=='nt' and sys.version_info[:2]==(3,12),'Windows Python 3.12 preparation')
    def test_partial_environment_is_preserved(self):
        (self.root/'.venv').mkdir();(self.root/'.venv/keep.txt').write_text('keep')
        with patch.object(setup,'check_state',return_value={'node_ready':True,'ready':False}),\
             self.assertRaises(setup.SetupError):setup.install(self.root)
        self.assertEqual((self.root/'.venv/keep.txt').read_text(),'keep')

    @unittest.skipUnless(os.name=='nt' and sys.version_info[:2]==(3,12),'Windows Python 3.12 preparation')
    def test_active_rag_job_blocks_environment_mutation(self):
        response=BytesIO(json.dumps({'service':'rag-workbench','busy':True}).encode())
        with patch.object(setup,'urlopen',return_value=response),patch.object(setup,'run') as run,\
             self.assertRaises(setup.SetupError):setup.install(self.root)
        run.assert_not_called()


class WorkspaceOwnership(unittest.TestCase):
    def test_healthy_does_not_reuse_another_repository_service(self):
        wrong={'service':'rag-workbench','workspace_id':'another-repository','ready':True}
        with patch.object(launch,'urlopen',return_value=BytesIO(json.dumps(wrong).encode())):
            self.assertFalse(launch.healthy('http://127.0.0.1:8765'))

    def test_healthy_accepts_matching_repository(self):
        identity=hashlib.sha256(str(launch.ROOT.resolve()).rstrip('\\/').lower().encode()).hexdigest()[:16]
        own={'service':'rag-workbench','workspace_id':identity,'ready':True}
        with patch.object(launch,'urlopen',return_value=BytesIO(json.dumps(own).encode())):
            self.assertTrue(launch.healthy('http://127.0.0.1:8765'))


if __name__=='__main__':unittest.main()
