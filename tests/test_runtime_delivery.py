"""E41 release path: runtime index format, loader refusals, release whitelist.

Offline. No model weights are loaded and no network is used. The one test
that rebuilds records from the real Markdown needs the pinned GME *tokenizer*
in the local Hugging Face cache and is skipped when it is absent.
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import _runtime_bundle as rb  # noqa: E402
import _prepare_delivery as delivery  # noqa: E402

DIM = 1536
E41 = ROOT / 'data/metadata/retrieval-eval/experiment-41-final-delivery'


def unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def catalog_for(tmp: Path) -> Path:
    (tmp / 'src').mkdir(parents=True, exist_ok=True)
    (tmp / 'src/doc.md').write_text('# Doc\n\nHello.\n', encoding='utf-8')
    (tmp / 'src/fig.png').write_bytes(b'\x89PNG\r\n\x1a\n')
    path = tmp / 'data/metadata/documents.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'documents': [{
        'document_id': 'doc', 'document_title': 'Doc', 'source_path': 'src/doc.md',
        'source_url': 'https://example.test/doc', 'include_in_index': True}]}), encoding='utf-8')
    return path


def meta(**extra):
    base = {'document_id': 'doc', 'document_title': 'Doc', 'source_path': 'src/doc.md',
            'source_url': 'https://example.test/doc', 'section_path': []}
    base.update(extra)
    return base


def records():
    return [
        {'schema_version': 3, 'chunk_id': 'doc::text::0001', 'kind': 'text', 'seq': 1,
         'text': 'Body paragraph about retrieval.', 'metadata': meta()},
        {'schema_version': 3, 'chunk_id': 'doc::text::0002', 'kind': 'text', 'seq': 2,
         'text': 'Figure 1: Overview.', 'metadata': meta()},
        {'schema_version': 3, 'chunk_id': 'doc::figure::f1', 'kind': 'visual_description', 'seq': 3,
         'text': 'Figure 1: Overview. A diagram.', 'retrieval_text': 'TITLE: Doc\nFigure 1: Overview. A diagram.',
         'metadata': meta(image_path='src/fig.png', visual_type='figure', label='Figure 1',
                          prev_text_chunk_id='doc::text::0002', next_text_chunk_id=None,
                          caption_chunk_ids=['doc::text::0002'], relation_kind='caption',
                          association_provenance={'origin': 'test'}, text_is_image_generated=True)},
    ]


class RuntimeIndexFormat(unittest.TestCase):
    def build(self, tmp: Path, recs=None, ref=('doc::text::0002',)):
        recs = recs or records()
        body_ids, desc_ids = rb.split_ids(recs, ref)
        body = np.stack([unit(i) for i, _ in enumerate(body_ids)])
        desc = np.stack([unit(100 + i) for i, _ in enumerate(desc_ids)])
        index = tmp / 'index'
        rb.write_runtime_index(index, recs, list(ref), body, desc, {'test': True})
        return index

    def test_round_trip_gives_the_runner_bundle_shape(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            catalog = catalog_for(tmp)
            bundle = rb.load_runtime_bundle(self.build(tmp), catalog_path=catalog)
            for key in ('dir', 'manifest', 'config', 'catalog', 'chunks', 'by_id', 'body', 'body_ids',
                        'body_row', 'desc', 'desc_ids', 'desc_row', 'all_ids', 'filter_records',
                        'reference_only_ids', 'mode', 'selected_layers'):
                self.assertIn(key, bundle)
            self.assertEqual(bundle['body_ids'], ['doc::text::0001'])
            self.assertEqual(bundle['desc_ids'], ['doc::figure::f1'])
            self.assertEqual(bundle['reference_only_ids'], {'doc::text::0002'})
            self.assertIn('doc::text::0002', bundle['by_id'], 'reference-only text stays readable')
            import _description_store as ds
            self.assertEqual(set(ds.current_index_hashes(bundle)),
                             {'config', 'body_vectors', 'description_vectors', 'all_ids', 'inherited_gme_v1'})

    def test_the_accepted_ranking_code_runs_on_a_runtime_bundle(self):
        import _rag_e2e as runner
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            catalog = catalog_for(tmp)
            bundle = rb.load_runtime_bundle(self.build(tmp), catalog_path=catalog)
            dedup, _ = runner.selected_modules()
            plan = {'id': 'q1', 'original_query': 'retrieval overview', 'intent': 'text',
                    'english_query': 'retrieval overview',
                    'requests': [{'id': 'r1', 'evidence_types': ['text'], 'user_evidence': 'retrieval overview',
                                  'english_query': 'retrieval overview',
                                  'filter': {'document_ids': [], 'source_evidence': [],
                                             'visual_labels': [], 'label_evidence': []}}]}
            results, _ = dedup.rank_plans_dedup([plan], bundle, lambda t: unit(0), 10, caption_route=True)
            hits = results[0]['requests'][0]['groups'][0]['hits']
            self.assertEqual([h['chunk_id'] for h in hits], ['doc::text::0001'],
                             'the reference-only caption must not be ranked as text')

    def test_tampered_vectors_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            catalog = catalog_for(tmp)
            index = self.build(tmp)
            np.save(index / 'body_vectors.npy', np.stack([unit(7)]), allow_pickle=False)
            with self.assertRaises(rb.RuntimeIndexError):
                rb.load_runtime_bundle(index, catalog_path=catalog)

    def test_writing_into_a_non_empty_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / 'index').mkdir()
            (tmp / 'index' / 'old.txt').write_text('x')
            with self.assertRaises(rb.RuntimeIndexError):
                self.build(tmp)

    def test_vector_count_must_match_ids(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(rb.RuntimeIndexError):
                rb.write_runtime_index(Path(d) / 'i', records(), ['doc::text::0002'],
                                       np.stack([unit(1), unit(2)]), np.stack([unit(3)]), {})

    def test_the_legacy_layered_index_is_not_mistaken_for_a_runtime_index(self):
        self.assertFalse(rb.is_runtime_index(ROOT / 'data' / 'index' / 'description-v1'))

    def test_caption_link_to_another_document_is_refused(self):
        recs = records()
        recs[2]['metadata']['caption_chunk_ids'] = ['other::text::0001']
        with self.assertRaises(rb.RuntimeIndexError):
            rb.validate_records(recs, [])

    def test_reference_only_id_must_be_text(self):
        with self.assertRaises(rb.RuntimeIndexError):
            rb.validate_records(records(), ['doc::figure::f1'])

    def test_metadata_must_agree_with_the_catalog(self):
        with tempfile.TemporaryDirectory() as d:
            catalog = json.loads(catalog_for(Path(d)).read_text(encoding='utf-8'))
            recs = records()
            recs[0]['metadata']['source_url'] = 'https://elsewhere.test'
            with self.assertRaises(rb.RuntimeIndexError):
                rb.validate_records(recs, [], catalog)


class ReleaseWhitelist(unittest.TestCase):
    def test_forbidden_paths(self):
        for rel in ('.env', 'config/.env.local', '.venv/Lib/site.py', 'data/index/description-v1/body_vectors.npy',
                    'data/chunks/rag-guide.chunks.jsonl', 'models/bge/model.safetensors',
                    'data/metadata/retrieval-eval/experiment-40-e2e-repairs/costs.json',
                    'scripts/__pycache__/x.pyc', 'secrets/api_key.txt'):
            self.assertIsNotNone(delivery.is_forbidden(rel), rel)
        for rel in ('.env.example', 'scripts/_rag_e2e.py', 'README.md',
                    'data/metadata/retrieval-eval/experiment-22-caption-dedup/claude_caption_dedup.py',
                    'data/metadata/retrieval-eval/experiment-41-final-delivery/release-inputs/visual-descriptions.jsonl'):
            self.assertIsNone(delivery.is_forbidden(rel), rel)

    def test_planned_release_contains_no_pdf_html_index_or_results(self):
        planned = [rel for rel, _ in delivery.planned_files()]
        self.assertTrue(planned)
        for rel in planned:
            self.assertFalse(rel.lower().endswith(('.pdf', '.html', '.npy', '.npz')), rel)
            self.assertIsNone(delivery.is_forbidden(rel), rel)

    def test_every_catalog_markdown_and_referenced_image_is_planned(self):
        planned = {rel for rel, _ in delivery.planned_files()}
        catalog = json.loads((ROOT / 'data/metadata/documents.json').read_text(encoding='utf-8'))
        for doc in catalog['documents']:
            if doc.get('include_in_index'):
                self.assertIn(doc['source_path'], planned)
        descriptions = (E41 / 'release-inputs/visual-descriptions.jsonl').read_text(encoding='utf-8')
        for line in descriptions.splitlines():
            if line.strip():
                self.assertIn(json.loads(line)['metadata']['image_path'], planned)


class ReleaseInputs(unittest.TestCase):
    def test_reviewed_inputs_match_their_recorded_provenance(self):
        prov = json.loads((E41 / 'release-inputs/provenance.json').read_text(encoding='utf-8'))
        for key in ('visual_descriptions', 'reference_only_text_ids'):
            path = E41 / 'release-inputs' / prov[key]['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), prov[key]['sha256'])
        self.assertEqual(prov['visual_descriptions']['records'], 68)
        self.assertEqual(prov['reference_only_text_ids']['count'], 24)

    def test_no_vector_file_is_a_release_input(self):
        for path in (E41 / 'release-inputs').iterdir():
            self.assertNotIn(path.suffix, {'.npy', '.npz', '.pt', '.safetensors'})


class RunnerIntegration(unittest.TestCase):
    """The runtime loader is wired into _rag_e2e (E41 patch applied)."""

    def test_runner_loads_a_runtime_index_through_load_selected_bundle(self):
        import _rag_e2e as runner
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            catalog = catalog_for(tmp)
            index = RuntimeIndexFormat.build(self, tmp)
            original = rb.CATALOG_PATH
            rb.CATALOG_PATH = catalog
            try:
                bundle = runner.load_selected_bundle(index)
            finally:
                rb.CATALOG_PATH = original
            self.assertEqual(bundle['selected_layers']['runtime_format'], rb.RUNTIME_FORMAT)
            self.assertIn('standalone', runner.index_provenance(bundle))
            self.assertIn('freshly built', runner.index_provenance(bundle))
            self.assertNotIn('verified E16 base', runner.index_provenance(bundle))

    def test_default_layered_index_path_is_unchanged(self):
        import _rag_e2e as runner
        self.assertEqual(runner.INDEX_DIR, ROOT / 'data' / 'index' / 'description-v1')
        self.assertFalse(rb.is_runtime_index(runner.INDEX_DIR))
        legacy = {'selected_layers': {'base_index': str(runner.INDEX_DIR), 'mode': 'x'}}
        self.assertEqual(runner.index_provenance(legacy),
                         'verified E16 base + E23 enriched descriptions + E22 caption dedup; no rebuild')

    @unittest.skipUnless((ROOT / 'data' / 'index' / 'description-v1' / 'manifest.json').is_file(),
                         'original layered index is not part of a release tree')
    def test_default_layered_index_still_loads(self):
        import _rag_e2e as runner
        bundle = runner.load_selected_bundle()
        self.assertNotIn('runtime_format', bundle['selected_layers'])
        self.assertEqual(len(bundle['chunks']), 437)

    def test_bge_identity_follows_the_directory_in_use(self):
        import _rag_e2e as runner
        original = runner.BGE_MODEL_DIR
        try:
            runner.BGE_MODEL_DIR = runner.E29_BGE_MODEL_DIR
            self.assertEqual(runner._bge_reused_from_experiment(), 'E29')
            with tempfile.TemporaryDirectory() as d:
                runner.BGE_MODEL_DIR = Path(d)
                self.assertIsNone(runner._bge_reused_from_experiment())
                self.assertIn('RAG_BGE_MODEL_DIR', runner._bge_model_source())
        finally:
            runner.BGE_MODEL_DIR = original


def tokenizer_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download
        cfg = json.loads((ROOT / 'config/runtime.json').read_text(encoding='utf-8'))['models']['embedding']
        path = Path(snapshot_download(cfg['repo'], revision=cfg['revision'], local_files_only=True))
        return (path / 'tokenizer.json').is_file()
    except Exception:
        return False


@unittest.skipUnless(tokenizer_cached(), 'pinned GME tokenizer not in the local Hugging Face cache')
class RebuildFromMarkdown(unittest.TestCase):
    def test_records_are_derived_without_reading_chunk_or_index_snapshots(self):
        import _rebuild_runtime as rebuild
        opened = []
        watching = {'on': True}
        chunks = str((ROOT / 'data' / 'chunks').resolve()).lower()
        index = str((ROOT / 'data' / 'index').resolve()).lower()

        def hook(event, args):
            if watching['on'] and event == 'open' and args and isinstance(args[0], (str, Path)):
                p = str(Path(args[0]).resolve()).lower() if not str(args[0]).startswith('<') else ''
                if p.startswith(chunks) or p.startswith(index):
                    opened.append(p)

        sys.addaudithook(hook)
        try:
            recs, ref, report = rebuild.build_records(rebuild.load_config())
        finally:
            watching['on'] = False
        self.assertEqual(opened, [])
        self.assertEqual(report['counts'], {'records': 437, 'text': 369, 'visual_description': 68,
                                            'reference_only': 24, 'body': 345, 'documents': 10})
        self.assertTrue(all(row['ok'] for row in
                            report['carried_reviewed_inputs']['reference_only_text_ids']['checks']))


if __name__ == '__main__':
    unittest.main()
