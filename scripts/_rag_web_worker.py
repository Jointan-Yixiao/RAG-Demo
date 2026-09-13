"""Isolated UI job: reuse the accepted pipeline, adding progress notifications."""
from pathlib import Path
import argparse
import functools
import json
import os
import sys
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True, type=Path)
    parser.add_argument('--index-dir', required=True, type=Path)
    args = parser.parse_args()
    job = args.job_dir.resolve()
    import _rag_e2e as runner
    import _answer_semantics as review

    def progress(stage):
        tmp = job / (uuid.uuid4().hex + '.tmp')
        tmp.write_text(json.dumps({'stage': stage}, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, job / 'progress.json')

    def wrap(module, name, stage):
        original = getattr(module, name)
        @functools.wraps(original)
        def call(*a, **kw):
            progress(stage)
            return original(*a, **kw)
        setattr(module, name, call)

    for name, stage in [('run_frontend','正在理解问题'), ('embed_queries','正在检索资料'),
                        ('retrieve','正在检索资料'), ('rerank','正在筛选相关证据'),
                        ('build_contexts','正在整理证据'), ('generate','正在生成回答'),
                        ('export_answers','正在整理回答与引用')]:
        wrap(runner, name, stage)
    wrap(review, 'run_stage', '正在核对回答')
    return runner.main(['--queries-file', str(job/'query.json'), '--out-dir', str(job/'run'),
                        '--config',str(job/'generation.json'), '--rules-file',str(job/'rules.txt'),
                        '--index-dir',str(args.index_dir), '--execute'])


if __name__ == '__main__':
    sys.exit(main())
