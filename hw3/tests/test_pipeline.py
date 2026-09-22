"""Регрессии для гейтов, дедупликации и контракта данных; без сетевых запросов."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import diversity
from src.config import load_params
from src.dedup import cross_near_duplicates, near_duplicates
from src.pii import scrub
from src.schema import Example, SchemaError, iter_examples
from src.textnorm import normalize_group, shingles


class PipelineTests(unittest.TestCase):
    def test_index_matches_brute_force_jaccard(self):
        rng=random.Random(42)
        texts=[' '.join(rng.choices(['aa','bb','cc','dd','ee','ff'],k=rng.randrange(2,15))) for _ in range(80)]
        texts += [texts[0], texts[0]+' zz']
        for threshold in [0.5,0.85,1.0]:
            left,right=texts[:40],texts[40:]
            a=[shingles(t,2) for t in left];b=[shingles(t,2) for t in right]
            expected={(i,j) for i,x in enumerate(a) for j,y in enumerate(b)
                      if len(x&y)/len(x|y)>=threshold}
            self.assertEqual(set(cross_near_duplicates(left,right,2,64,threshold)),expected)

    def test_near_dup_keeps_only_first_representative(self):
        a=' '.join(f'слово{i}' for i in range(100))
        self.assertEqual(near_duplicates([a,a+' добавка','совсем другой вопрос'],4,64,.85),[1])

    def test_pii_masking_makes_questions_identical(self):
        a='Для связи user1@example.org, номер +7 (999) 123-45-67'
        b='Для связи user2@example.org, номер +7 (999) 234-56-78'
        self.assertEqual(scrub(a)[0],scrub(b)[0])
        masked,hits=scrub('дата рождения 12.03.1975; выпуск 12.03.2024; /Users/test/code; token="abcde"')
        self.assertIn('[DATE]',masked)
        self.assertIn('12.03.2024',masked)
        self.assertIn('/Users/[USER]/code',masked)
        self.assertIn('[SECRET]',masked)
        self.assertEqual(sum(hits.values()),3)

    def test_error_reports_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'broken.jsonl';p.write_text('{}\n',encoding='utf8')
            with self.assertRaisesRegex(SchemaError,r'broken.jsonl:1:'):
                list(iter_examples(p))

    def test_blank_topic_rejected(self):
        with self.assertRaises(ValueError):
            Example(id='a',topic=' ',messages=[{'role':r,'content':'text'} for r in ['system','user','assistant']])

    def test_group_aliases_normalize(self):
        self.assertEqual(normalize_group(' Docker — Linux | alias '),normalize_group('docker - linux'))

    def test_threshold_mismatch_rejected(self):
        p=load_params();p['contamination']['threshold']=.8
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'params.yaml';f.write_text(yaml.safe_dump(p),encoding='utf8')
            with self.assertRaisesRegex(ValueError,'должны совпадать'):load_params(str(f))

    def test_degenerate_dataset_has_multiple_violations(self):
        cfg=load_params()['diversity']
        stats=dict(examples=1200,system_prompts=1,groups=1,largest_group_share=1.,
                   largest_group='одна',answer_len={'ratio_p90_p10':1.},same_length_share=1.,duplicate_answer_share=.99)
        self.assertEqual(len(diversity.violations(stats,cfg)),6)

    def test_diversity_failure_exits_nonzero_and_writes_evidence(self):
        p=load_params()
        with tempfile.TemporaryDirectory() as tmp:
            p['paths']['metrics_diversity']=str(Path(tmp)/'metrics.json')
            stats=dict(examples=1200,system_prompts=1,groups=1,largest_group_share=1.,
                       largest_group='одна',answer_len={'ratio_p90_p10':1.},same_length_share=1.,duplicate_answer_share=.99)
            with patch.object(diversity,'load_params',return_value=p),patch.object(diversity,'measure',return_value=stats):
                with self.assertRaises(SystemExit):diversity.main()
            self.assertFalse(json.loads(Path(p['paths']['metrics_diversity']).read_text())['passed'])


if __name__=='__main__':unittest.main()
