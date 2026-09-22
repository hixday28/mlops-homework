#!/usr/bin/env python3
"""Отдельный гейт: проверяются все три пары train/val/test."""
import json
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_params
from src.schema import iter_examples
from src.contamination import report,is_clean


def main():
    p=load_params();paths=p['paths'];nd=p['clean']['near_dup']
    buckets={name:list(iter_examples(paths[name])) for name in ['train','val','test']}
    reports={}
    for left,right in combinations(buckets,2):
        r=report(buckets[left],buckets[right],nd['shingle_words'],nd['num_perm'],p['contamination']['threshold'])
        reports[f'{left}_{right}']=r
        print(f"{left} ↔ {right}: пересечение по id: {r['id_overlap']}; по тексту: {r['text_overlap']}; по группам: {r['group_overlap']}; near-dup: {r['near_dup_pairs']}")
    passed=all(is_clean(r) for r in reports.values()) and all(buckets.values())
    out=Path(paths['metrics_contamination']);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'passed':passed,'pairs':reports},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('контаминации нет' if passed else 'КОНТАМИНАЦИЯ или пустой сплит')
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
