"""Преобразовать тематический срез Stack Overflow в chat JSONL и атрибуцию."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from src.config import load_params, source_files
from src.schema import Example, dump


def pick_prompt(example_id, variants):
    return variants[int(hashlib.sha256(example_id.encode()).hexdigest(), 16) % len(variants)]


def license_for(timestamp):
    # Версия по дате исходного сообщения; поздние правки в архиве не размечены.
    return 'CC BY-SA 2.5' if timestamp < 1302220800 else ('CC BY-SA 3.0' if timestamp < 1525219200 else 'CC BY-SA 4.0')


def main():
    p = load_params(); cfg=p['collect']; paths=p['paths']
    metrics=Counter(); candidates=[]; seen=set()
    for path in source_files(p):
        for line in path.open(encoding='utf8'):
            row=json.loads(line); metrics['rows_scanned']+=1
            qid=str(row['question_id'])
            if qid in seen:
                metrics['duplicate_source_ids']+=1; continue
            seen.add(qid)
            answers=[a for a in row['answers'] if a['is_accepted'] and a['score']>=cfg['min_answer_score']]
            if len(answers)!=1:
                metrics['dropped_answer_filter']+=1; continue
            answer=answers[0]
            user=row['title'].strip()+'\n\n'+row['text_markdown'].strip()
            body=answer['text_markdown'].strip()
            if len(re.findall('[А-Яа-яЁё]',body))<cfg['min_cyrillic_chars'] or len(body)<cfg['min_answer_chars']:
                metrics['dropped_answer_language_length']+=1; continue
            if '<img' in row['text_html'] or '<img' in answer['text_html']:
                metrics['dropped_image_dependency']+=1; continue
            if not user.strip() or not body:
                metrics['dropped_empty']+=1; continue
            topic=' / '.join(sorted(set(row['tags'])))
            record=Example(id='ru-so-'+qid, topic=topic, messages=[
                {'role':'system','content':pick_prompt(qid,cfg['system_prompts'])},
                {'role':'user','content':user},{'role':'assistant','content':body}])
            attribution={'id':record.id,'question_url':row['url'],'question_author':row['author'],
                         'question_title':row['title'],
                         'answer_url':f"https://ru.stackoverflow.com/a/{answer['answer_id']}",
                         'answer_author':answer['author'],'question_timestamp':row['timestamp'],
                         'answer_timestamp':answer['timestamp'],'question_license':license_for(row['timestamp']),
                         'answer_license':license_for(answer['timestamp']),
                         'accepted':True,'answer_score':answer['score'], 'source_tags':row['tags']}
            for kind in ['question','answer']:
                attribution[kind+'_license_url']='https://creativecommons.org/licenses/by-sa/'+attribution[kind+'_license'].split()[-1]+'/'
            attribution['adaptation_license_url']='https://creativecommons.org/licenses/by-sa/4.0/'
            attribution['changes']='Выбор принятого ответа, добавление system и заголовка; далее фильтры, маскирование и дедупликация.'
            candidates.append((record,attribution))
    # Один и тот же порядок в обеих версиях: v1 является префиксом v2.
    candidates.sort(key=lambda x: hashlib.sha256(x[0].id.encode()).hexdigest())
    groups=Counter(); selected=[]
    for ex,attribution in candidates:
        if groups[ex.topic]>=cfg['max_per_group']:
            metrics['dropped_group_cap']+=1; continue
        if len(selected)>=cfg['limits'][cfg['version']]:break
        groups[ex.topic]+=1;selected.append((ex,attribution))
    for key in ['raw','attribution']:Path(paths[key]).parent.mkdir(parents=True,exist_ok=True)
    with Path(paths['raw']).open('w',encoding='utf8') as raw, Path(paths['attribution']).open('w',encoding='utf8') as attrs:
        for ex,attribution in selected:
            raw.write(dump(ex)+'\n');attrs.write(json.dumps(attribution,ensure_ascii=False)+'\n')
    result=dict(metrics,version=cfg['version'],rows_written=len(selected),eligible_candidates=len(candidates),
                not_selected_limit=len(candidates)-len(selected)-metrics['dropped_group_cap'],
                groups=len(groups),system_prompt_variants=len({x.messages[0].content for x,_ in selected}),
                transformation='tag scope; accepted positive-score answers; no image dependencies; chat conversion; deterministic system variants')
    target=Path(paths['metrics_collect']);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(f"collect: {result['rows_scanned']} → {len(selected)}, версия {cfg['version']}, групп {len(groups)}")


if __name__=='__main__':main()
