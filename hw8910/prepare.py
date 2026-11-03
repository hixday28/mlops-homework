"""Проверка входа и длин отрендеренных промптов; результат — JSON, не тензоры."""
import json
from pathlib import Path
from release import atomic_json,cache_key,digest


def validate_data(root,input_path,config,tokenizer_path=None,generation=None):
    root=Path(root);path=Path(input_path);payload=path.read_bytes();lines=payload.decode('utf-8').splitlines()
    if not lines or any(not x.strip() for x in lines):raise ValueError('Пустая строка/вход')
    lengths=[];ids=[]
    if tokenizer_path:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
        if digest(tok.chat_template.encode())!=config['template_sha256']:raise ValueError('Чужой шаблон')
        for line in lines:
            row=json.loads(line)
            if not isinstance(row.get('id'),str) or not row['id'] or not isinstance(row.get('input'),str) or not row['input'].strip():
                raise ValueError('Неверная схема строки')
            if not isinstance(row.get('choices'),list) or row.get('expected') not in row['choices']:
                raise ValueError('Неверные choices/expected')
            ids.append(row['id'])
            messages=[{'role':'system','content':generation['system']},{'role':'user','content':row['input']}]
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=generation['enable_thinking'])
            length=len(tok(text)['input_ids']);lengths.append(length)
            if length>config['max_length']:raise ValueError('Вход превышает лимит токенов')
        if len(ids)!=len(set(ids)):raise ValueError('Повторный ID')
    else:
        lengths=[len(x) for x in lines]
        if max(lengths)>config['max_length']:raise ValueError('Строка STUB превышает max_length')
    key=cache_key(path,config);target=root/'prepared'/f'{key}.json'
    result={'input_sha256':digest(payload),'rows':len(lines),'max_length_observed':max(lengths),
            'config':config,'kind':'rendered-token-length-validation' if tokenizer_path else 'STUB ONLY line validation'}
    atomic_json(target,result)
    print('PREPARED',key,flush=True)
    return {'uri':str(target.resolve()),'sha256':digest(target.read_bytes()),'input_sha256':digest(payload),'rows':len(lines),'cache_key':key}
