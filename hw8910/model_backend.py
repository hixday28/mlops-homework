"""Настоящая генерация из принятого manifest, без чтения params.yaml."""
from importlib.metadata import version
from pathlib import Path
from release import canonical,digest,file_digest


class TransformersBackend:
    label='transformers (real model)'
    def __init__(self,manifest,root):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.manifest=manifest;self.root=Path(root);self.runtime=manifest['runtime'];self.calls=0
        self.loaded_id=manifest['model_id'];self.runtime_sha256=manifest['runtime_sha256']
        for package,expected in self.runtime['library_versions'].items():
            if version(package)!=expected:raise ValueError(f'Изменена библиотека {package}: нужна повторная оценка')
        artifact=self.root/manifest['local_artifact']
        self.tokenizer=AutoTokenizer.from_pretrained(artifact,local_files_only=True)
        files={n:file_digest(artifact/n) for n in self.runtime['tokenizer_files']}
        if digest(canonical(files))!=self.runtime['tokenizer_sha256'] or digest(self.tokenizer.chat_template.encode())!=self.runtime['template_sha256']:
            raise ValueError('Не совпали токенизатор/шаблон')
        self.device=torch.device(self.runtime['device']);dtype=getattr(torch,self.runtime['dtype'])
        torch.manual_seed(self.runtime['generation']['seed'])
        if manifest['model_kind']=='base':self.model=AutoModelForCausalLM.from_pretrained(artifact,local_files_only=True,dtype=dtype)
        else:
            from peft import PeftModel
            import json
            cfg=json.loads((artifact/'manifest.json').read_text())['model']
            if cfg['revision']!=self.runtime['base_revision'] or cfg['dtype']!=self.runtime['dtype']:raise ValueError('Чужая база или dtype адаптера')
            base=AutoModelForCausalLM.from_pretrained(cfg['name'],revision=self.runtime['base_revision'],local_files_only=True,dtype=dtype)
            self.model=PeftModel.from_pretrained(base,artifact)
        self.model=self.model.to(self.device).eval()

    def _inputs(self,prompt):
        g=self.runtime['generation']
        text=self.tokenizer.apply_chat_template([{'role':'system','content':g['system']},{'role':'user','content':prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=g['enable_thinking'])
        return self.tokenizer(text,return_tensors='pt')

    def validate_prompt(self,prompt):
        if self._inputs(prompt)['input_ids'].shape[1]>self.runtime['max_input_tokens']:
            raise ValueError('Превышен лимит входных токенов')

    def generate(self,prompt,limit):
        import torch
        ids=self._inputs(prompt).to(self.device);self.calls+=1
        with torch.inference_mode():
            result=self.model.generate(**ids,max_new_tokens=limit,do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        if self.device.type=='mps':torch.mps.synchronize()
        generated=result[0,ids['input_ids'].shape[1]:]
        return {'answer':self.tokenizer.decode(generated,skip_special_tokens=True),'output_tokens':len(generated)}

    def warmup(self):
        self.generate('Вопрос по MLOps: для чего нужен реестр моделей? A. Версии. B. Рецепты. choices: A, B, REFUSE.',self.runtime['generation']['max_new_tokens'])
