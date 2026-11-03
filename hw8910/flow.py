"""Prefect: подготовка → существующий артефакт → исходный quality gate → публикация."""
import argparse
import json
import os
from pathlib import Path
import uuid

PROJECT=Path(__file__).resolve().parent
os.environ.setdefault('PREFECT_HOME',str(PROJECT/'.prefect'))
os.environ.setdefault('PREFECT_API_URL','http://127.0.0.1:4208/api')
os.environ.setdefault('PREFECT_SERVER_ANALYTICS_ENABLED','false')
from prefect import flow,task,get_run_logger
from prefect.filesystems import LocalFileSystem
from prefect.runtime import flow_run
from release import atomic_json,cache_key,file_digest,verify_files,validate_release
from prepare import validate_data
from publication import append,publish_once
from gate_bridge import decision,verify_source_bindings


def read(path):return json.loads(Path(path).read_text())


def input_key(context,parameters):
    return cache_key(Path(parameters['input_path']),parameters['config'])


@task(name='prepare-input',persist_result=True,result_serializer='json',cache_key_fn=input_key,retries=0)
def prepare_task(root,input_path,config,tokenizer_path,generation):
    return validate_data(PROJECT/'.prefect/validated',input_path,config,tokenizer_path,generation)


@task(name='get-existing-artifact',persist_result=False,retries=0)
def artifact_task(root,manifest_name):
    manifest=read(Path(root)/manifest_name);verify_files(manifest,root)
    get_run_logger().info('READ EXISTING ARTIFACT %s — no training',manifest['artifact_uri'])
    return {'uri':manifest['artifact_uri'],'manifest_uri':str(Path(root)/manifest_name),'model_id':manifest['model_id'],
            'sha256':manifest['model_sha256']}


@task(name='check-complete-evaluation',persist_result=False,retries=0)
def gate_task(root,artifact,prepared):
    manifest=read(artifact['manifest_uri']);report=decision(root,manifest,prepared['input_sha256'])
    if report.get('passed') is not True:get_run_logger().error('GATE REJECTED: %s',report.get('reason','synthetic refusal'))
    validate_release(manifest,report)
    return {'report_uri':str(Path(root)/'release-report.json'),'evaluation_run_id':report['evaluation_run_id']}


@task(name='publish-accepted',persist_result=False,retries=0)
def publish_task(root,artifact,evaluation,fail_after_pointer=False):
    manifest=read(artifact['manifest_uri']);report=read(evaluation['report_uri'])
    verify_source_bindings(root,report)
    ident=publish_once(root,manifest,report,fail_after_pointer=fail_after_pointer)
    return {'release_id':ident,'model_id':manifest['model_id']}


@flow(name='model-release',log_prints=True,retries=0)
def release_flow(workspace:str,fail_after_prepare:bool=False,fail_after_pointer:bool=False,stub:bool=False,refresh_preparation:bool=False):
    root=Path(workspace).resolve();attempt_id=str(uuid.uuid4())
    append(root/'attempts.jsonl',{'attempt_id':attempt_id,'flow_run_id':str(flow_run.id)})
    config=read(root/'config.json');config['implementation_sha256']=file_digest(PROJECT/'prepare.py')
    tokenizer_path=None;generation=None
    if not stub:
        base=read(root/'manifest-baseline.json');verify_files(base,root)
        config['tokenizer_sha256']=base['runtime']['tokenizer_sha256']
        generation=base['runtime']['generation'];config['generation']=generation
        tokenizer_path=str(root/base['local_artifact'])
    storage=LocalFileSystem(basepath=str(PROJECT/'.prefect/results'))
    storage.save('hw8910-results',overwrite=True)
    state=prepare_task.with_options(result_storage='local-file-system/hw8910-results',refresh_cache=refresh_preparation)(
        str(root),str(root/('input.txt' if stub else 'input.jsonl')),config,tokenizer_path,generation,return_state=True)
    prepared=state.result()
    entry={'attempt_id':attempt_id,'flow_run_id':str(flow_run.id),'task':'prepare','state':state.name,
           'cache_key':prepared['cache_key'],'input_sha256':prepared['input_sha256'],'rows':prepared['rows']}
    append(root/'task-states.jsonl',entry);get_run_logger().info('PREPARE_STATE %s %s',state.name,prepared['cache_key'])
    if fail_after_prepare:raise RuntimeError('Инъекция сбоя после подготовки')
    artifact=artifact_task(str(root),'manifest-B.json' if stub else 'manifest-candidate.json')
    evaluation=gate_task(str(root),artifact,prepared)
    return publish_task(str(root),artifact,evaluation,fail_after_pointer)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',default='.real');p.add_argument('--stub',action='store_true')
    p.add_argument('--fail-after-prepare',action='store_true');p.add_argument('--fail-after-pointer',action='store_true')
    p.add_argument('--refresh-preparation',action='store_true')
    a=p.parse_args();print(release_flow(str(Path(a.workspace).resolve()),a.fail_after_prepare,a.fail_after_pointer,a.stub,a.refresh_preparation))


if __name__=='__main__':main()
