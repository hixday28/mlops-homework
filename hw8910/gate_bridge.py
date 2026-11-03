"""Пересчёт старого отчёта из raw, затем неизменённое правило ДЗ 6+7."""
import json
import os
from pathlib import Path
import sys
from release import canonical,digest,file_digest,atomic_json

PROJECT=Path(__file__).resolve().parent
OLD=PROJECT.parent/'hw67'


def read(path):return json.loads(Path(path).read_text())


def check_saved_evaluation():
    sys.path.insert(0,str(OLD))
    from evaluation.contract import fingerprint,read_rows,validate_golden
    from evaluation.solution import score,quality_gate
    cfg=read(OLD/'evaluation/params.json');base=read(OLD/'evaluation/baseline.json')
    rows=read_rows(OLD/cfg['golden']);behavior=read_rows(OLD/cfg['behavioral']);training=read_rows(OLD/cfg['training_index'])
    validate_golden(rows,training,behavior)
    protocol=dict(config=cfg,golden=rows,behavioral=behavior,training_index=training,
                  evaluation_code={p.name:p.read_text() for p in sorted((OLD/'evaluation').glob('*.py'))})
    reports={};sources=[OLD/'evaluation/params.json',OLD/'evaluation/baseline.json',OLD/cfg['golden'],OLD/cfg['behavioral'],OLD/cfg['training_index']]
    sources+=list((OLD/'evaluation').glob('*.py'))
    for role in ('baseline','candidate'):
        report_path=OLD/f'reports/{role}-eval.json';outputs_path=OLD/f'reports/{role}-outputs.json'
        sources += [report_path,outputs_path];report=read(report_path);outputs=read(outputs_path)
        if (report['protocol_sha256']!=fingerprint(protocol) or report['golden_sha256']!=fingerprint(rows)
            or report['baseline_spec_sha256']!=fingerprint(base) or report['outputs_sha256']!=fingerprint(outputs)):
            raise ValueError('Отчёт устарел: протокол, golden, baseline или raw изменены')
        by_id={x['id']:x for x in outputs}
        if len(by_id)!=len(outputs) or set(by_id)!={r['id'] for r in rows+behavior}:raise ValueError('Изменены ID ответов')
        calculated=score(rows,[by_id[r['id']] for r in rows]);bh=score(behavior,[by_id[r['id']] for r in behavior])
        if any(report[k]!=calculated[k] for k in ('overall','slices','per_item')) or report['behavioral']!=bh:
            raise ValueError('Метрики не соответствуют raw')
        reports[role]=report
    spec=read(OLD/'reports/candidate-model.json')
    if reports['candidate']['model']!=spec or reports['baseline']['model']!=base:raise ValueError('Чужая версия модели')
    return reports,sources,cfg,quality_gate


def decision(root,manifest,prepared_input_sha256=None):
    root=Path(root)
    if manifest.get('model_kind')=='stub' or manifest['runtime']['base_revision']=='fixture-only':
        report=read(root/'report-B.json');report['synthetic']=True
        atomic_json(root/'release-report.json',report);return report
    report={k:manifest[k] for k in ('model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256')}
    report.update(status='failed',passed=False,evaluation_run_id=manifest.get('evaluation_run_id',''),metrics={})
    try:
        reports,sources,cfg,quality_gate=check_saved_evaluation()
        reference=read(PROJECT/'evidence/candidate-manifest.json')
        for key in ('model_id','model_sha256','runtime_sha256','protocol_sha256','golden_sha256'):
            if manifest[key]!=reference[key]:raise ValueError('Manifest/runtime изменён относительно связанной оценки')
        sources+=[PROJECT/'evidence/candidate-manifest.json',OLD/'uv.lock']
        if prepared_input_sha256 is not None and prepared_input_sha256!=file_digest(OLD/cfg['golden']):
            raise ValueError('Подготовленные данные отличаются от набора оценки; нужна новая оценка')
        candidate=reports['candidate']
        if candidate['model']['run_id']!=manifest['model_id'] or candidate['protocol_sha256']!=manifest['protocol_sha256'] or candidate['golden_sha256']!=manifest['golden_sha256']:
            raise ValueError('Manifest не принадлежит оценённому кандидату')
        g=manifest['runtime']['generation']
        if any(g[k]!=cfg[k] for k in ('system','max_new_tokens','seed','enable_thinking')) or g['do_sample'] is not False:
            raise ValueError('Изменены настройки генерации; нужна новая оценка')
        expected_revision=read(OLD/'evaluation/baseline.json')['revision']
        if manifest['runtime']['base_revision']!=expected_revision:raise ValueError('Изменена база')
        report.update(status='complete',metrics={k:candidate['overall'][k] for k in ('accuracy','format_rate')},
                      sources_root=os.path.relpath(PROJECT,root),
                      source_bindings={os.path.relpath(p,PROJECT):file_digest(p) for p in sources})
        try:
            result=quality_gate(reports['baseline'],candidate,cfg['critical_slices'])
            report.update(passed=result['passed'],gate=result)
        except ValueError as exc:report['reason']=str(exc)
    except (ValueError,KeyError,FileNotFoundError) as exc:report['reason']=str(exc)
    atomic_json(root/'release-report.json',report)
    return report


def verify_source_bindings(root,report):
    for name,expected in report.get('source_bindings',{}).items():
        p=Path(root)/report['sources_root']/name
        if file_digest(p)!=expected:raise ValueError('Исходный отчёт/данные изменены после допуска')
