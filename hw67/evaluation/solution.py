"""Строгий scorer меток и правило допуска без разрешённой регрессии."""
import json
import math
from collections import defaultdict


def _ids(items, label):
    if not isinstance(items, list) or not items:
        raise ValueError(f'{label}: пустой набор')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'] or item['id'] in result:
            raise ValueError(f'{label}: отсутствующий или повторный id')
        result[item['id']] = item
    return result


def _counts(items):
    n = len(items)
    correct = sum(item['correct'] for item in items)
    valid = sum(item['valid'] for item in items)
    return dict(n=n, correct=correct, valid=valid,
                errors=sum(item['error'] is not None for item in items),
                accuracy=correct / n, format_rate=valid / n)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повторный ключ JSON')
        result[key] = value
    return result


def score(rows, outputs):
    tasks, answers = _ids(rows, 'rows'), _ids(outputs, 'outputs')
    if tasks.keys() != answers.keys():
        raise ValueError('Наборы id заданий и ответов различаются')
    per_item, groups = [], defaultdict(list)
    for ident, row in tasks.items():
        if not isinstance(row.get('choices'), list) or not row['choices'] or row.get('expected') not in row['choices']:
            raise ValueError(f'{ident}: неверные choices/expected')
        slices = row.get('slices')
        if not isinstance(slices, list) or not slices or any(not isinstance(s, str) or not s for s in slices) or len(set(slices)) != len(slices):
            raise ValueError(f'{ident}: неверные срезы')
        answer = answers[ident]
        raw, error = answer.get('raw'), answer.get('error')
        if error is not None:
            error = str(error) or 'generation error'
        parsed, valid = None, False
        if error is None and isinstance(raw, str):
            try:
                obj = json.loads(raw, object_pairs_hook=_unique_object)
                if isinstance(obj, dict) and set(obj) == {'answer'} and isinstance(obj['answer'], str) and obj['answer'] in row['choices']:
                    parsed, valid = obj['answer'], True
            except (ValueError, TypeError):
                pass
        item = dict(id=ident, raw=raw, parsed=parsed, valid=valid,
                    correct=valid and parsed == row['expected'], error=error, slices=list(slices))
        per_item.append(item)
        for name in slices:
            groups[name].append(item)
    return dict(overall=_counts(per_item), slices={k: _counts(v) for k, v in sorted(groups.items())}, per_item=per_item)


def _validate_report(report, minimum):
    if not isinstance(report, dict):
        raise ValueError('Отчёт отсутствует')
    items = _ids(report.get('per_item'), 'per_item')
    if len(items) < minimum:
        raise ValueError(f'Недостаточно примеров: {len(items)} < {minimum}')
    groups = defaultdict(list)
    for item in items.values():
        if type(item.get('valid')) is not bool or type(item.get('correct')) is not bool or 'error' not in item:
            raise ValueError('Неполный per_item')
        if item['correct'] and not item['valid'] or item['error'] is not None and item['valid']:
            raise ValueError('Противоречивые valid/correct/error')
        if item['valid'] and not isinstance(item.get('parsed'), str):
            raise ValueError('Валидный ответ без строковой метки')
        slices = item.get('slices')
        if not isinstance(slices, list) or not slices or any(not isinstance(s, str) or not s for s in slices) or len(set(slices)) != len(slices):
            raise ValueError('Нет срезов per_item')
        for name in slices:
            groups[name].append(item)

    def verify(actual, expected):
        if not isinstance(actual, dict):
            raise ValueError('Отсутствуют counts/метрики')
        for key in ('n', 'correct', 'valid', 'errors'):
            if type(actual.get(key)) is not int or actual[key] != expected[key]:
                raise ValueError(f'Counts не соответствуют per_item: {key}')
        for key in ('accuracy', 'format_rate'):
            val = actual.get(key)
            if type(val) not in (int, float) or not math.isfinite(val) or not math.isclose(val, expected[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'Метрика не соответствует counts: {key}')
        if actual['errors']:
            raise ValueError('Есть ошибки генерации')

    verify(report.get('overall'), _counts(list(items.values())))
    slices = report.get('slices')
    if not isinstance(slices, dict) or slices.keys() != groups.keys():
        raise ValueError('Список срезов не соответствует per_item')
    for key, members in groups.items():
        verify(slices[key], _counts(members))
    return items


def quality_gate(baseline, candidate, critical_slices):
    try:
        if not isinstance(critical_slices, list) or not critical_slices or any(not isinstance(s, str) for s in critical_slices):
            raise ValueError('Нужен список критических срезов')
        for report in (baseline, candidate):
            if report.get('status') != 'complete' or not isinstance(report.get('model'), dict) or not report['model'] or not report['model'].get('kind'):
                raise ValueError('Отчёт неполный или модель не определена')
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if not isinstance(baseline.get(key), str) or not baseline[key] or baseline[key] != candidate.get(key):
                raise ValueError(f'Несовместимость: {key}')
        a, b = _validate_report(baseline, 50), _validate_report(candidate, 50)
        if a.keys() != b.keys():
            raise ValueError('Наборы id различаются')
        if len(baseline['slices']) < 3 or baseline['slices'].keys() != candidate['slices'].keys():
            raise ValueError('Нужны одинаковые минимум три среза')
        for ident in a:
            if set(a[ident]['slices']) != set(b[ident]['slices']):
                raise ValueError(f'Изменились срезы вопроса {ident}')
        for key in ('accuracy', 'format_rate'):
            if candidate['overall'][key] < baseline['overall'][key]:
                raise ValueError(f'Регрессия {key}: {candidate["overall"][key]:.4f} < {baseline["overall"][key]:.4f}')
        for name in critical_slices:
            if name not in baseline['slices']:
                raise ValueError(f'Нет критического среза {name}')
            if candidate['slices'][name]['accuracy'] < baseline['slices'][name]['accuracy']:
                raise ValueError(f'Регрессия критического среза {name}')
        ba = _validate_report(baseline.get('behavioral'), 3)
        bc = _validate_report(candidate.get('behavioral'), 3)
        if ba.keys() != bc.keys() or any(set(ba[i]['slices']) != set(bc[i]['slices']) for i in ba):
            raise ValueError('Behavioral-наборы несовместимы')
        if not all(item['correct'] for item in bc.values()):
            raise ValueError('Обязательный behavioral-кейс не пройден')
        return dict(passed=True, baseline=baseline['overall'], candidate=candidate['overall'],
                    critical_slices=critical_slices, behavioral_passed=len(bc),
                    rule='Нулевая регрессия accuracy/format_rate и критических срезов; все behavioral проходят')
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f'Неполный или некорректный отчёт: {exc}') from exc
