"""Сборка docs/anatomy.md и графика норм активаций."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без дисплея: скрипт должен работать и в CI

import matplotlib.pyplot as plt  # noqa: E402  (backend выбирается до импорта)

MODE_TITLES = {
    "inference": "инференс",
    "full_ft": "full fine-tune",
    "lora": "LoRA (r=8, q/v)",
}


def thousands(n: int) -> str:
    """Число с неразрывными пробелами по разрядам."""
    return f"{n:,}".replace(",", " ")


def plot_activations(activations: dict, path: str) -> None:
    """Две панели: норма по позициям токена и средняя норма по трём блокам."""
    labels = list(activations["norms"])
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=(2, 1))

    for label in labels:
        values = activations["norms"][label]
        ax_left.plot(values, linewidth=1.4,
                     label=f"{label} (слой {activations['layers'][label]})")
    ax_left.set_xlabel("позиция токена")
    ax_left.set_yscale("log")   # без лога всё придавит выброс massive activations
    ax_left.set_ylabel("‖h‖₂ (лог. шкала)")
    ax_left.set_title("Норма скрытого состояния по позициям")
    ax_left.legend(fontsize=9)
    ax_left.grid(alpha=0.3)

    means = [sum(activations["norms"][x]) / len(activations["norms"][x]) for x in labels]
    ax_right.bar(labels, means, color=["#4c78a8", "#f58518", "#54a24b"])
    ax_right.set_ylabel("средняя ‖h‖₂")
    ax_right.set_title("Средняя норма по блоку")
    ax_right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def markdown_report(report: dict, params: dict) -> str:
    c, e, acts = report['config'], report['environment'], report['activations']
    modes = {x['mode']: x for x in report['memory']}
    inference = modes['inference']['peak_mb']
    mib = lambda n: n / 2**20
    lines = [f"# Анатомия {report['model'].split('/')[-1]}", '',
             f"Дата разбора (UTC): {report['timestamp_utc']}. Источник чисел: `report.json`.", '',
             '## 1. Конфигурация и параметры', '', '| Параметр | Значение |', '|---|---:|']
    lines += [f'| {key} | {value} |' for key, value in c.items()]
    lines += ['', '| Группа | Модулей | Shape | Уникальных параметров | Доля | Общих параметров |',
              '|---|---:|---|---:|---:|---:|']
    for row in report['params_by_group']:
        lines.append(f"| {row['group']} | {row['modules']} | {row['shape']} | {thousands(row['params'])} | {row['share']:.4%} | {thousands(row['tied_params'])} |")
    lines += [f"| Итого | | | {thousands(report['params_total'])} | 100% | |", '',
              f"Прямой подсчёт по model.parameters(): **{thousands(report['params_direct'])}**. Разница с таблицей: **{report['params_total'] - report['params_direct']}**.",
              'Все строки «имя параметра → shape → число → tied» сохранены в `report.json`.', '',
              'Общие входные и выходные веса учитываются один раз. В строке lm_head',
              'показан размер переиспользуемой матрицы, но он не прибавляется к итогу.',
              'У SwiGLU три широкие проекции, поэтому суммарный MLP тяжелее attention.',
              f"GQA использует {c['num_attention_heads']} Q-голов и {c['num_key_value_heads']} KV-голов: при одинаковых длине контекста и dtype размер KV-cache составляет {c['num_key_value_heads']/c['num_attention_heads']:.2f} от MHA.", '',
              '## 2. Условия замера', '', '| Условие | Значение |', '|---|---|',
              f"| платформа | {e['platform']} |", f"| устройство | {e['device']} |",
              f"| dtype базовых весов | {e['dtype']} |", f"| seq_len × batch | {e['seq_len']} × {e['batch_size']} |",
              f"| прогонов на режим | {e['repeats']}, отдельный процесс на каждый прогон |",
              f"| память измерена | {e['memory_metric']} |", f"| дополнительная RSS | {e['rss_metric']} |",
              f"| Python | {e['python']} |", f"| torch | {e['torch']} |",
              f"| transformers | {e['transformers']} |", f"| peft | {e['peft']} |",
              f"| seed / learning rate | {e['seed']} / {e['lr']} |",
              f"| use_cache | {e['use_cache']} |", '| оптимизатор | AdamW, один шаг |', '',
              'Вход — случайные token ID, одинаковые при одинаковом seed. Измеряется',
              'инференс (один forward под inference_mode), full fine-tune и LoRA',
              '(forward + backward + optimizer.step). Базовые веса перед каждым',
              'режимом загружаются заново. Родитель освобождает свою модель перед замерами.',
              'Общее время включает загрузку; отдельно сохранено время вычислительного шага.', '',
              'На MPS память драйвера опрашивается каждые 10 мс, а также после загрузки,',
              'forward, backward, шага оптимизатора и очистки градиентов. Это максимум',
              'наблюдений: очень короткий пик между выборками может быть пропущен.',
              'Метрика включает буферы Metal/MPSGraph и кеш аллокатора. На CUDA используется',
              'max_memory_allocated со сбросом high-water mark; на CPU — пик RSS.',
              'Все размеры обозначены в MiB (2²⁰ байт); исторические поля JSON *_mb имеют те же единицы.', '',
              '## 3. Активации и хуки', '', f"Промпт: «{params['hooks']['prompt']}». После chat template: {acts['n_tokens']} токенов.",
              f"![Нормы активаций]({Path(params['hooks']['plot']).name})", '',
              '| Блок | Индекс | Средняя L2-норма | Максимальная | Позиция максимума |', '|---|---:|---:|---:|---:|']
    for label, index in acts['layers'].items():
        values = acts['norms'][label]
        lines.append(f'| {label} | {index} | {sum(values)/len(values):.3f} | {max(values):.3f} | {values.index(max(values))} |')
    lines += ['', 'Снята L2-норма скрытого вектора каждого токена на выходе декодер-блока.',
              'Residual-связи позволяют накапливать изменения скрытых состояний; наблюдаемое',
              'изменение норм не является оценкой качества модели. Отдельные высокие пики',
              'показывают неоднородность активаций, но сами по себе не доказывают механизм attention sink.',
              f"Хуков до/после первого/после повторного прогона: **{acts['hooks_before']} / {acts['hooks_after']} / {acts['hooks_after_repeat']}**. Нормы двух прогонов совпадают: **{acts['repeat_identical']}**.",
              'Хуки удаляются в finally; исходные train/eval-флаги каждого модуля восстанавливаются.', '',
              '## 4. Арифметика LoRA', '',
              'Для Linear с формой веса (out, in): A имеет форму (r, in), B — (out, r).',
              'Число новых параметров: **r × (in + out)** на каждый целевой модуль,',
              'суммируем по всем слоям. Смещения отключены. Конфиг «все линейные»',
              'означает семь типов проекций декодера из params.yaml, без общего lm_head.', '',
              '| Конфиг | Формула | PEFT | Разность | Доля от базовой модели |', '|---|---:|---:|---:|---:|']
    for row in report['lora']:
        lines.append(f"| {row['name']} | {thousands(row['formula'])} | {thousands(row['peft'])} | {row['formula']-row['peft']} | {row['share_of_base']:.4%} |")
    lines += ['', 'Доля PEFT в stdout использует знаменатель «база + адаптер», поэтому',
              'отличается от доли относительно только базовой модели в таблице.', '',
              '## 5. Память в трёх режимах', '',
              '| Режим | Пик, MiB | RSS, MiB | К инференсу | Всего, с | Шаг, с | PID |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('inference','full_ft','lora'):
        x=modes[mode]
        lines.append(f"| {mode} | {x['peak_mb']:.1f} | {x['peak_rss_mb']:.1f} | {x['peak_mb']/inference:.2f} | {x['seconds']:.3f} | {x['step_seconds']:.3f} | {x['pid']} |")
    lines += ['', 'Размеры реально существующих тензоров (градиенты — до zero_grad,',
              'состояния оптимизатора — после первого optimizer.step):', '',
              '| Режим | Веса, MiB | Градиенты, MiB | AdamW, MiB |', '|---|---:|---:|---:|']
    for mode, x in modes.items():
        t=x['tensor_memory']
        lines.append(f"| {mode} | {mib(t['weights']['bytes']):.3f} | {mib(t['gradients']['bytes']):.3f} | {mib(t['optimizer']['bytes']):.3f} |")
    lines += ['', 'Фактические dtype тензоров:', '']
    for mode,x in modes.items():
        t=x['tensor_memory']
        lines.append(f"- {mode}: веса {t['weights']['by_dtype']}; градиенты {t['gradients']['by_dtype']}; AdamW {t['optimizer']['by_dtype']} (значения в байтах).")
    full, lora = modes['full_ft'], modes['lora']
    lines += ['', f"Базовые веса занимают **{mib(report['weights_bytes']):.3f} MiB**. Все три измеренных пика не меньше размера весов.",
              f"Полное обучение добавляет к инференсу **{full['peak_mb']-inference:.1f} MiB**, LoRA — **{lora['peak_mb']-inference:.1f} MiB**.",
              'При full fine-tune градиенты и два момента AdamW создаются для всех параметров.',
              'В LoRA базовые веса заморожены, эти тензоры нужны только адаптерам. PEFT',
              'по умолчанию повышает точность обучаемых адаптеров до float32; это учитывается',
              'по фактическим element_size, а не предположением «везде 2 байта».',
              'Разницу между пиком и суммой перечисленных тензоров нельзя целиком назвать',
              'активациями: туда входят временные буферы, logits, KV-cache, кеш аллокатора',
              'и служебные выделения. Компоненты также могут достигать максимумов в разное время.',
              f"Разброс пиков RSS между режимами: **{max(x['peak_rss_mb'] for x in modes.values())-min(x['peak_rss_mb'] for x in modes.values()):.1f} MiB**. На Apple Silicon RSS не отражает весь объём буферов Metal.", '',
              'Для сравнения: в лекции при seq_len=256, batch=1, MPS приведены пики',
              '2222 / 7166 / 3606 MiB. Наши результаты относятся к приведённым выше версиям',
              'библиотек и условиям, поэтому совпадение чисел с лекцией не ожидается.', '',
              '## 6. Исправленные дефекты', '']
    bad=sum(x['numel'] for x in report['parameter_rows'])
    lines += ['### 1. Общие веса считались повторно', '',
              f"В заготовке каждая строка named_parameters(remove_duplicate=False) имела tied=False. Получалось {thousands(bad)} вместо {thousands(report['params_direct'])}: лишних {thousands(bad-report['params_direct'])} параметров.",
              'Добавлен учёт id параметра. Повторная ссылка видна в таблице, но исключена',
              'из суммы. После исправления таблица и прямой подсчёт совпадают точно.', '',
              '### 2. Хуки оставались на модели', '',
              'register_forward_hook вызывался без сохранения handle и последующего remove.',
              'При трёх различных блоках каждый вызов добавлял три хука: после двух — шесть.',
              f"Контекстный менеджер теперь снимает свои хуки даже при исключении. В реальном разборе после двух вызовов осталось {acts['hooks_after_repeat']} хуков; повторные активации совпадают.", '',
              '### 3. Измерялся остаток памяти вместо пика', '',
              'PeakMemory читал память только в __exit__, после gc.collect. Промежуточные',
              'максимумы терялись. Теперь на MPS идёт фоновый опрос и сохраняется максимум,',
              'на CUDA используется high-water mark. Границы стадий записаны в JSON.',
              'Регрессионный тест задаёт последовательность 100 → 900 → 50 байт:',
              'сохранённый максимум равен 900, а не последнему значению 50.',
              'Каждый режим и повтор перенесён в отдельный процесс, чтобы пики предыдущего',
              'режима не влияли на последующие.', '']
    for mode,x in modes.items():
        stages=x['stage_device_bytes']
        end=list(stages.values())[-1]
        lines.append(f"- {mode}: {x['samples']} выборок; пик {x['peak_mb']:.1f} MiB; последняя граница стадии {mib(end):.1f} MiB.")
    lines += ['', '### 4. На ускорителе использовалась RSS', '',
              'device_allocated_bytes и device_metric_source возвращали результат peak_rss',
              'даже для MPS/CUDA, хотя подпись называла его памятью аллокатора.',
              f"Теперь реально вызывается `{e['memory_metric']}`. Например, full fine-tune: **{full['peak_mb']:.1f} MiB** по метрике устройства против **{full['peak_rss_mb']:.1f} MiB** RSS.",
              'Разделены основной показатель, дополнительная RSS и название источника.', '',
              '## 7. Проверка и сводная таблица', '',
              '`make test` проверяет tied weights, очистку хуков и восстановление режима',
              'при ошибке, источники памяти, сохранение промежуточного пика и отдельные процессы.',
              '`make check` выполняет семь проверок задания, в том числе реальные прогоны модели.',
              'Сводная таблица `anatomy-worksheet.xlsx` заполнена по данным этого отчёта.',
              'При повторном make inspect JSON и Markdown обновляются; значения в Excel нужно',
              'сверить заново, поскольку это отдельная сводная таблица.', '',
              'Источники: задание ДЗ 2 и лекция 2 из выданного архива;',
              '[память драйвера MPS](https://docs.pytorch.org/docs/main/generated/torch.mps.driver_allocated_memory.html);',
              '[точность адаптеров PEFT](https://huggingface.co/docs/peft/developer_guides/troubleshooting).', '']
    return '\n'.join(lines)


def write_report(report: dict, params: dict) -> None:
    plot_activations(report['activations'], params['hooks']['plot'])
    path=Path(params['report']['markdown'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report, params), encoding='utf-8')
