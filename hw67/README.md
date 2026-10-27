# ДЗ 6+7. Эксперименты и проверка качества

Продолжение ДЗ 4–5, тема — вопросы по MLOps на русском.
Исходные тензоры: 1267 train / 111 val. Короткий поиск использует общий
пул train 32, val 8, четыре шага на конфигурацию, batch 1 × accumulation 2.
Шесть комбинаций lr, r и target_modules. Test в выборе не участвует.

Нужны Python 3.12, uv и работающий Docker Desktop. Обучение идёт на ноутбуке,
контейнеры обслуживают MLflow и Postgres. UI: http://127.0.0.1:5006.

```bash
cd hw67
uv sync --frozen
uv run dvc pull
make up
make check-eval
make sweep
make report
make candidate
make baseline-init
make evaluate
make eval
make check-negative
# make promote — только если make eval завершился PASS
make check-storage
make check-all
```

`baseline-init` выполняется один раз. Если gate отказал, пропустите promote:
это допустимый итог задания, champion остаётся прежним или отсутствует.
`make eval` при отказе возвращает ненулевой код намеренно.
Не используйте повторный sweep для записи видео: он создаёт новые runs.

## Данные и доступ

Тензоры скопированы из ДЗ 4 без изменений. Полный training-index получен
из `hw4/data/train.jsonl`; ID сверены с ID всех тензоров в том же порядке.
Код подготовки: `tools/prepare_eval.py`, доказательство: `docs/data-provenance.json`.
Golden содержит 50 сценариев по лекциям 6–7, пять срезов по десять вопросов.
В каждой строке сохранены источник, ожидаемая метка и обоснование разметки.
Промпт модели не содержит expected. Behavioral считаются отдельно от golden.

Файлы `inputs.dvc` и `eval.dvc` фиксируют версии. Локальный DVC
remote — `dvc_remote_hw67` рядом с `mlops-homework`. На другой машине для
тензоров нужно это хранилище либо выход ДЗ 4. Веса находятся в artifact volume
MLflow; базовая модель берётся из Hugging Face с точной commit-ревизией.

[Приложение с набором оценки](https://github.com/hixday28/mlops-homework/releases/tag/hw67-submission)
содержит golden, behavioral, полный training-index и атрибуцию Stack Overflow.
Это отдельный архив без весов и обучающих тензоров. Разметка golden составлена
по лекциям; вопросы Stack Overflow в индексе сохраняют исходную атрибуцию,
изменения описаны в паспорте ДЗ 3. Адаптация — CC BY-SA 4.0.

Автоматический контроль находит точные нормализованные совпадения и общие
группы. Он не доказывает отсутствие всех смысловых пересечений.

## Результаты и видео

Все девять проверок MLflow и 24 теста оценки пройдены. Лучшая конфигурация —
all-r8-high, candidate версии 1. Accuracy: база 37/50, кандидат 32/50;
формат: 49/50 и 45/50. Quality gate отказал; champion не назначен.
Подробности и ограничения — в docs/conclusion.md.

- `docs/comparison.csv`: шесть реальных runs, выгрузка с сервера.
- `docs/defects.md`: четыре дефекта и доказательства исправлений.
- `docs/conclusion.md`: протокол, метрики, пары ответов и решение.
- `reports/`: копии JSON-отчётов, сырые ответы и ссылки на MLflow runs.

Для видео 5–8 минут выполните команды по одной:

```bash
make up
HF_HUB_OFFLINE=1 make eval
HF_HUB_OFFLINE=1 make check-negative
HF_HUB_OFFLINE=1 make check-storage
HF_HUB_OFFLINE=1 make check-all
```

`make eval` ожидаемо завершится отказом: 0.64 < 0.74. Это решение gate,
его не нужно исправлять ослаблением порога. `make promote` здесь не выполняется.

1. MLflow: experiment hw67-mlops, завершённый sweep и шесть дочерних runs.
   Откройте таблицу параметров/val loss и модель mlops-qwen3-adapter: версия 1,
   alias candidate; champion отсутствует.
2. docs/comparison.csv: шесть сопоставимых конфигураций и ресурсы.
3. docs/conclusion.md: accuracy/формат, пять срезов, behavioral и разбор пар.
4. reports/quality-gate.json, negative-gate.json: реальный отказ и отрицательный тест.
5. Терминал: PASS сохранности, итог 9/9, отпечаток c42ac7788026 и OK (24 теста).
6. docs/defects.md: четыре исправления. При необходимости покажите compose.yaml,
   src/sweep.py, src/promote.py и evaluation/solution.py.

Повторные sweep/evaluate для записи не нужны. Сырые ответы и результаты
находятся в reports; состояние сервера сохраняется в volumes.

После работы `make down` останавливает сервисы, сохраняя volumes.
Команда `docker compose down -v` уничтожает хранилища — для обычной остановки
она не нужна.
