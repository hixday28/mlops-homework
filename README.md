# MLOps — домашние задания

## ДЗ 1. Окружение и инференс

Код и конфигурация находятся в [hw1](hw1).
[Отчёт о железе и бенчмарке](hw1/docs/hardware.md).

```bash
cd hw1
make install
make generate
make bench
make check
```

## ДЗ 2. Анатомия модели

Код и конфигурация находятся в [hw2](hw2).
[Отчёт](hw2/docs/anatomy.md), [график активаций](hw2/docs/activations.png),
[сводная таблица Excel](hw2/docs/anatomy-worksheet.xlsx).

```bash
cd hw2
make install
make inspect
make check
```

Команды выполняются из соответствующей папки. Нужны Python 3.12 и uv.
При первом запуске модель скачивается с Hugging Face.
После повторных замеров числа в отчётах и Excel нужно сверить с обновлёнными JSON.

## ДЗ 3. Датасет и DVC

[Код и запуск](hw3/README.md), [паспорт датасета](hw3/docs/datasheet.md),
[разбор дефектов](hw3/docs/defects.md).
Данные не хранятся в Git. Версии отмечены тегами `hw3-v1` и `hw3-v2`.

## ДЗ 4. Токенизация, шаблон чата и маска лосса

[Код и запуск](hw4/README.md), [отчёт токенизации](hw4/docs/tokenize_report.md),
[разбор четырёх дефектов](hw4/docs/defects.md).
Проверка: `cd hw4 && make check`. Данные и тензоры хранятся в DVC.

## ДЗ 5. LoRA-дообучение

[Код и запуск](hw5/README.md), [кривые](hw5/docs/curves.png),
[сравнение базы и адаптера](hw5/docs/compare.md),
[разбор пяти дефектов](hw5/docs/defects.md).
Данные, адаптеры и JSON-метрики хранятся в DVC; проверка — `cd hw5 && make check`.

## ДЗ 6+7. MLflow и проверка качества

[Код и запуск](hw67/README.md), [сравнение](hw67/docs/comparison.csv),
[единый вывод](hw67/docs/conclusion.md), [разбор дефектов](hw67/docs/defects.md).
