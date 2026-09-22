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
