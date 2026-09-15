#!/usr/bin/env bash
# Самопроверка домашней работы 1.
# Зелёный check.sh необходим для сдачи, но не достаточен: код читается глазами.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp_dir=$(mktemp -d)
cp params.yaml "$tmp_dir/params.yaml"
restore() { cp "$tmp_dir/params.yaml" params.yaml; rm -rf "$tmp_dir"; }
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fails=0
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; fails=$((fails+1)); }

echo
echo "1. Имя модели не захардкожено в коде"
if grep -rqE '"(Qwen|HuggingFaceTB|meta-llama)/' src/ 2>/dev/null; then
  fail "в src/ найдено имя модели строкой — оно должно жить только в params.yaml"
  grep -rnE '"(Qwen|HuggingFaceTB|meta-llama)/' src/ | sed 's/^/      /'
else
  ok "в src/ имён моделей нет"
fi

echo
echo "2. Смена модели в params.yaml не требует правок в коде"
uv run --locked python - <<'PY'
from pathlib import Path
import yaml
p = Path('params.yaml')
cfg = yaml.safe_load(p.read_text())
cfg['model']['name'] = 'HuggingFaceTB/SmolLM2-135M-Instruct'
p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
PY
# Мало убедиться, что не упало: проверяем, что загрузилась именно та модель,
# которая указана в конфиге. Иначе захардкоженное имя проходит незамеченным.
if uv run --locked python -c "
from src.config import load_params
from src.model import load_model, generate
p = load_params()
_, m = load_model(p)
got = str(getattr(m.config, 'name_or_path', ''))
assert p['model']['name'] in got, f'конфиг просит {p[\"model\"][\"name\"]}, а загружено {got}'
text, n = generate(_, m, p, p['bench']['prompt'])
assert text.strip() and 0 < n <= p['generate']['max_new_tokens']
" > "$tmp_dir/swap.log" 2>&1; then
  ok "загружается именно та модель, что указана в params.yaml"
else
  fail "загружена не та модель — имя берётся не из конфига"
  cat "$tmp_dir/swap.log"
fi
cp "$tmp_dir/params.yaml" params.yaml

echo
echo "3. Генерация воспроизводима при temperature = 0"
uv run --locked python -c "from src.config import load_params; assert load_params()['generate']['temperature'] == 0"
if make generate > "$tmp_dir/out1.txt" 2> "$tmp_dir/run1.log" &&
   make generate > "$tmp_dir/out2.txt" 2> "$tmp_dir/run2.log" &&
   test -s "$tmp_dir/out1.txt" &&
   diff -q "$tmp_dir/out1.txt" "$tmp_dir/out2.txt" > /dev/null 2>&1; then
  ok "два прогона дали идентичный вывод"
else
  fail "прогоны отличаются — не зафиксирован seed либо включён сэмплинг"
  cat "$tmp_dir"/run*.log
  diff "$tmp_dir/out1.txt" "$tmp_dir/out2.txt" || true
fi

echo
echo "4. Зависимости зафиксированы"
if [ -f uv.lock ]; then
  uv lock --check
  ok "uv.lock на месте"
else
  fail "нет uv.lock — выполните uv sync и закоммитьте файл"
fi

if grep -qE '^ *"(torch|transformers|accelerate|numpy|pyyaml)" *,' pyproject.toml 2>/dev/null; then
  fail "зависимости без версий в pyproject.toml"
else
  ok "версии зависимостей указаны"
fi

echo
echo "5. Замеры разделены и есть прогрев"
if grep -q "warmup" src/bench.py && grep -q "load_time" src/bench.py; then
  ok "прогрев и отдельный замер загрузки присутствуют"
else
  fail "в bench.py нет прогрева либо загрузка не выделена в отдельный замер"
fi

echo
echo "6. Проверки логики бенчмарка и конфигурации"
if make unit; then
  ok "модульные проверки пройдены"
else
  fail "модульные проверки не прошли"
fi

echo
if [ "$fails" -eq 0 ]; then
  printf '\033[32mВсе проверки пройдены.\033[0m Не забудьте docs/hardware.md с разбором дефектов.\n\n'
else
  printf '\033[31mПровалено проверок: %s\033[0m\n\n' "$fails"
  exit 1
fi
