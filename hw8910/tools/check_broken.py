"""Красный локальный CI в временной копии; рабочий код остаётся неизменным."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='hw8910-broken-check-') as tmp:
    target=Path(tmp);shutil.copy2(ROOT/'release.py',target/'release.py');(target/'tests').mkdir()
    shutil.copy2(ROOT/'tests/test_release.py',target/'tests/test_release.py')
    p=target/'release.py';text=p.read_text();begin=text.index('def cache_key(');end=text.index('\n\ndef validate_release',begin)
    text=text[:begin]+'''def cache_key(data_path, config):
    return hashlib.sha256(str(data_path).encode()).hexdigest()
'''+text[end:];p.write_text(text)
    print('INTENTIONAL MUTATION: cache key based only on path; temporary copy',flush=True)
    r=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=target)
    print('Mutated tests exit:',r.returncode,flush=True)
    sys.exit(r.returncode)
