"""Проверки без загрузки весов: границы замера, медиана, RSS, конфиг."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
import torch

from src import bench
from src.config import load_params
from src.model import generate_tokens, resolve_device


class PipelineTests(unittest.TestCase):
    def test_benchmark_excludes_loading_and_warmup(self):
        params = load_params()
        params['bench'].update(warmup_runs=1, measure_runs=3)
        fake_model = SimpleNamespace(device=torch.device('cpu'), config=SimpleNamespace())
        events = []
        def generate(*args):
            events.append('generate')
            return list(range(20))
        def clock():
            events.append('clock')
            return next(ticks)
        ticks = iter([0, 100, 200, 202, 300, 301, 400, 404])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(bench, 'ROOT', Path(directory)), \
             patch.object(bench, 'load_params', return_value=params), \
             patch.object(bench, 'load_model', return_value=(None, fake_model)), \
             patch.object(bench, 'prepare_inputs', return_value={}), \
             patch.object(bench, 'generate_tokens', side_effect=generate), \
             patch.object(bench.time, 'perf_counter', side_effect=clock), \
             patch('builtins.print'):
            bench.main()
            report = json.loads((Path(directory) / 'docs/bench.json').read_text())
        self.assertEqual(report['load_time_sec'], 100)
        self.assertEqual(report['tokens_per_sec'], 10)
        self.assertEqual(report['tokens_per_sec_all'], [10, 20, 5])
        self.assertEqual(events[:3], ['clock', 'clock', 'generate'])
        self.assertEqual(events.count('generate'), 4)

    def test_peak_rss_units(self):
        for system, value in [('darwin', 1048576), ('linux', 1024)]:
            with self.subTest(system=system), patch.object(bench.sys, 'platform', system), \
                 patch.object(bench.resource, 'getrusage', return_value=SimpleNamespace(ru_maxrss=value)):
                self.assertEqual(bench.peak_rss_mb(), 1)

    def test_invalid_config_fails_before_model_loading(self):
        valid = load_params()
        for group, key, value in [('bench', 'warmup_runs', 0), ('bench', 'measure_runs', 0),
                                  ('generate', 'temperature', -1), ('generate', 'max_new_tokens', 0)]:
            cfg = copy.deepcopy(valid)
            cfg[group][key] = value
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'params.yaml'
                path.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    load_params(path)

    def test_auto_prefers_mps_when_cuda_unavailable(self):
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(resolve_device('auto'), 'mps')

    def test_greedy_output_excludes_prompt(self):
        from unittest.mock import Mock
        model = Mock()
        model.generation_config.pad_token_id = None
        model.generation_config.eos_token_id = 9
        model.generate.return_value = torch.tensor([[1, 2, 3, 8, 9]])
        cfg = load_params()
        cfg['generate']['temperature'] = 0
        tokens = generate_tokens(model, {'input_ids': torch.tensor([[1, 2, 3]])}, cfg)
        self.assertEqual(tokens.tolist(), [8, 9])
        self.assertFalse(model.generate.call_args.kwargs['do_sample'])
        self.assertGreater(model.generate.call_args.kwargs['temperature'], 0)


if __name__ == '__main__':
    unittest.main()
