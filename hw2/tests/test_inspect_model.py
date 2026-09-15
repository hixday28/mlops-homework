"""Регрессии для ошибок подсчёта, хуков и измерения памяти; без загрузки весов."""

import json
import types
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src import inspect_model as im


class ParameterTests(unittest.TestCase):
    def test_shared_parameter_is_counted_once_but_alias_is_visible(self):
        model = torch.nn.Module()
        model.embed_tokens = torch.nn.Embedding(5, 3)
        model.lm_head = torch.nn.Linear(3, 5, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
        # Равные значения разных параметров не означают общие веса.
        model.q_proj = torch.nn.Linear(3, 5, bias=False)
        model.q_proj.weight.data.copy_(model.embed_tokens.weight)
        rows = im.parameter_rows(model)
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(r['numel'] for r in rows if not r['tied']), 30)
        table = {r['group']: r for r in im.group_table(rows)}
        self.assertEqual(table['lm_head']['params'], 0)
        self.assertEqual(table['lm_head']['tied_params'], 15)
        self.assertEqual(sum(r['share'] for r in table.values()), 1)

    def test_lora_formula_handles_rectangular_projections(self):
        model = torch.nn.Module()
        model.q_proj = torch.nn.Linear(3, 7, bias=False)
        model.v_proj = torch.nn.Linear(3, 2, bias=False)
        model.lm_head = torch.nn.Linear(3, 11, bias=False)
        self.assertEqual(im.lora_params_formula(model, 4, ['q_proj','v_proj']), 60)

    def test_peft_comparison_restores_model_and_grad_flags(self):
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, tie_word_embeddings=True))
        next(model.parameters()).requires_grad_(False)
        before = [(id(p), p.requires_grad) for p in model.parameters()]
        configs = [{'name':'q/v', 'r':2, 'target_modules':['q_proj','v_proj']},
                   {'name':'all', 'r':4, 'target_modules':['q_proj','k_proj','v_proj',
                                                        'o_proj','gate_proj','up_proj','down_proj']}]
        result = im.lora_report(model, {'lora': {'alpha_ratio':2,'dropout':0.,'configs':configs}})
        self.assertTrue(all(r['match'] for r in result))
        self.assertEqual(before, [(id(p),p.requires_grad) for p in model.parameters()])
        self.assertFalse(hasattr(model, 'peft_config'))
        self.assertFalse(any('lora_' in name for name,_ in model.named_parameters()))


class HookTests(unittest.TestCase):
    def test_cleanup_after_success_and_error_preserves_foreign_hooks(self):
        layer = torch.nn.Identity()
        foreign = layer.register_forward_hook(lambda *_: None)
        for should_fail in (False, True):
            try:
                with im.forward_hooks({'first':layer}) as norms:
                    layer(torch.tensor([[[3.,4.],[0.,2.]]]))
                    self.assertEqual(norms['first'], [5.,2.])
                    if should_fail:
                        raise ValueError('forward failed')
            except ValueError:
                pass
            self.assertEqual(list(layer._forward_hooks), [foreign.id])
        foreign.remove()

    def test_partial_registration_failure_does_not_leak(self):
        good = torch.nn.Identity()
        bad = types.SimpleNamespace(register_forward_hook=lambda _: (_ for _ in ()).throw(ValueError()))
        with self.assertRaises(ValueError):
            with im.forward_hooks({'good':good,'bad':bad}):
                pass
        self.assertEqual(len(good._forward_hooks), 0)

    def test_activation_error_restores_mixed_training_state(self):
        class FailingModel(torch.nn.Module):
            device = torch.device('cpu')
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
            def get_decoder(self):
                return self
            def forward(self, **kwargs):
                self.layers[0](torch.ones(1,2,3))
                raise RuntimeError('model failure')
        class Inputs(dict):
            def to(self, device):
                return self
        model = FailingModel()
        model.layers[1].eval()
        flags = [m.training for m in model.modules()]
        tokenizer = lambda *a, **kw: Inputs(input_ids=torch.ones(1,2,dtype=torch.long))
        with patch.object(im, 'build_prompt', return_value='prompt'):
            with self.assertRaisesRegex(RuntimeError, 'model failure'):
                im.activation_norms(tokenizer, model, {'hooks':{'prompt':'x'}})
        self.assertEqual([m.training for m in model.modules()], flags)
        self.assertEqual(sum(len(m._forward_hooks) for m in model.modules()), 0)


class MemoryTests(unittest.TestCase):
    def test_gpu_metrics_call_device_api_not_rss(self):
        with patch.object(torch.mps,'driver_allocated_memory',return_value=123) as mps, \
             patch.object(torch.cuda,'max_memory_allocated',return_value=456) as cuda, \
             patch.object(im,'peak_rss',side_effect=AssertionError('RSS on GPU')):
            for dev, expected, source in [('mps',123,'torch.mps.driver_allocated_memory'),
                                           ('cuda',456,'torch.cuda.max_memory_allocated')]:
                self.assertEqual(im.device_allocated_bytes(torch.device(dev)), expected)
                self.assertEqual(im.device_metric_source(torch.device(dev)), source)
            mps.assert_called_once()
            cuda.assert_called_once()

    def test_intermediate_peak_survives_final_drop_and_thread_stops(self):
        with patch.object(im,'synchronize'), \
             patch.object(im,'device_allocated_bytes',side_effect=[100,900,50]), \
             patch.object(im,'peak_rss',return_value=(1000,'ru_maxrss')):
            peak = im.PeakMemory(torch.device('mps'), interval=60)
            with peak:
                peak.stage('transient')
            self.assertEqual(peak.used, 900)
            self.assertEqual(peak.samples, 3)
            self.assertFalse(peak.thread.is_alive())

    def test_sampler_stops_on_model_exception(self):
        with patch.object(im,'synchronize'), patch.object(im,'device_allocated_bytes',return_value=100):
            peak = im.PeakMemory(torch.device('mps'), interval=60)
            with self.assertRaisesRegex(ValueError, 'step'):
                with peak:
                    raise ValueError('step')
            self.assertFalse(peak.thread.is_alive())

    def test_cuda_peak_reset_before_measurement(self):
        with patch.object(im,'synchronize'), patch.object(im,'device_allocated_bytes',return_value=100), \
             patch.object(torch.cuda,'reset_peak_memory_stats') as reset:
            with im.PeakMemory(torch.device('cuda')):
                reset.assert_called_once_with(torch.device('cuda'))

    def test_rss_units_and_windows_fallback(self):
        resource = types.SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda _:types.SimpleNamespace(ru_maxrss=17))
        with patch.object(im,'resource',resource):
            for platform, expected in [('darwin',17),('linux',17*1024)]:
                with patch.object(im.sys,'platform',platform):
                    self.assertEqual(im.peak_rss(), (expected,'ru_maxrss'))
        psutil = types.SimpleNamespace(Process=lambda:types.SimpleNamespace(
            memory_info=lambda:types.SimpleNamespace(peak_wset=42)))
        with patch.object(im,'resource',None), patch.object(im,'psutil',psutil):
            self.assertEqual(im.peak_rss(), (42,'peak_wset'))

    def test_subprocess_receives_config_without_rewriting_yaml(self):
        params = {'memory':{'seq_len':256}}
        result = types.SimpleNamespace(returncode=0, stdout='notice\n{"mode":"lora","peak_mb":123}',stderr='')
        with patch.object(im.subprocess,'run',return_value=result) as run:
            self.assertEqual(im.probe_mode('lora',params)['peak_mb'], 123)
        args, kwargs = run.call_args
        self.assertIn('--params-stdin', args[0])
        self.assertEqual(json.loads(kwargs['input']),params)
        self.assertTrue(kwargs['capture_output'])
        result.returncode = 1
        result.stderr = 'out of memory'
        with patch.object(im.subprocess,'run',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'out of memory'):
                im.probe_mode('lora',params)

    def test_each_mode_and_repeat_is_a_new_probe(self):
        calls = []
        def probe(mode, params):
            calls.append(mode)
            return {'mode':mode,'peak_mb':len(calls),'pid':len(calls)}
        with patch.object(im,'probe_mode',side_effect=probe):
            results = im.memory_profile({'memory':{'repeats':2}})
        self.assertEqual(calls, ['inference','inference','full_ft','full_ft','lora','lora'])
        self.assertEqual([r['peak_mb'] for r in results], [2,4,6])
        self.assertEqual(len(set(pid for r in results for pid in r['pids'])), 6)


if __name__ == '__main__':
    unittest.main()
