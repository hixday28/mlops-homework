"""Загрузка модели и сборка промпта. Общий код для генерации и замеров."""

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """Зафиксировать источники случайности, чтобы прогон воспроизводился."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS недоступен на этой машине")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна на этой машине")
    return requested


def synchronize(model) -> None:
    """Дождаться завершения асинхронных операций перед чтением таймера."""
    if model.device.type == "mps":
        torch.mps.synchronize()
    elif model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


def load_model(params: dict):
    """Загрузить токенизатор и модель по имени из конфига."""
    set_seed(params["generate"]["seed"])
    name = params["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=getattr(torch, params["model"]["dtype"]),
        device_map=resolve_device(params["model"]["device"]),
    )
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, params: dict, text: str) -> str:
    """Собрать промпт шаблоном модели.

    Никогда не склеивайте роли вручную: у каждой модели свой формат,
    а расхождение шаблонов обучения и инференса — самая частая тихая ошибка.
    """
    messages = [{"role": "user", "content": text}]
    kwargs = {}
    # Параметр есть только у моделей с режимом рассуждений (Qwen3);
    # остальные шаблоны его молча проигнорируют.
    if params["generate"].get("enable_thinking") is not None:
        kwargs["enable_thinking"] = params["generate"]["enable_thinking"]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def prepare_inputs(tokenizer, model, params: dict, text: str):
    prompt = build_prompt(tokenizer, params, text)
    # Шаблон уже добавил специальные токены.
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)


def generate_tokens(model, inputs, params: dict):
    """Общий путь генерации; токенизация и декодирование вне таймера bench."""
    temperature = params["generate"]["temperature"]
    kwargs = {"do_sample": temperature > 0}
    if temperature > 0:
        kwargs["temperature"] = temperature
    else:
        # Сброс sampling-настроек, поставляемых с моделью.
        kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=params["generate"]["max_new_tokens"],
            pad_token_id=model.generation_config.pad_token_id
            if model.generation_config.pad_token_id is not None
            else model.generation_config.eos_token_id,
            **kwargs,
        )
    return output[0][inputs["input_ids"].shape[1]:]


def generate(tokenizer, model, params: dict, text: str) -> tuple[str, int]:
    """Вернуть только ответ и фактическое число новых токенов (включая EOS)."""
    inputs = prepare_inputs(tokenizer, model, params, text)
    new_tokens = generate_tokens(model, inputs, params)
    return tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens)
