"""Тот же causal LM loss, без логитов словаря на замаскированном промпте.

Проекции словаря считаются блоками с checkpointing: длинный ответ не
создаёт одновременно несколько гигабайт float32-логитов. Входной контекст
и маска внимания остаются полными, сдвиг labels выполняется ровно один раз.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def causal_loss(model, batch: dict, chunk_size: int = 256):
    base = model.get_base_model()
    hidden = base.model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        use_cache=False,
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    mask = targets != -100
    states, targets = hidden[:, :-1][mask], targets[mask]
    if targets.numel() == 0:
        raise ValueError("В батче нет обучаемых токенов")

    def block_loss(x, y):
        return F.cross_entropy(base.lm_head(x).float(), y, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        x, y = states[start:start + chunk_size], targets[start:start + chunk_size]
        if torch.is_grad_enabled():
            total = total + checkpoint(block_loss, x, y, use_reentrant=False)
        else:
            total = total + block_loss(x, y)
    return total / targets.numel()
