import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from src.loss import causal_loss


class LossTest(unittest.TestCase):
    def test_matches_dense_loss_and_gradients(self):
        torch.manual_seed(42)
        hidden = torch.randn(2, 7, 5, requires_grad=True)
        head = torch.nn.Linear(5, 13, bias=False)
        labels = torch.tensor([[-100, -100, -100, 1, 2, 3, 4], [-100, -100, -100, -100, 5, 6, 7]])
        base = SimpleNamespace(model=lambda **kw: SimpleNamespace(last_hidden_state=hidden), lm_head=head)
        model = SimpleNamespace(get_base_model=lambda: base)
        batch = dict(input_ids=torch.zeros(2, 7, dtype=torch.long), attention_mask=torch.ones(2, 7), labels=labels)
        sparse = causal_loss(model, batch, chunk_size=2)
        sparse.backward()
        grad_h, grad_w = hidden.grad.clone(), head.weight.grad.clone()
        hidden.grad = None
        head.weight.grad = None
        dense = F.cross_entropy(head(hidden[:, :-1]).reshape(-1, 13), labels[:, 1:].reshape(-1), ignore_index=-100)
        dense.backward()
        torch.testing.assert_close(sparse, dense)
        torch.testing.assert_close(grad_h, hidden.grad)
        torch.testing.assert_close(grad_w, head.weight.grad)


if __name__ == '__main__':
    unittest.main()
