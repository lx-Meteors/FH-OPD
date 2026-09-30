from types import SimpleNamespace

import torch
from torch import nn

from verl.utils.memory_tokens import add_memory_tokens, resize_model_for_memory_tokens


class FakeTokenizer:
    def __init__(self, size=5):
        self.vocab = {f"token_{index}": index for index in range(size)}

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, special_tokens, replace_additional_special_tokens=False):
        del replace_additional_special_tokens
        for token in special_tokens["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab[tokens]
        return [self.vocab[token] for token in tokens]


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size=5, hidden_size=4):
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size, initializer_range=0.02)

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def resize_token_embeddings(self, new_size, mean_resizing=False):
        del mean_resizing
        old_input = self.input_embeddings
        old_output = self.output_embeddings
        self.input_embeddings = nn.Embedding(new_size, old_input.embedding_dim)
        self.output_embeddings = nn.Linear(old_output.in_features, new_size, bias=False)
        with torch.no_grad():
            copy_size = min(old_input.num_embeddings, new_size)
            self.input_embeddings.weight[:copy_size].copy_(old_input.weight[:copy_size])
            self.output_embeddings.weight[:copy_size].copy_(old_output.weight[:copy_size])
        return self.input_embeddings


def test_add_and_resize_memory_tokens_as_input_only_parameters():
    tokenizer = FakeTokenizer(size=5)
    info = add_memory_tokens(tokenizer, count=3)

    assert info.original_vocab_size == 5
    assert info.token_ids == (5, 6, 7)
    assert len(tokenizer) == 8

    model = FakeCausalLM(vocab_size=5)
    resize_model_for_memory_tokens(model, tokenizer, info)

    assert model.get_input_embeddings().weight.shape == (8, 4)
    assert model.get_output_embeddings().weight.shape == (8, 4)
    assert torch.count_nonzero(model.get_input_embeddings().weight[5:]) > 0
    assert torch.count_nonzero(model.get_output_embeddings().weight[5:]) == 0
    assert model.config.memory_token_output_vocab_size == 5
    assert model.config.memory_token_count == 3


def test_add_memory_tokens_is_idempotent_for_saved_tokenizer():
    tokenizer = FakeTokenizer(size=5)
    first = add_memory_tokens(tokenizer, count=2)
    second = add_memory_tokens(tokenizer, count=2)

    assert first == second
    assert len(tokenizer) == 7
