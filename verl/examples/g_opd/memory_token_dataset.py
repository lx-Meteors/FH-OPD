"""Data-free prompt dataset for OPD experiments.

Each item is a deterministic sequence sampled from a shared bank of virtual
token IDs. Their embedding rows are ordinary model parameters, so gradients
from the student OPD loss update the memory tokens without reading a training
parquet file.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from verl.utils.memory_tokens import build_memory_token_strings
from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score import default_compute_score


class MemoryTokenDataset(Dataset):
    """Generate fixed virtual-token prompts entirely in memory."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        del data_files, processor
        memory_config = config.get("memory_prompt", {})
        self.num_prompts = int(memory_config.get("num_prompts", config.get("train_batch_size", 1024)))
        self.prompt_length = int(memory_config.get("prompt_length", 32))
        self.token_count = int(memory_config.get("token_count", 256))
        self.seed = int(memory_config.get("seed", 42))
        self.max_prompt_length = int(config.get("max_prompt_length", self.prompt_length))
        token_template = memory_config.get("token_template", "<|memory_token_{index:06d}|>")

        if max_samples > 0:
            self.num_prompts = min(self.num_prompts, max_samples)
        if self.num_prompts < 1:
            raise ValueError("memory_prompt.num_prompts must be positive")
        if self.prompt_length < 1:
            raise ValueError("memory_prompt.prompt_length must be positive")
        if self.token_count < 1:
            raise ValueError("memory_prompt.token_count must be positive")
        if self.prompt_length > self.max_prompt_length:
            raise ValueError(
                f"memory prompt length {self.prompt_length} exceeds data.max_prompt_length "
                f"{self.max_prompt_length}"
            )
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id for memory prompts")

        tokens = build_memory_token_strings(self.token_count, token_template)
        self.memory_token_ids = np.asarray(tokenizer.convert_tokens_to_ids(list(tokens)), dtype=np.int64)
        if len(set(self.memory_token_ids.tolist())) != self.token_count:
            raise ValueError("Memory tokens are missing from the tokenizer or do not have unique IDs")
        if tokenizer.unk_token_id is not None and np.any(self.memory_token_ids == tokenizer.unk_token_id):
            raise ValueError("At least one configured memory token maps to unk_token_id")

        self.pad_token_id = int(tokenizer.pad_token_id)

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, index: int) -> dict:
        # Sampling without replacement gives each prompt more local diversity.
        # If a prompt is longer than the bank, repeated IDs are unavoidable.
        rng = np.random.default_rng(self.seed + int(index))
        replace = self.prompt_length > self.token_count
        prompt_ids = rng.choice(self.memory_token_ids, size=self.prompt_length, replace=replace)

        input_ids = torch.full((self.max_prompt_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((self.max_prompt_length,), dtype=torch.long)
        prompt_tensor = torch.from_numpy(prompt_ids.copy()).long()
        input_ids[-self.prompt_length :] = prompt_tensor
        attention_mask[-self.prompt_length :] = 1
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": prompt_ids.tolist(),
            "data_source": "memory_prompt",
            "reward_model": {"ground_truth": ""},
            "extra_info": {"index": int(index)},
            "index": int(index),
        }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Use a zero placeholder reward for OPD prompts and normal validation rewards."""

    if data_source == "memory_prompt":
        return 0.0
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
