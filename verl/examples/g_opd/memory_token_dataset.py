"""Data-free continuous-prompt dataset for OPD experiments.

Each item contains indices into an actor-owned ``nn.Parameter`` memory bank.
The ordinary input IDs are inert placeholders: actor/ref forwards replace
their embeddings, and vLLM receives the selected vectors via prompt_embeds.
No tokenizer or vocabulary resize is involved.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from verl.utils.model import compute_position_id_with_mask
from verl.utils.reward_score import default_compute_score


class ContinuousMemoryPromptDataset(Dataset):
    """Generate deterministic index sequences into a continuous memory bank."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        del data_files, processor
        memory_config = config.get("memory_prompt", {})
        self.num_prompts = int(memory_config.get("num_prompts", config.get("train_batch_size", 1024)))
        self.prompt_length = int(memory_config.get("prompt_length", 32))
        self.bank_size = int(memory_config.get("bank_size", memory_config.get("token_count", 256)))
        self.seed = int(memory_config.get("seed", 42))

        if max_samples > 0:
            self.num_prompts = min(self.num_prompts, max_samples)
        if self.num_prompts < 1:
            raise ValueError("memory_prompt.num_prompts must be positive")
        if self.prompt_length < 1:
            raise ValueError("memory_prompt.prompt_length must be positive")
        if self.bank_size < 1:
            raise ValueError("memory_prompt.bank_size must be positive")
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define pad_token_id for memory prompts")

        self.pad_token_id = int(tokenizer.pad_token_id)

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, index: int) -> dict:
        # Sampling without replacement gives each prompt more local diversity.
        # If a prompt is longer than the bank, repeated vectors are unavoidable.
        rng = np.random.default_rng(self.seed + int(index))
        replace = self.prompt_length > self.bank_size
        memory_indices = rng.choice(self.bank_size, size=self.prompt_length, replace=replace)

        # Placeholder IDs keep every downstream tensor shape/token-label path
        # valid, but their ordinary embeddings are never used for this prompt.
        input_ids = torch.full((self.prompt_length,), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.ones((self.prompt_length,), dtype=torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "memory_prompt_indices": torch.from_numpy(memory_indices.copy()).long(),
            "raw_prompt_ids": input_ids.tolist(),
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


# Keep old Hydra overrides working while changing their semantics to the
# continuous implementation.
MemoryTokenDataset = ContinuousMemoryPromptDataset
