# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for data-free continuous memory prompts.

Continuous memory prompts deliberately live outside the tokenizer vocabulary.
The student owns a small trainable bank, while rollout and reference-policy
forwards receive the selected vectors directly as input embeddings.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


MEMORY_BANK_MODULE_NAME = "_verl_continuous_memory_bank"
MEMORY_BANK_STATE_KEY_SUFFIX = f"{MEMORY_BANK_MODULE_NAME}.memory_bank"
MEMORY_BANK_BYTES_KEY = "continuous_memory_bank_bytes"
MEMORY_BANK_SHAPE_KEY = "continuous_memory_bank_shape"


class ContinuousMemoryBank(nn.Module):
    """A trainable bank of vectors with the same width as token embeddings."""

    def __init__(
        self,
        bank_size: int,
        hidden_size: int,
        init_std: float,
        *,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        adversarial: bool,
        gradient_scale: float,
    ) -> None:
        super().__init__()
        if bank_size <= 0:
            raise ValueError(f"bank_size must be positive, got {bank_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        if gradient_scale <= 0:
            raise ValueError(f"gradient_scale must be positive, got {gradient_scale}")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        values = torch.randn(bank_size, hidden_size, generator=generator, dtype=torch.float32)
        values = values.mul_(init_std).to(device=device, dtype=dtype)
        self.memory_bank = nn.Parameter(values)
        self.adversarial = bool(adversarial)
        self.gradient_scale = float(gradient_scale)

        # The model minimizes the OPD loss. Reversing only this parameter's
        # gradient gives the memory prompt the opposite (hard-prompt) objective.
        if self.adversarial:
            self.memory_bank.register_hook(lambda grad: -self.gradient_scale * grad)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.memory_bank[indices]


def attach_continuous_memory_bank(
    model: nn.Module,
    bank_size: int,
    init_std: float,
    *,
    device: torch.device | None = None,
    seed: int,
    adversarial: bool,
    gradient_scale: float,
) -> ContinuousMemoryBank:
    """Attach a continuous memory bank without modifying model vocabulary."""

    if hasattr(model, MEMORY_BANK_MODULE_NAME):
        raise ValueError("A continuous memory bank is already attached to this model")

    input_embeddings = model.get_input_embeddings()
    weight = input_embeddings.weight
    bank = ContinuousMemoryBank(
        bank_size=bank_size,
        hidden_size=weight.shape[1],
        init_std=init_std,
        dtype=weight.dtype,
        device=weight.device if device is None else device,
        seed=seed,
        adversarial=adversarial,
        gradient_scale=gradient_scale,
    )
    model.add_module(MEMORY_BANK_MODULE_NAME, bank)
    return bank


def find_continuous_memory_bank(model: nn.Module) -> ContinuousMemoryBank | None:
    """Find the bank through FSDP/PEFT wrappers without relying on key prefixes."""

    for module in model.modules():
        if isinstance(module, ContinuousMemoryBank):
            return module
    return None


def is_continuous_memory_bank_state_key(name: str) -> bool:
    return name.endswith(MEMORY_BANK_STATE_KEY_SUFFIX)


def serialize_memory_bank(bank: torch.Tensor) -> dict[str, object]:
    """Serialize a bank as bf16 bytes so DataProto meta-info stays compact."""

    bank_cpu = bank.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    raw_bytes = bank_cpu.view(torch.uint8).numpy().tobytes()
    return {
        MEMORY_BANK_BYTES_KEY: raw_bytes,
        MEMORY_BANK_SHAPE_KEY: tuple(bank_cpu.shape),
    }


def deserialize_memory_bank(
    meta_info: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Restore a bank serialized by :func:`serialize_memory_bank`."""

    raw_bytes = meta_info.get(MEMORY_BANK_BYTES_KEY)
    shape = meta_info.get(MEMORY_BANK_SHAPE_KEY)
    if raw_bytes is None and shape is None:
        return None
    if raw_bytes is None or shape is None:
        raise ValueError("Incomplete continuous memory-bank metadata")

    expected_numel = 1
    for dimension in shape:
        expected_numel *= int(dimension)
    expected_bytes = expected_numel * torch.tensor([], dtype=torch.bfloat16).element_size()
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Invalid continuous memory-bank payload: expected {expected_bytes} bytes, got {len(raw_bytes)}"
        )

    byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
    bank = byte_tensor.view(torch.bfloat16).reshape(tuple(int(dim) for dim in shape))
    return bank.to(device=device, dtype=dtype, non_blocking=True)


@contextmanager
def replace_input_embeddings(
    model: nn.Module,
    replacement_values: torch.Tensor | None,
    replacement_mask: torch.Tensor | None,
) -> Iterator[None]:
    """Temporarily replace selected embedding outputs during one model forward."""

    if replacement_values is None or replacement_mask is None:
        yield
        return

    embedding_module = model.get_input_embeddings()

    def replace_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected tensor embedding output, got {type(output)}")
        values = replacement_values.to(device=output.device, dtype=output.dtype)
        mask = replacement_mask.to(device=output.device, dtype=torch.bool).unsqueeze(-1)
        if output.shape != values.shape or output.shape[:-1] != mask.shape[:-1]:
            raise ValueError(
                "Continuous memory prompt shape mismatch: "
                f"embedding={tuple(output.shape)}, values={tuple(values.shape)}, mask={tuple(mask.shape)}"
            )
        return torch.where(mask, values, output)

    hook = embedding_module.register_forward_hook(replace_hook)
    try:
        yield
    finally:
        hook.remove()
