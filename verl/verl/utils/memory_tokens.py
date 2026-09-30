# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Utilities for data-free prompts backed by trainable token embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_MEMORY_TOKEN_TEMPLATE = "<|memory_token_{index:06d}|>"


@dataclass(frozen=True)
class MemoryTokenInfo:
    """Description of a tokenizer's appended memory-token range."""

    original_vocab_size: int
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]

    @property
    def expanded_vocab_size(self) -> int:
        return self.original_vocab_size + len(self.token_ids)


def build_memory_token_strings(
    count: int, token_template: str = DEFAULT_MEMORY_TOKEN_TEMPLATE
) -> tuple[str, ...]:
    if count < 0:
        raise ValueError(f"memory_token_count must be non-negative, got {count}")
    if count and "{index" not in token_template:
        raise ValueError("memory_token_template must contain an {index} format field")
    return tuple(token_template.format(index=index) for index in range(count))


def add_memory_tokens(
    tokenizer,
    count: int,
    token_template: str = DEFAULT_MEMORY_TOKEN_TEMPLATE,
) -> MemoryTokenInfo | None:
    """Append deterministic special tokens and return their contiguous ID range.

    The strict contiguity check is intentional: actor, reference model, dataset,
    and rollout engine exchange raw token IDs, so silently reusing a token that
    already existed in only one tokenizer would condition them on different
    prompts.
    """

    if count == 0:
        return None

    tokens = build_memory_token_strings(count, token_template)
    existing_vocab = tokenizer.get_vocab()
    already_present = tuple(token in existing_vocab for token in tokens)
    if any(already_present):
        if not all(already_present):
            raise ValueError("Only part of the configured memory-token range already exists")
        token_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in tokens)
        original_vocab_size = token_ids[0]
        expected_ids = tuple(range(original_vocab_size, original_vocab_size + count))
        if token_ids != expected_ids or len(tokenizer) != original_vocab_size + count:
            raise ValueError("Existing memory tokens are not the tokenizer's final contiguous ID range")
        return MemoryTokenInfo(
            original_vocab_size=original_vocab_size,
            token_ids=token_ids,
            tokens=tokens,
        )

    original_vocab_size = len(tokenizer)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(tokens)},
        replace_additional_special_tokens=False,
    )
    token_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in tokens)
    expected_ids = tuple(range(original_vocab_size, original_vocab_size + count))
    if token_ids != expected_ids:
        raise ValueError(
            "Memory tokens must be newly appended as one contiguous range. "
            f"Expected IDs [{expected_ids[0]}, {expected_ids[-1]}], "
            f"got [{min(token_ids)}, {max(token_ids)}]. Choose a different memory_token_template."
        )
    if len(tokenizer) != original_vocab_size + count:
        raise ValueError(
            f"Tokenizer size mismatch after adding memory tokens: expected "
            f"{original_vocab_size + count}, got {len(tokenizer)}"
        )
    return MemoryTokenInfo(
        original_vocab_size=original_vocab_size,
        token_ids=token_ids,
        tokens=tokens,
    )


def resize_model_for_memory_tokens(
    model,
    tokenizer,
    info: MemoryTokenInfo | None,
    token_template: str = DEFAULT_MEMORY_TOKEN_TEMPLATE,
) -> None:
    """Resize a causal LM and mark the normal output-vocabulary boundary.

    New input rows are initialized from existing embedding rows with a small
    perturbation. Output logits are capped elsewhere at ``original_vocab_size``;
    memory tokens are prompt-only parameters and are never valid responses.
    """

    if info is None:
        return

    memory_rows_already_initialized = (
        getattr(model.config, "memory_token_output_vocab_size", None) == info.original_vocab_size
        and model.get_input_embeddings().num_embeddings == len(tokenizer)
    )
    try:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except TypeError:
        # Compatibility with older Transformers releases.
        model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight
    # Non-root FSDP ranks may construct the model on the meta device. Rank 0
    # initializes real rows and sync_module_states broadcasts them later.
    if not input_embeddings.is_meta and not memory_rows_already_initialized:
        with torch.no_grad():
            new_rows = input_embeddings[info.original_vocab_size :]
            source_indices = torch.randint(
                low=0,
                high=info.original_vocab_size,
                size=(new_rows.size(0),),
                device=input_embeddings.device,
            )
            new_rows.copy_(input_embeddings[source_indices])
            initializer_range = float(getattr(model.config, "initializer_range", 0.02))
            new_rows.add_(torch.randn_like(new_rows) * initializer_range * 0.01)

            output_layer = model.get_output_embeddings()
            if output_layer is not None and output_layer.weight.data_ptr() != input_embeddings.data_ptr():
                output_layer.weight[info.original_vocab_size :].zero_()

    model.config.vocab_size = len(tokenizer)
    model.config.memory_token_output_vocab_size = info.original_vocab_size
    model.config.memory_token_count = len(info.token_ids)
    model.config.memory_token_template = token_template
