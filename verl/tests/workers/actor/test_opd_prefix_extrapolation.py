import pytest
import torch

from verl.workers.actor.dp_actor import compute_single_teacher_reverse_kl


@pytest.fixture
def log_probs():
    old = torch.tensor([[4.0, 4.0, 4.0, 4.0]])
    ref = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    base = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    return old, ref, base


def test_zero_tokens_is_standard_opd_and_does_not_require_base(log_probs):
    old, ref, _ = log_probs
    actual = compute_single_teacher_reverse_kl(old, ref, None, lambda_vals=1.25, extrapolation_max_tokens=0)
    torch.testing.assert_close(actual, old - ref)


def test_prefix_extrapolation_only_changes_leading_tokens(log_probs):
    old, ref, base = log_probs
    actual = compute_single_teacher_reverse_kl(old, ref, base, lambda_vals=1.25, extrapolation_max_tokens=2)
    expected = torch.tensor([[3.25, 3.25, 3.0, 3.0]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("token_limit", [-1, 4, 8])
def test_full_or_oversized_limit_matches_full_gopd(log_probs, token_limit):
    old, ref, base = log_probs
    actual = compute_single_teacher_reverse_kl(
        old,
        ref,
        base,
        lambda_vals=1.25,
        extrapolation_max_tokens=token_limit,
    )
    expected = old - base - 1.25 * (ref - base)
    torch.testing.assert_close(actual, expected)


def test_invalid_negative_limit_is_rejected(log_probs):
    old, ref, base = log_probs
    with pytest.raises(ValueError, match="extrapolation_max_tokens"):
        compute_single_teacher_reverse_kl(old, ref, base, lambda_vals=1.25, extrapolation_max_tokens=-2)
