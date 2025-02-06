import torch


def normalize_logits(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Logit Normalization for NER

    Args:
        logits (torch.Tensor): Logits (N, L, K)
        tau (float): temperature parameter

    Returns:
        torch.Tensor: normalized logits.
    """
    norm = torch.norm(logits, p=2, dim=-1, keepdim=True)
    logit_norm = torch.div(logits, norm) / tau
    return logit_norm
