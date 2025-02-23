import gpn.distributions as UD
import torch
import torch.distributions as D
import torch.nn.functional as F
from gpn.utils import to_one_hot


def loss_reduce(loss: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """utility function to reduce raw losses

    Args:
        loss (torch.Tensor): raw loss which should be reduced
        reduction (str, optional): reduction method ('sum' | 'mean' | 'none')

    Returns:
        torch.Tensor: reduced loss
    """

    if reduction == "sum":
        return loss.sum()

    if reduction == "mean":
        return loss.mean()

    if reduction == "none":
        return loss

    raise ValueError(f"{reduction} is not a valid value for reduction")


def uce_loss(
    alpha: torch.Tensor, y: torch.Tensor, reduction: str = "sum"
) -> torch.Tensor:
    """utility function computing uncertainty cross entropy /
    bayesian risk of cross entropy

    Args:
        alpha (torch.Tensor): parameters of Dirichlet distribution
        y (torch.Tensor): ground-truth class labels (not one-hot encoded)
        reduction (str, optional): reduction method ('sum' | 'mean' | 'none').
            Defaults to 'sum'.

    Returns:
        torch.Tensor: loss
    """

    if alpha.dim() == 1:
        alpha = alpha.view(1, -1)

    a_sum = alpha.sum(-1)
    a_true = alpha.gather(-1, y.view(-1, 1)).squeeze(-1)
    uce = a_sum.digamma() - a_true.digamma()
    return loss_reduce(uce, reduction=reduction)


def entropy_reg(
    alpha: torch.Tensor,
    beta_reg: float,
    approximate: bool = False,
    reduction: str = "sum",
) -> torch.Tensor:
    """calculates entopy regularizer

    Args:
        alpha (torch.Tensor): dirichlet-alpha scores
        beta_reg (float): regularization factor
        approximate (bool, optional): flag specifying if the entropy is approximated or not. Defaults to False.
        reduction (str, optional): loss reduction. Defaults to 'sum'.

    Returns:
        torch.Tensor: REG
    """

    if approximate:
        reg = UD.Dirichlet(alpha).entropy()
    else:
        reg = D.Dirichlet(alpha).entropy()

    reg = loss_reduce(reg, reduction=reduction)

    return -beta_reg * reg


def uce_loss_and_reg(
    alpha: torch.Tensor,
    y: torch.Tensor,
    beta_reg: float = 0.001,
    reduction: str = "sum",
) -> torch.Tensor:
    """calculates uncertain cross-entropy and entropy regularization at the same time

    Args:
        alpha (torch.Tensor): dirichlet-alpha scores
        y (torch.Tensor): ground-truth labels
        beta_reg (float): regularization factor
        reduction (str, optional): loss reduction. Defaults to 'sum'.

    Returns:
        torch.Tensor: UCE + REG
    """

    uce = uce_loss(alpha, y, reduction="none")
    reg = entropy_reg(alpha, beta_reg, reduction="none")

    loss = uce + reg
    return loss_reduce(loss, reduction=reduction)


def cross_entropy(
    y_hat: torch.Tensor, y: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """wrapper for cross-entropy loss

    Args:
        y_hat (torch.Tensor): predicted class probabilities
        y (torch.Tensor): ground-truth labels
        reduction (str, optional): loss reduction. Defaults to 'mean'.

    Returns:
        torch.Tensor: CE
    """

    log_soft = torch.log(y_hat)
    return F.nll_loss(log_soft, y, reduction=reduction).cpu().detach()


def bayesian_risk_sosq(
    alpha: torch.Tensor, y: torch.Tensor, reduction: str = "sum"
) -> torch.Tensor:
    """bayesian-risk-loss of sum-of-squares

    Args:
        alpha (torch.Tensor): dirichlet-alpha scores
        y (torch.Tensor): ground-truth labels
        reduction (str, optional): loss reduction. Defaults to 'sum'.

    Returns:
        torch.Tensor: loss
    """

    alpha_0 = alpha.sum(dim=-1, keepdim=True)
    y_pred = alpha / alpha_0
    num_classes = alpha.size(-1)
    y_one_hot = to_one_hot(y, num_classes)
    loss_err = (y_one_hot - y_pred) ** 2
    loss_var = y_pred * (1 - y_pred) / (alpha_0 + 1.0)
    loss = (loss_err + loss_var).sum(-1)

    return loss_reduce(loss, reduction=reduction)


def kl_divergence(alpha, beta):

    S_alpha = torch.sum(alpha, dim=2, keepdim=False)
    S_beta = torch.sum(beta, dim=2, keepdim=False)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=2, keepdim=False)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=2, keepdim=False) - torch.lgamma(S_beta)
    dg0 = torch.digamma(torch.sum(alpha, dim=2, keepdim=True))
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=2, keepdim=False) + lnB + lnB_uni

    return kl


def get_tagger_one_hot(label, N, off_value, on_value, input_mask, gpu):
    label_new = torch.clamp(label, min=0)
    label_mask = torch.unsqueeze(
        torch.where(label > 0, 1, 0), dim=2
    )  # in our setting, the 'O' label is 0

    size = list(label.size())
    size.append(N)
    label = label_new.view(-1)
    ones = torch.sparse.torch.eye(N) * on_value

    ones = ones.index_select(0, label.cpu())
    ones += off_value
    ones = ones.view(*size)
    if gpu:
        ones = ones.cuda()
    ones = ones * label_mask

    return ones
