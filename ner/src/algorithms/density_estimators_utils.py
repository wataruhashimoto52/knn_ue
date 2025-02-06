from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy.random import permutation, randint
from torch import Tensor
from torch.nn import ReLU, Tanh
from torch.nn import functional as F

# This implementation of MADE is copied from: https://github.com/e-hulten/made.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(n_in, n_out, bias)
        self.mask = None

    def initialise_mask(self, mask: Tensor):
        """Internal method to initialise mask."""
        self.mask = mask.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(
        self,
        n_in: int,
        hidden_dims: List[int],
        gaussian: bool = False,
        random_order: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """Initalise MADE model.

        Args:
            n_in: Size of input.
            hidden_dims: List with sizes of the hidden layers.
            gaussian: Whether to use Gaussian MADE. Default: False.
            random_order: Whether to use random order. Default: False.
            seed: Random seed for numpy. Default: None.
        """
        super().__init__()
        # Set random seed.
        np.random.seed(seed)
        self.n_in = n_in
        self.n_out = 2 * n_in if gaussian else n_in
        self.hidden_dims = hidden_dims
        self.random_order = random_order
        self.gaussian = gaussian
        self.masks = {}
        self.mask_matrix = []
        self.layers = []

        # List of layers sizes.
        dim_list = [self.n_in, *hidden_dims, self.n_out]
        # Make layers and activation functions.
        for i in range(len(dim_list) - 2):
            self.layers.append(
                MaskedLinear(dim_list[i], dim_list[i + 1]),
            )
            self.layers.append(ReLU())
        # Hidden layer to output layer.
        self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        # Create model.
        self.model = nn.Sequential(*self.layers)
        # Get masks for the masked activations.
        self._create_masks()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.gaussian:
            # If the output is Gaussan, return raw mus and sigmas.
            return self.model(x)
        else:
            # If the output is Bernoulli, run it trough sigmoid to squash p into (0,1).
            return torch.sigmoid(self.model(x))

    def _create_masks(self) -> None:
        """Create masks for the hidden layers."""
        # Define some constants for brevity.
        L = len(self.hidden_dims)
        D = self.n_in

        # Whether to use random or natural ordering of the inputs.
        self.masks[0] = permutation(D) if self.random_order else np.arange(D)

        # Set the connectivity number m for the hidden layers.
        # m ~ DiscreteUniform[min_{prev_layer}(m), D-1]
        for l in range(L):
            low = self.masks[l].min()
            size = self.hidden_dims[l]
            self.masks[l + 1] = randint(low=low, high=D - 1, size=size)

        # Add m for output layer. Output order same as input order.
        self.masks[L + 1] = self.masks[0]

        # Create mask matrix for input -> hidden_1 -> ... -> hidden_L.
        for i in range(len(self.masks) - 1):
            m = self.masks[i]
            m_next = self.masks[i + 1]
            # Initialise mask matrix.
            M = torch.zeros(len(m_next), len(m)).to(device)
            for j in range(len(m_next)):
                # Use broadcasting to compare m_next[j] to each element in m.
                M[j, :] = torch.from_numpy((m_next[j] >= m).astype(int)).to(device)
            # Append to mask matrix list.
            self.mask_matrix.append(M)

        # If the output is Gaussian, double the number of output units (mu,sigma).
        # Pairwise identical masks.
        if self.gaussian:
            m = self.mask_matrix.pop(-1)
            self.mask_matrix.append(torch.cat((m, m), dim=0).to(device))

        # Initalise the MaskedLinear layers with weights.
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(next(mask_iter))


class BatchNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x: torch.Tensor):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean.clone()
            v = self.batch_var.clone()

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v)).to(x.device)
        return x_hat, log_det

    def backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean
            v = self.batch_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det

    def set_batch_stats_func(self, x):
        print("setting batch stats for validation")
        self.batch_mean = x.mean(dim=0)
        self.batch_var = x.var(dim=0) + self.eps


class BatchNorm_running(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.momentum = 0.01
        self.gamma = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.running_mean = torch.zeros(1, dim).to(device)
        self.running_var = torch.ones(1, dim).to(device)

    def forward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * m
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * v
        else:
            m = self.running_mean
            v = self.running_var

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma) - 0.5 * torch.sum(torch.log(v))
        return x_hat, log_det

    def backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.running_mean *= 1 - self.momentum
            self.running_mean += self.momentum * m
            self.running_var *= 1 - self.momentum
            self.running_var += self.momentum * v
        else:
            m = self.running_mean
            v = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det


class MAFLayer(nn.Module):
    def __init__(self, dim: int, hidden_dims: List[int], reverse: bool):
        super(MAFLayer, self).__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, gaussian=True, seed=None)
        self.reverse = reverse

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.made(x.float())
        mu, logp = torch.chunk(out, 2, dim=1)
        u = (x - mu) * torch.exp(0.5 * logp)
        u = u.flip(dims=(1,)) if self.reverse else u
        log_det = 0.5 * torch.sum(logp, dim=1)
        return u, log_det

    def backward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u).to(device)
        for dim in range(self.dim):
            out = self.made(x)
            mu, logp = torch.chunk(out, 2, dim=1)
            mod_logp = torch.clamp(-0.5 * logp, max=10)
            x[:, dim] = mu[:, dim] + u[:, dim] * torch.exp(mod_logp[:, dim])
        log_det = torch.sum(mod_logp, axis=1)
        return x, log_det
