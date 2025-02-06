import math
from typing import List, Protocol, Tuple

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
from einops import rearrange
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial

from .density_estimators_utils import BatchNormLayer, MAFLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IDensityEstimator(Protocol):
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class DensityEstimatorWrapper:
    def __init__(self, density_estimator: IDensityEstimator) -> None:
        self.density_estimator = density_estimator

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.density_estimator.log_prob(x)


class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior):
        super(RealNVP, self).__init__()

        self.prior: tdist.Distribution = prior
        self.masks = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z: torch.Tensor):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.masks[i]
            s = self.s[i](x_) * (1 - self.masks[i])
            t = self.t[i](x_) * (1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x: torch.Tensor):
        log_det_J, z = x.new_zeros(x.shape[0], device=x.device), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](z_) * (1 - self.masks[i])
            t = self.t[i](z_) * (1 - self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x: torch.Tensor):
        z, logp = self.f(x)
        return self.prior.log_prob(z).to(z.device) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


class MAF(nn.Module):
    def __init__(
        self, dim: int, n_layers: int, hidden_dims: List[int], use_reverse: bool = True
    ):
        """
        Args:
            dim: Dimension of input. E.g.: dim = 784 when using MNIST.
            n_layers: Number of layers in the MAF (= number of stacked MADEs).
            hidden_dims: List with of sizes of the hidden layers in each MADE.
            use_reverse: Whether to reverse the input vector in each MADE.
        """
        super().__init__()
        self.dim = dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()

        for _ in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNormLayer(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_sum = torch.zeros(x.shape[0]).to(x.device)
        # Forward pass.
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det

        return x, log_det_sum

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        u, log_det = self.forward(x)
        negloglik = 0.5 * (u**2).sum(dim=1)
        negloglik += 0.5 * x.shape[-1] * np.log(2 * math.pi)
        negloglik -= log_det
        return -negloglik

    def backward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_sum = torch.zeros(x.shape[0]).to(x.device)
        # Backward pass.
        for layer in reversed(self.layers):
            x, log_det = layer.backward(x)
            log_det_sum += log_det

        return x, log_det_sum


class GMM(nn.Module):
    def __init__(self, num_labels: int, gda_size: int) -> None:
        """_summary_

        Args:
            num_labels (int): num of labels.
            gda_size (int): size of hidden representation.
        """
        super(GMM, self).__init__()
        self.num_labels = num_labels
        self.gda_size = gda_size
        self.mu = torch.zeros(num_labels, gda_size).to(device)
        self.Sigma = torch.stack(
            [torch.eye(gda_size, gda_size) for _ in range(num_labels)]
        ).to(device)
        self.determinants = torch.zeros(num_labels).to(device)

    def fit(self, X: torch.Tensor, y: torch.Tensor, id2label: dict[int, str]) -> None:
        for lid in id2label.keys():
            num_batch_classes = (y == lid).long().sum()
            self.mu[lid] = X[y == lid].mean(dim=0)
            self.Sigma[lid] = torch.FloatTensor(
                np.cov(X[y == lid].T.detach().cpu().numpy())
            ).to(X.device) * (num_batch_classes - 1)

            self.determinants[lid] = torch.det(self.Sigma[lid, :, :])
            self.Sigma[lid, :, :] = torch.linalg.inv(self.Sigma[lid, :, :])

    def log_prob(self, x: torch.Tensor) -> torch.FloatTensor:
        x = x.unsqueeze(1)  # (batch_size x seq_length) x 1 x input_size
        x = x.repeat(1, self.num_labels, 1)
        diff = x - self.mu  # (batch_size x seq_length) x output_size x input_size
        diff_t = rearrange(diff, "b o i -> b i o")
        probs = torch.log(
            1
            - (
                1
                / (2 * math.pi * self.determinants + 1e-6)
                * torch.exp(
                    -0.5 * torch.einsum("boi,oii,bio->bo", diff, self.Sigma, diff_t)
                )
            )
            + 1e-6
        )  # (batch_size x seq_length) x output_size
        return probs


class NormalizingFlow(nn.Module):
    def __init__(self, dim, flow_length, flow_type="planar_flow"):
        super(NormalizingFlow, self).__init__()
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        if self.flow_type == "radial_flow":
            self.transforms = nn.Sequential(*(Radial(dim) for _ in range(flow_length)))
        elif self.flow_type == "planar_flow":
            self.transforms = nn.Sequential(*(Planar(dim) for _ in range(flow_length)))
        elif self.flow_type == "iaf_flow":
            self.transforms = nn.Sequential(
                *(
                    affine_autoregressive(dim, hidden_dims=[128, 128])
                    for _ in range(flow_length)
                )
            )
        else:
            raise NotImplementedError

    def forward(self, z):

        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(
                z, z_next
            )
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x


"""

class PosteriorNetwork(nn.Module):
    def __init__(
        self,
        density_type: str,
        latent_dim: int,
        n_density: int,
        num_labels: int
    ) -> None:
        self.density_type = density_type
        self.latent_dim = latent_dim
        self.n_density = n_density
        self.num_labels = num_labels
        
        if self.density_type == "radial_flow":
            self.density_estimation = nn.ModuleList(
                [
                    NormalizingFlow(dim=self.latent_dim, flow_length=self.n_density, flow_type=self.density_type)
                    for _ in range(self.num_labels)
                ]
            )
        else:
            raise NotImplementedError()
        
        """


class RealNVPFactory:
    """RealNVP density estimator generator."""

    @staticmethod
    def create(embedding_dim: int, device: torch.device) -> RealNVP:
        nets = lambda: nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, embedding_dim),
            nn.Tanh(),
        )
        nett = lambda: nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, embedding_dim),
        )
        full_masks = []
        mask = torch.arange(0, embedding_dim) % 2
        for i in range(8):
            full_masks.append(mask)
            mask = 1 - mask
        full_masks = torch.stack(full_masks)

        prior = tdist.MultivariateNormal(
            torch.zeros(embedding_dim).to(device), torch.eye(embedding_dim).to(device)
        )
        density_estimator = RealNVP(nets=nets, nett=nett, masks=full_masks, prior=prior)
        return density_estimator
