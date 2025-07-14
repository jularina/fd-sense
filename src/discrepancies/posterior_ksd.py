from src.kernels.base import BaseKernel
from src.discrepancies.ksd import KernelizedSteinDiscrepancy
from src.bayesian_model.base import BayesianModel
from src.utils.typing import ArrayLike


class PosteriorKSD:
    def __init__(
        self,
        samples: ArrayLike,
        model: BayesianModel,
        kernel: BaseKernel,
    ):
        """
        Computes Kernelized Stein Discrepancy (KSD) components between posterior samples and candidate posterior.

        Args:
            samples: Posterior samples (m, D)
            model: Bayesian model with score functions
            kernel: Kernel object with derivative methods
        """
        self.samples = samples
        self.model = model
        self.kernel = kernel
        self.ksd = KernelizedSteinDiscrepancy(model.posterior_score, self.kernel)

    def estimate_ksd(self) -> float:
        return self.ksd.compute(self.samples)

    def prior_term(self) -> float:
        return KernelizedSteinDiscrepancy(self.model.prior_score, self.kernel).compute(self.samples)

    def loss_term(self) -> float:
        return KernelizedSteinDiscrepancy(self.model.loss_score, self.kernel).compute(self.samples)

    def cross_term(self) -> float:
        m = len(self.samples)
        k_vals = self.kernel(self.samples, self.samples)  # (m, m)
        s_pi_vals = self.model.prior_score(self.samples)  # (m, D)
        s_l_vals = self.model.loss_score(self.samples)  # (m, D)

        cross = 0.0
        for i in range(m):
            for j in range(m):
                cross += s_pi_vals[i] @ s_l_vals[j] * k_vals[i, j]

        return (2 / m**2) * cross

    def hessian_term(self) -> float:
        grad2 = self.kernel.grad_xy(self.samples, self.samples)  # (m, m)
        return grad2.mean()

    def full_ksd_decomposition(self) -> dict[str, float]:
        prior = self.prior_term()
        loss = self.loss_term()
        cross = self.cross_term()
        hess = self.hessian_term()
        total = prior + loss + cross - hess
        return {
            "prior_term": prior,
            "loss_term": loss,
            "cross_term": cross,
            "hessian_term": hess,
            "estimated_ksd": total,
        }