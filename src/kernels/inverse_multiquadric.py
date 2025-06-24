from src.kernels.base import BaseKernel


class InverseMultiquadricKernel(BaseKernel):
    def __init__(self, lengthscale=1.0, variance=1.0, isotropic=True, alpha=1.0):
        super().__init__(lengthscale, variance, isotropic)
        self.alpha = alpha

    def __call__(self, X1, X2):
        sq_dist = self._squared_distance(X1, X2)
        return self.variance * (1 + sq_dist) ** (-self.alpha)