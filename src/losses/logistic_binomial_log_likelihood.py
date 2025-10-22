import numpy as np

from src.utils.typing import ArrayLike


class LogisticBinomialLogLikelihood:
    """
    Binomial outcomes with logistic link:
        y_i ~ Bin(n_i, theta_i),   theta_i = sigmoid(alpha + beta * x_i)
    Score (grad log-lik) w.r.t. [alpha, beta] is a 2x1 vector:
        [[ sum_i (y_i - n_i*theta_i) ],
         [ sum_i x_i (y_i - n_i*theta_i) ]].
    """

    def __init__(self):
        pass

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # stable sigmoid
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def grad_log_pdf(
            self,
            theta: ArrayLike,
            y: ArrayLike,
            x: ArrayLike,
            n: ArrayLike
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        x = np.asarray(x, dtype=float).reshape(-1)
        n_trials = np.asarray(n, dtype=float).reshape(-1)
        assert y.shape == x.shape == n_trials.shape, "y, x, n must have same length"

        # linear predictor for all thetas and all observations -> (m, n)
        eta = theta[:, 0:1] + theta[:, 1:2] * x[None, :]
        p = self._sigmoid(eta)  # (m, n)

        # residuals (m, n)
        resid = y[None, :] - n_trials[None, :] * p

        # scores per theta -> (m,)
        g_alpha = np.sum(resid, axis=1)
        g_beta = np.sum(resid * x[None, :], axis=1)

        # stack -> (m, 2)
        return np.stack([g_alpha, g_beta], axis=1)