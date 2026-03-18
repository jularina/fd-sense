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


class ECMOBinomialLogLikelihood:
    r"""
    ECMO model (Moreno 2000 / Kass-Greenhouse):

        y1 ~ Bin(n1, p1),   y2 ~ Bin(n2, p2)
        eta1 = logit(p1) = gamma - delta/2
        eta2 = logit(p2) = gamma + delta/2

    Parameter vector:
        theta = [gamma, delta]

    For each theta (possibly batched), the gradient of the log-likelihood is:

        d/dgamma log L = (y1 - n1*p1) + (y2 - n2*p2)
        d/ddelta log L = (-1/2)*(y1 - n1*p1) + (1/2)*(y2 - n2*p2)

    (Binomial coefficients are constants and drop out.)
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
        n: ArrayLike,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        theta : array-like, shape (m, 2) or (2,)
            [gamma, delta] values. If (2,), treated as a single theta.
        y : array-like, shape (2,)
            [y1, y2] successes in control and ECMO arms.
        n : array-like, shape (2,)
            [n1, n2] trials in control and ECMO arms.

        Returns
        -------
        grad : np.ndarray, shape (m, 2)
            Gradients of log-likelihood w.r.t. [gamma, delta] for each theta.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 1:
            theta = theta[None, :]
        assert theta.shape[1] == 2, "theta must have shape (m, 2) with columns [gamma, delta]"

        y = np.asarray(y, dtype=float).reshape(-1)
        n_trials = np.asarray(n, dtype=float).reshape(-1)
        assert y.shape == (2,) and n_trials.shape == (2,), "y and n must be length-2 vectors: [y1,y2], [n1,n2]"

        gamma = theta[:, 0]
        delta = theta[:, 1]

        # linear predictors for both arms -> (m,)
        eta1 = gamma - 0.5 * delta
        eta2 = gamma + 0.5 * delta

        p1 = self._sigmoid(eta1)
        p2 = self._sigmoid(eta2)

        # residuals per theta -> (m,)
        r1 = y[0] - n_trials[0] * p1
        r2 = y[1] - n_trials[1] * p2

        g_gamma = r1 + r2
        g_delta = -0.5 * r1 + 0.5 * r2

        return np.stack([g_gamma, g_delta], axis=1)
