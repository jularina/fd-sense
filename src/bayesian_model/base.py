from __future__ import annotations

class BayesianModel:
    def set_prior_parameters(self, prior_params, distribution_cls):
        pass

    def sample_posterior(self):
        pass

    def prior_score(self):
        pass

    def loss_score(self):
        pass

    def posterior_score(self):
        pass

    def jacobian_sufficient_statistics(self):
        pass

    def grad_log_base_measure(self):
        pass