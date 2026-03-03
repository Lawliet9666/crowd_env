import numpy as np
from crowd_sim.env.robot.obstacle import GMMNoiseModel

class TrajPredictor:
    def __init__(self, seed=1234, gmm=None, gmm_params=None):
        """
        Fake predictor that mirrors the environment's GMMNoiseModel.
        If gmm is provided, it will be used directly; otherwise a new one
        is created with the same defaults as the obstacle model.
        """
        if gmm is not None:
            self.gmm = gmm
        else:
            kwargs = dict(gmm_params or {})
            kwargs.setdefault("seed", seed)
            self.gmm = GMMNoiseModel(**kwargs)

    def predict_gmm(self, current_vel):
        """
        Fakes a prediction model that outputs a GMM for the next velocity.
        In reality, this would use a neural network or statistical model.
        
        Returns:
            weights: (n_components,)
            means: (n_components, 2)
            covariances: (n_components,) - assuming spherical covariance
        """
        num_components = int(len(self.gmm.weights_))
        means = np.zeros((num_components, 2), dtype=float)
        for comp_idx in range(num_components):
            mu, _ = self.gmm._build_component_mean(current_vel, comp_idx)
            means[comp_idx] = mu
        weights = self.gmm.weights_.copy()
        covariances = self.gmm.covariances_.copy()
        return weights, means, covariances

    # def predict_vel(self, current_vel):
    #     """
    #     Takes the actual velocity and returns a noisy 'predicted' velocity.
    #     """
    #     return self.gmm.sample(nominal_action=current_vel)

    def predict_vel_expectation(self, current_vel):
        """
        Returns the expected velocity under the GMM prediction.
        """
        # weights, means, _ = self.predict_gmm(current_vel)
        # expected_vel = np.sum(weights[:, np.newaxis] * means, axis=0)
        expected_vel = current_vel.copy()  # Default to current velocity if GMM is not used
        return expected_vel
    
