"""OpenBMI subject-specific CSP and LDA inference from numeric weights."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch


class OpenBMICspLda:
    def __init__(self, weights):
        shapes = {"filters": (6, 62), "coef": (1, 6), "intercept": (1,)}
        if weights.keys() != shapes.keys() or any(
            tuple(weights[key].shape) != shape or not torch.isfinite(weights[key]).all()
            for key, shape in shapes.items()
        ):
            raise ValueError("Invalid OpenBMI CSP/LDA parameters.")
        self.filters = weights["filters"].numpy()
        self.coef = weights["coef"].numpy()
        self.intercept = weights["intercept"].numpy()
        self.bandpass = butter(4, [8, 30], fs=1000, btype="bandpass", output="sos")

    def __call__(self, tensor):
        microvolts = tensor[:, 0].numpy().astype(np.float64) * 1e6
        signal = sosfiltfilt(self.bandpass, microvolts, axis=-1)
        power = np.mean((self.filters @ signal) ** 2, axis=-1)
        if not np.all(np.isfinite(power) & (power > 0)):
            raise ValueError("OpenBMI CSP power must be positive and finite.")
        features = np.log(power)
        decision = features @ self.coef.T + self.intercept
        logits = np.concatenate([np.zeros_like(decision), decision], axis=1)
        return torch.from_numpy(features), torch.from_numpy(logits)
