import numpy as np


class SGDOptimizer:
    def __init__(self, lr, momentum_hyperparameter=0.0, clip_norm=None):
        self.lr = lr
        self.momentum_hyperparameter = momentum_hyperparameter
        self.clip_norm = clip_norm
        self.cache = {}

    def update(self, param_name, param, param_grad, epoch):
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)

        if self.clip_norm is not None:
            if np.linalg.norm(param_grad) > self.clip_norm:
                param_grad = (
                    param_grad * self.clip_norm / np.linalg.norm(param_grad)
                )

        delta = (
            self.momentum_hyperparameter * self.cache[param_name]
            + self.lr * param_grad
        )
        self.cache[param_name] = delta
        return delta
