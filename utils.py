import numpy as np


def D_norm(A):
    return np.diag(np.sum(A, axis=1) ** (-0.5))  # Degree matrix
