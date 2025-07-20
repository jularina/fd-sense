import numpy as np

def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)
