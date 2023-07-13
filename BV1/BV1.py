import numpy as np


def homogenize(p: np.array):
    """Turn a"""
    return np.append(p, 1)

def dehomogenize(p: np.array):
    return (p / p[-1])[:-1]

def get_translation_matrix(tau: np.array = np.zeros((3, 1)), C: np.array = np.identity(3)):
    a = np.concatenate((C, np.zeros((1,3))))
    t = np.reshape(np.expand_dims(np.append(tau, 1), axis=0), (4, 1))
    return np.concatenate((a, t), axis=1)

def get_intrinsic_param_matrix(f: float):
    return np.array([[f, 0, 0, 0], [0, f, 0, 0], [0, 0, 1, 0]])

