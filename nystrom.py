
import numpy as np
from scipy.linalg import svd
import random

class Nystrom:

    def __init__(self, kernel, num_sample) -> None:
        
        self.kernel = kernel
        self.num_sample = num_sample

        self.basis_ = None
        self.basis_indxs_ = None 
        self.normalization_ = None
        self._trained_flag = False
        self._prev_transform = None

    def fit(self, X, kern_el_dict=None, gram_matrix=None):
        """ This function randomly chooses landmark points and constructs the Nystrom embedding."""
        assert self.num_sample <= len(X), "Cannot sample more than supplied. {} <= {}".format(self.num_sample, len(X)) 
        X = np.array(X)
        
        self.basis_indxs_ = np.random.choice(len(X), self.num_sample)
        self.basis_ = X[self.basis_indxs_]

        if gram_matrix is None:
            basis_kernel = self.kernel(self.basis_, self.basis_, kern_el_dict=kern_el_dict)
        else:
            basis_kernel = gram_matrix[self.basis_indxs_]
            basis_kernel = basis_kernel[:, self.basis_indxs_]

        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        
        self._trained_flag = True
        return self

    def transform(self, X, kern_el_dict=None, gram_matrix=None):
        assert self._trained_flag, "Must train prior to transformation."

        if gram_matrix is None:
            embedded = self.kernel(X, self.basis_, kern_el_dict=kern_el_dict)
        else:
            embedded = gram_matrix[:,self.basis_indxs_]

        return np.dot(embedded, self.normalization_.T)

    def fit_transform(self, X, kern_el_dict=None, gram_matrix=None):
        self.fit(X, kern_el_dict=kern_el_dict, gram_matrix=gram_matrix)
        self._prev_transform = self.transform(X, kern_el_dict=kern_el_dict, gram_matrix=gram_matrix)
        return self._prev_transform