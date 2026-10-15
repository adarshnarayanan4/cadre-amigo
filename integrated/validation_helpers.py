"""Small NumPy-backed AMIGO vector/matrix adapters for solver checks."""

import numpy as np
from scipy.sparse import csr_matrix


class Vector:
    def __init__(self, a):
        self.array = np.asarray(a, dtype=float).copy()

    def get_array(self):
        return self.array

    def copy_device_to_host(self):
        pass

    def copy_host_to_device(self):
        pass


class Matrix:
    def __init__(self, a):
        self.array = csr_matrix(a)

    def get_nonzero_structure(self):
        a = self.array

        return *a.shape, a.nnz, a.indptr, a.indices

    def get_data(self):
        return self.array.data

    def copy_data_device_to_host(self):
        pass
