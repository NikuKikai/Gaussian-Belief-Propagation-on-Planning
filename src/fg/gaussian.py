from typing import Tuple
import numpy as np


class Gaussian:
    def __init__(self, dims: list, mean: np.ndarray = None, cov: np.ndarray = None):
        self._dims = list(dims)
        self._info, self._prec = None, None

        if mean is not None and cov is not None:
            mean = np.array(mean)
            cov = np.array(cov)
            if np.shape(mean) == ():
                mean = np.array([[mean]])
            elif len(np.shape(mean)) == 1:
                mean = np.array(mean)[:, None]
            assert len(dims) == len(mean) == np.shape(cov)[0] == np.shape(cov)[1]

            self._info, self._prec = self.mean2info(mean, cov)

    @staticmethod
    def info2mean(info: np.ndarray, prec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cov = np.linalg.inv(prec)
        mean = cov @ info
        return mean, cov

    @staticmethod
    def mean2info(mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prec = np.linalg.inv(cov)
        info = prec @ mean
        return info, prec

    @classmethod
    def identity(cls, dims: list) -> 'Gaussian':
        n = len(dims)
        return cls.from_info(dims, np.zeros((n, 1)), np.diag(np.ones(n) * 0.00001))

    @classmethod
    def from_info(cls, dims: list, info: np.ndarray, prec: np.ndarray) -> 'Gaussian':
        g = cls(dims)
        g._info = info
        g._prec = prec
        return g

    @property
    def dims(self) -> list:
        return self._dims.copy()

    @property
    def mean(self) -> np.ndarray:
        return np.linalg.inv(self._prec) @ self._info

    @property
    def cov(self) -> np.ndarray:
        return np.linalg.inv(self._prec)

    def copy(self) -> 'Gaussian':
        return Gaussian.from_info(self._dims.copy(), self._info.copy(), self._prec.copy())

    def __str__(self) -> str:
        return f'[info={self._info}, prec={self._prec}]'

    def __mul__(self, other: 'Gaussian') -> 'Gaussian':
        if other is None:
            return self.copy()
        # Merge dims
        dims = list(self._dims)
        for d in other._dims:
            if d not in dims:
                dims.append(d)
        # Extend self matrix
        prec_self = np.zeros((len(dims), len(dims)))
        info_self = np.zeros((len(dims), 1))
        idxs_self = [dims.index(d) for d in self._dims]
        prec_self[np.ix_(idxs_self, idxs_self)] = self._prec
        info_self[np.ix_(idxs_self, [0])] = self._info
        # Extend other matrix
        prec_other = np.zeros((len(dims), len(dims)))
        info_other = np.zeros((len(dims), 1))
        idxs_other = [dims.index(d) for d in other._dims]
        prec_other[np.ix_(idxs_other, idxs_other)] = other._prec
        info_other[np.ix_(idxs_other, [0])] = other._info
        # Add
        prec = prec_other + prec_self
        info = info_other + info_self
        return Gaussian.from_info(dims, info, prec)

    def __imul__(self, other: 'Gaussian') -> 'Gaussian':
        return self.__mul__(other)

    def marginalize(self, dims: list):
        """Given dims will be marginalized out.
        """
        info, prec = self._info, self._prec
        axis_a = [idx for idx, d in enumerate(self._dims) if d not in dims]
        axis_b = [idx for idx, d in enumerate(self._dims) if d in dims]
        info_a = info[np.ix_(axis_a, [0])]
        prec_aa = prec[np.ix_(axis_a, axis_a)]
        info_b = info[np.ix_(axis_b, [0])]
        prec_ab = prec[np.ix_(axis_a, axis_b)]
        prec_ba = prec[np.ix_(axis_b, axis_a)]
        prec_bb = prec[np.ix_(axis_b, axis_b)]

        prec_bb_inv = np.linalg.inv(prec_bb)
        info_ = info_a - prec_ab @ prec_bb_inv @ info_b
        prec_ = prec_aa - prec_ab @ prec_bb_inv @ prec_ba

        new_dims = tuple(d for d in self._dims if d not in dims)
        return Gaussian.from_info(new_dims, info_, prec_)
