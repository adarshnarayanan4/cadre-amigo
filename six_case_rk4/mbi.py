"""
MBI tensor B-spline fit
-----------------------
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import BSpline


def knotopen(k, m):
    """Open uniform knot vector of m control points of order k."""
    d = np.zeros(k + m)
    den = m - k + 1
    for i in range(k, m):
        d[i] = (i - k + 1) / den
    d[m:] = 1.0
    return d


def basis(k, t, d):
    """Nonzero basis values at t and the index of the first one."""
    m = len(d) - k
    i0 = -1
    for i in range(k, m + 1):
        if d[i - 1] <= t < d[i]:
            i0 = i - k

    B = np.zeros(k)
    B[k - 1] = 1.0
    if t == d[m + k - 1]:
        i0 = m - k

    # Cox de Boor recursion in the order of basis.f90
    for i in range(2, k + 1):
        l = i - 1
        j1 = k - l
        j2 = k
        n = i0 + j1
        if d[n + l] != d[n]:
            B[j1 - 1] = (d[n + l] - t) / (d[n + l] - d[n]) * B[j1]
        else:
            B[j1 - 1] = 0.0
        for j in range(j1 + 1, j2):
            n = i0 + j
            if d[n + l - 1] != d[n - 1]:
                B[j - 1] = (t - d[n - 1]) / (d[n + l - 1] - d[n - 1]) * B[j - 1]
            else:
                B[j - 1] = 0.0
            if d[n + l] != d[n]:
                B[j - 1] = B[j - 1] + (d[n + l] - t) / (d[n + l] - d[n]) * B[j]
        n = i0 + j2
        if d[n + l - 1] != d[n - 1]:
            B[j2 - 1] = (t - d[n - 1]) / (d[n + l - 1] - d[n - 1]) * B[j2 - 1]
        else:
            B[j2 - 1] = 0.0
    return B, i0


def paramuni(k, m, n):
    """Parameters of n data points that the control polygon places uniformly."""
    d = knotopen(k, m)
    C = np.array([l / (m - 1) for l in range(m)])
    t = np.zeros(n)

    # The Fortran literal 1e-15 is single precision
    tol = float(np.float32(1e-15))

    def residual(x, P):
        B, i0 = basis(k, x, d)
        f = -P
        for i in range(k):
            f = f + B[i] * C[i0 + i]
        return f

    # Secant iteration with reflection at the ends as in paramuni.f90
    for l in range(n):
        P = l / (n - 1)
        x0, x = 0.0, 1.0
        f0 = residual(x0, P)
        f = residual(x, P)
        for _ in range(100):
            if abs(f) < tol:
                break
            xnew = x - f * (x - x0) / (f - f0)
            if xnew < 0:
                xnew = 0 - xnew
            elif xnew > 1:
                xnew = 2 - xnew
            x0 = x
            x = xnew
            f0 = f
            f = residual(x, P)
        t[l] = x
    return t


def basis_matrix(ks, ms, ts):
    """Sparse basis matrix on the tensor grid of the axis parameters ts."""
    ns = [len(t) for t in ts]
    nx = len(ks)
    nP = int(np.prod(ns))
    na = int(np.prod(ks))

    # Grid index of every point along every axis with the first axis fastest
    points = np.arange(nP)
    Ba = np.ones((nP, na))
    Bj = np.zeros((nP, na), dtype=np.int64)
    for ix in range(nx):
        k, m = ks[ix], ms[ix]
        d = knotopen(k, m)
        values = np.zeros((ns[ix], k))
        first = np.zeros(ns[ix], dtype=np.int64)
        for i, t in enumerate(ts[ix]):
            values[i], first[i] = basis(k, t, d)

        grid = (points // int(np.prod(ns[:ix]))) % ns[ix]
        digit = (np.arange(na) // int(np.prod(ks[:ix]))) % k
        Ba *= values[grid][:, digit]
        Bj += (first[grid][:, None] + digit[None, :]) * int(np.prod(ms[:ix]))

    Bi = np.repeat(points, na)
    return scipy.sparse.csc_matrix((Ba.ravel(), (Bi, Bj.ravel())))


class MBI:
    """Least squares tensor B-spline fit of gridded data as MBI.MBI builds it.

    P holds the samples on the grid of the axes xs with any trailing axis
    holding separate functions. The coefficients C have one row per control
    point with the first axis fastest and the coordinate splines Cx map the
    parameter of each axis to its physical values.
    """

    def __init__(self, P, xs, ms, ks):
        nx = len(xs)
        nf = 1 if len(P.shape) == nx else P.shape[nx]
        ns = list(P.shape[:nx])
        self.ms = [int(m) for m in ms]
        self.ks = [min(int(k), int(m)) for k, m in zip(ks, ms)]
        self.nx, self.nf = nx, nf
        self.ts = [paramuni(k, m, n) for k, m, n in zip(self.ks, self.ms, ns)]

        # Control points of the data by conjugate gradients as MBI.py solves them
        B = basis_matrix(self.ks, self.ms, self.ts)
        BT = B.transpose()
        P = np.reshape(P, (int(np.prod(ns)), nf), order="F")
        BTB, BTP = BT.dot(B), BT.dot(P)
        self.C = np.zeros((int(np.prod(self.ms)), nf), order="F")
        for i in range(nf):
            self.C[:, i] = scipy.sparse.linalg.cg(BTB, BTP[:, i])[0]

        # Coordinate splines with their end values pinned to the axis ends
        self.Cx = []
        for i in range(nx):
            B = basis_matrix([self.ks[i]], [self.ms[i]], [self.ts[i]])
            BT = B.transpose()
            BTB, BTP = BT.dot(B), BT.dot(xs[i])
            cx = scipy.sparse.linalg.cg(BTB, BTP)[0]
            cx[0] = xs[i][0]
            cx[-1] = xs[i][-1]
            self.Cx.append(cx)

    def knots(self, axis):
        """Knot vector of one axis."""
        return knotopen(self.ks[axis], self.ms[axis])

    def coefficients(self):
        """Control points reshaped to one axis per table dimension and one for the functions."""
        return self.C.reshape(tuple(self.ms) + (self.nf,), order="F")


def basis_values(k, m, t, nu=0):
    """Values or derivatives of all m basis functions at the parameters t, one row per parameter."""
    t = np.clip(np.atleast_1d(np.asarray(t, dtype=float)), 0.0, 1.0)
    return BSpline(knotopen(k, m), np.eye(m), k - 1)(t, nu)


def inverse(cx, k, x):
    """Parameters where the coordinate spline cx takes the values x, held at the ends outside its range."""
    cx = np.asarray(cx, dtype=float)
    if not np.all(np.diff(cx) > 0.0):
        raise ValueError("The coordinate spline must increase")
    spline = BSpline(knotopen(k, len(cx)), cx, k - 1)
    x = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), cx[0], cx[-1])

    # Bisection to the last bit and then a Newton polish on the monotone spline
    lo = np.zeros_like(x)
    hi = np.ones_like(x)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = spline(mid) < x
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    t = 0.5 * (lo + hi)
    for _ in range(2):
        t = np.clip(t - (spline(t) - x) / spline(t, 1), 0.0, 1.0)
    return t
