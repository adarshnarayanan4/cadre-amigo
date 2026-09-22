import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def knotopen(k, m):
    d = np.zeros(k + m)
    den = m - k + 1

    for i in range(k):
        d[i] = 0.0

    for i in range(k, m):
        d[i] = (i - k + 1) / den

    for i in range(m, m + k):
        d[i] = 1.0

    return d


def basis(k, t, d):
    m = len(d) - k
    i0 = -1

    for i in range(k - 1, m):
        if d[i] <= t < d[i + 1]:
            i0 = i - k + 1

    B = np.zeros(k)
    B[k - 1] = 1.0

    if t == d[m + k - 1]:
        i0 = m - k

    for ii in range(2, k + 1):
        offset = ii - 1
        j1 = k - offset
        j2 = k

        n = i0 + j1

        if d[n + offset] != d[n]:
            B[j1 - 1] = (d[n + offset] - t) / (d[n + offset] - d[n]) * B[j1]
        else:
            B[j1 - 1] = 0.0

        for j in range(j1 + 1, j2):
            n = i0 + j

            if d[n + offset - 1] != d[n - 1]:
                B[j - 1] = (
                    (t - d[n - 1]) / (d[n + offset - 1] - d[n - 1]) * B[j - 1]
                )
            else:
                B[j - 1] = 0.0

            if d[n + offset] != d[n]:
                B[j - 1] += (d[n + offset] - t) / (d[n + offset] - d[n]) * B[j]

        n = i0 + j2

        if d[n + offset - 1] != d[n - 1]:
            B[j2 - 1] = (
                (t - d[n - 1]) / (d[n + offset - 1] - d[n - 1]) * B[j2 - 1]
            )
        else:
            B[j2 - 1] = 0.0

    return B, i0


def paramuni(k, m, n):
    d = knotopen(k, m)
    C = np.linspace(0.0, 1.0, m)
    P = np.linspace(0.0, 1.0, n)
    t = np.zeros(n)

    for offset in range(n):
        x0 = 0.0
        x = 1.0

        B, i0 = basis(k, x0, d)
        f0 = -P[offset] + np.dot(B, C[i0 : i0 + k])

        B, i0 = basis(k, x, d)
        f = -P[offset] + np.dot(B, C[i0 : i0 + k])

        for _ in range(100):
            if abs(f) < 1e-15:
                break

            xnew = x - f * (x - x0) / (f - f0)

            if xnew < 0.0:
                xnew = -xnew
            elif xnew > 1.0:
                xnew = 2.0 - xnew

            x0 = x
            x = xnew
            f0 = f

            B, i0 = basis(k, x, d)
            f = -P[offset] + np.dot(B, C[i0 : i0 + k])

        t[offset] = x

    return t


def build_1d_matrix(t, k, m):
    d = knotopen(k, m)
    rows = []
    cols = []
    values = []

    for i in range(len(t)):
        B, i0 = basis(k, t[i], d)

        for j in range(k):
            rows.append(i)
            cols.append(i0 + j)
            values.append(B[j])

    return sp.csc_matrix((values, (rows, cols)), shape=(len(t), m))


def fit_coordinate_spline(x, k, m):
    t = paramuni(k, m, len(x))
    B = build_1d_matrix(t, k, m)
    BTB = B.T @ B
    BTx = B.T @ x
    Cx = spla.spsolve(BTB, BTx)
    Cx[0] = x[0]
    Cx[-1] = x[-1]

    return t, Cx


def inverse_map_value(x, Cx, k):
    m = len(Cx)
    d = knotopen(k, m)
    t = (x - Cx[0]) / (Cx[-1] - Cx[0])
    t = np.clip(t, 0.0, 1.0)

    for _ in range(100):
        B, i0 = basis(k, t, d)
        f = np.dot(B, Cx[i0 : i0 + k]) - x

        if abs(f) < 1e-13:
            break

        h = 1e-7
        tp = min(1.0, t + h)
        tm = max(0.0, t - h)

        Bp, i0p = basis(k, tp, d)
        Bm, i0m = basis(k, tm, d)

        fp = np.dot(Bp, Cx[i0p : i0p + k])
        fm = np.dot(Bm, Cx[i0m : i0m + k])

        dfdx = (fp - fm) / (tp - tm)

        if abs(dfdx) < 1e-14:
            break

        t = np.clip(t - f / dfdx, 0.0, 1.0)

    return t


class ModernMBI:
    def __init__(self, P, xs, ms, ks):
        self.xs = xs
        self.ms = np.array(ms, dtype=int)
        self.ks = np.array(ks, dtype=int)
        self.ns = np.array(P.shape[: len(xs)], dtype=int)
        self.nx = len(xs)

        self.ts = []
        self.Cx = []

        for i in range(self.nx):
            t, Cx = fit_coordinate_spline(
                np.asarray(xs[i], dtype=float), self.ks[i], self.ms[i]
            )
            self.ts.append(t)
            self.Cx.append(Cx)

        B0 = build_1d_matrix(self.ts[0], self.ks[0], self.ms[0])
        B1 = build_1d_matrix(self.ts[1], self.ks[1], self.ms[1])
        B2 = build_1d_matrix(self.ts[2], self.ks[2], self.ms[2])

        B = sp.kron(B2, sp.kron(B1, B0), format="csc")
        P_flat = np.asarray(P, dtype=float).reshape(-1, order="F")

        BTB = B.T @ B
        BTP = B.T @ P_flat

        print("Solving MBI spline coefficients...")
        self.C = spla.spsolve(BTB, BTP)

        print("MBI spline ready.")

    def evaluate_point(self, x):
        t0 = inverse_map_value(x[0], self.Cx[0], self.ks[0])
        t1 = inverse_map_value(x[1], self.Cx[1], self.ks[1])
        t2 = inverse_map_value(x[2], self.Cx[2], self.ks[2])

        B0, i00 = basis(self.ks[0], t0, knotopen(self.ks[0], self.ms[0]))
        B1, i01 = basis(self.ks[1], t1, knotopen(self.ks[1], self.ms[1]))
        B2, i02 = basis(self.ks[2], t2, knotopen(self.ks[2], self.ms[2]))

        value = 0.0

        for a in range(self.ks[0]):
            for b in range(self.ks[1]):
                for c in range(self.ks[2]):
                    i = i00 + a
                    j = i01 + b
                    k = i02 + c
                    index = i + self.ms[0] * j + self.ms[0] * self.ms[1] * k
                    value += B0[a] * B1[b] * B2[c] * self.C[index]

        return value

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            result[i] = self.evaluate_point(x[i, :])

        return result
