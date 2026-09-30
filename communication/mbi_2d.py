import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from power.mbi_modern import (
    basis,
    build_1d_matrix,
    fit_coordinate_spline,
    inverse_map_value,
    knotopen,
)


class ModernMBI2D:
    def __init__(self, P, xs, ms, ks):
        self.xs = xs
        self.ms = np.array(ms, dtype=int)
        self.ks = np.array(ks, dtype=int)

        self.ts = []
        self.Cx = []

        for i in range(2):
            t, Cx = fit_coordinate_spline(
                np.asarray(xs[i], dtype=float),
                self.ks[i],
                self.ms[i],
            )

            self.ts.append(t)
            self.Cx.append(Cx)

        B0 = build_1d_matrix(
            self.ts[0],
            self.ks[0],
            self.ms[0],
        )

        B1 = build_1d_matrix(
            self.ts[1],
            self.ks[1],
            self.ms[1],
        )

        B = sp.kron(
            B1,
            B0,
            format="csc",
        )

        P_flat = np.asarray(
            P,
            dtype=float,
        ).reshape(
            -1,
            order="F",
        )

        BTB = B.T @ B
        BTP = B.T @ P_flat

        print(
            "Solving 2-D MBI spline coefficients..."
        )

        self.C = spla.spsolve(
            BTB,
            BTP,
        )

        print(
            "2-D MBI spline ready."
        )

    def evaluate_point(self, x):
        t0 = inverse_map_value(
            x[0],
            self.Cx[0],
            self.ks[0],
        )

        t1 = inverse_map_value(
            x[1],
            self.Cx[1],
            self.ks[1],
        )

        B0, i00 = basis(
            self.ks[0],
            t0,
            knotopen(
                self.ks[0],
                self.ms[0],
            ),
        )

        B1, i01 = basis(
            self.ks[1],
            t1,
            knotopen(
                self.ks[1],
                self.ms[1],
            ),
        )

        value = 0.0

        for a in range(self.ks[0]):
            for b in range(self.ks[1]):
                i = i00 + a
                j = i01 + b

                index = (
                    i
                    + self.ms[0] * j
                )

                value += (
                    B0[a]
                    * B1[b]
                    * self.C[index]
                )

        return value

    def evaluate(self, x):
        x = np.asarray(
            x,
            dtype=float,
        )

        result = np.zeros(
            x.shape[0]
        )

        for i in range(x.shape[0]):
            result[i] = (
                self.evaluate_point(
                    x[i, :]
                )
            )

        return result