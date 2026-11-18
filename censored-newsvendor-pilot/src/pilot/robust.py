"""Robust order quantities over an ambiguity set.

Minimax regret:  min_q  max_{j in 0..M}  max_{p in P}  [ c(q) - c(j) ]' p
Minimax cost:    min_q  max_{p in P}  c(q)' p

where c(q)[y] = b (y - q)^+ + h (q - y)^+. Regret is measured against the best
integer order j; for integer demand the best order is always an integer, so
this is exact.

`solve_robust` writes the whole problem as ONE linear program. For a fixed q
and benchmark j the inner maximisation over p is an LP; replacing it by its
dual turns "min over q of a max" into "min over q and the dual variables",
and because c(q)[y] is the maximum of two functions linear in q, every
constraint stays linear. `worst_case` solves the inner primal LPs directly; the
tests use it to check that the single LP is right (strong duality).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .ambiguity import AmbiguitySet, EmptyAmbiguitySet
from .newsvendor import unit_costs


@dataclass
class RobustSolution:
    q: float          # robust order quantity
    value: float      # worst-case regret (or cost) of q over the set: the certificate
    criterion: str


def _benchmarks(M: int, b: float, h: float, criterion: str) -> list[np.ndarray]:
    if criterion == "regret":
        return [unit_costs(j, M, b, h) for j in range(M + 1)]
    if criterion == "cost":
        return [np.zeros(M + 1)]
    raise ValueError(criterion)


def solve_robust(amb: AmbiguitySet, b: float, h: float, criterion: str = "regret") -> RobustSolution:
    M, lam, T, L = amb.M, amb.lam, amb.T, amb.L
    if amb.has_signal and amb.min_delta() > amb.delta + 1e-9:
        raise EmptyAmbiguitySet(f"delta={amb.delta:.4g} < min feasible {amb.min_delta():.4g}")
    y_low = np.arange(lam, dtype=float)
    y_tail = np.arange(lam, M + 1, dtype=float)
    bench = _benchmarks(M, b, h, criterion)
    J = len(bench)

    # variable layout: [q, t, z_0..z_{lam-1}, then per block j: mu, nu+ (L), nu- (L), eta]
    blk = 1 + 2 * L + (1 if L else 0)
    base = 2 + lam
    nvar = base + J * blk
    rows, cols, vals, rhs = [], [], [], []
    r = 0

    def emit(row_idx, col_idx, v, b_vec):
        nonlocal r
        rows.append(row_idx + r)
        cols.append(col_idx)
        vals.append(v)
        rhs.append(b_vec)
        r += len(b_vec)

    # lower region epigraph: z_y >= c(q)[y] for y < lam (costs are known there)
    if lam:
        k = np.arange(lam)
        for coef_q, sign in ((-b, -1.0), (h, 1.0)):      # b(y-q) <= z ;  h(q-y) <= z
            emit(np.concatenate([k, k]), np.concatenate([np.zeros(lam, int), 2 + k]),
                 np.concatenate([np.full(lam, coef_q), -np.ones(lam)]),
                 (-b * y_low) if sign < 0 else (h * y_low))

    Kt = amb.K.T if L else None                        # (T, L)
    nzi, nzf = np.nonzero(np.abs(Kt) > 1e-15) if L else (np.array([], int), np.array([], int))
    for j, cj in enumerate(bench):
        o = base + j * blk
        mu, nup, num, eta = o, o + 1 + np.arange(L), o + 1 + L + np.arange(L), o + 1 + 2 * L
        ct = cj[lam:]
        i = np.arange(T)
        # c(q)[y] - c_j[y] <= mu + (K' (nu+ - nu-))[y], split into its two linear pieces
        for coef_q, b_vec in ((-b, -b * y_tail + ct), (h, h * y_tail + ct)):
            ri = np.concatenate([i, i, nzi, nzi])
            ci = np.concatenate([np.zeros(T, int), np.full(T, mu), nup[nzf] if L else nzf, num[nzf] if L else nzf])
            vi = np.concatenate([np.full(T, coef_q), -np.ones(T),
                                 -Kt[nzi, nzf] if L else [], Kt[nzi, nzf] if L else []])
            emit(ri, ci, vi, b_vec)
        if L:
            # dual feasibility for the slack u_f:  nu+_f + nu-_f <= eta
            f = np.arange(L)
            emit(np.concatenate([f, f, f]), np.concatenate([nup, num, np.full(L, eta)]),
                 np.concatenate([np.ones(L), np.ones(L), -np.ones(L)]), np.zeros(L))
        # epigraph: known part + dual objective <= t
        ci = [np.arange(lam) + 2, [mu], [1]]
        vi = [amb.p_low, [amb.tail_mass], [-1.0]]
        if L:
            ci += [nup, num, [eta]]
            vi += [amb.beta, -amb.beta, [amb.delta]]
        ci = np.concatenate(ci)
        emit(np.zeros(len(ci), int), ci, np.concatenate(vi), [amb.p_low @ cj[:lam]])

    A = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(r, nvar)).tocsr()
    b_ub = np.concatenate([np.atleast_1d(x) for x in rhs]).astype(float)
    bounds = [(0, M), (None, None)] + [(None, None)] * lam
    for _ in range(J):
        bounds += [(None, None)] + [(0, None)] * (2 * L) + ([(0, None)] if L else [])
    c = np.zeros(nvar)
    c[1] = 1.0
    res = linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"robust LP failed: {res.message}")
    return RobustSolution(q=float(res.x[0]), value=float(res.x[1]), criterion=criterion)


def worst_case(amb: AmbiguitySet, q: float, b: float, h: float, criterion: str = "regret"):
    """Worst-case regret (or cost) of a GIVEN order q, by solving the inner
    primal LPs one benchmark at a time. Returns (value, worst-case p)."""
    M, lam, T, L = amb.M, amb.lam, amb.T, amb.L
    cq = unit_costs(q, M, b, h)
    best_val, best_p = -np.inf, None
    for cj in _benchmarks(M, b, h, criterion):
        g = cq - cj
        const = amb.p_low @ g[:lam]
        if not L:
            i = int(np.argmax(g[lam:]))
            tail = np.zeros(T)
            tail[i] = amb.tail_mass
        else:
            c = -np.concatenate([g[lam:], np.zeros(L)])
            A_ub = np.vstack([np.hstack([amb.K, -np.eye(L)]), np.hstack([-amb.K, -np.eye(L)]),
                              np.concatenate([np.zeros(T), np.ones(L)])[None, :]])
            b_ub = np.concatenate([amb.beta, -amb.beta, [amb.delta]])
            A_eq = np.concatenate([np.ones(T), np.zeros(L)])[None, :]
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[amb.tail_mass],
                          bounds=(0, None), method="highs")
            if res.status != 0:
                raise EmptyAmbiguitySet(res.message)
            tail = res.x[:T]
        val = const + g[lam:] @ tail
        if val > best_val:
            best_val, best_p = val, np.concatenate([amb.p_low, tail])
    return float(best_val), best_p
