"""
2D Laplace / Poisson solver on a square grid.

Methods : Jacobi, red-black SOR (true Gauss-Seidel + over-relaxation)
Demos   : parallel-plate capacitor, E-field plot
Checks  : line charge near a grounded wall (method of images),
          order-of-convergence study, iterations vs N, iterations vs omega

Run:  python laplace_poisson_solver.py
Needs: numpy, matplotlib
"""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Save plots in a "plots" folder next to this script (not the current directory,
# which can be read-only, e.g. "/" when launched from some terminals/IDEs).
OUT = Path(__file__).resolve().parent / "plots"
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------
def _masks(fixed):
    """Cells allowed to change: not conductors, and not the outer edge."""
    upd = ~fixed
    upd[0, :] = upd[-1, :] = upd[:, 0] = upd[:, -1] = False   # outer edge = Dirichlet
    return upd


def jacobi(V0, fixed, rho=None, a=1.0, eps0=1.0, tol=1e-6, max_iter=200000):
    V = V0.copy()
    upd = _masks(fixed)
    src = 0 if rho is None else rho * a**2 / (4 * eps0)
    for it in range(1, max_iter + 1):
        new = V.copy()
        new[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] +
                                  V[1:-1, 2:] + V[1:-1, :-2])
        new = np.where(upd, new + src, V)
        change = np.abs(new - V).max()
        V = new
        if change < tol:
            break
    return V, it


def sor(V0, fixed, rho=None, a=1.0, eps0=1.0, omega=None, tol=1e-6, max_iter=200000):
    """Red-black SOR."""
    V = V0.copy()
    ny, nx = V.shape
    upd = _masks(fixed)
    src = np.zeros_like(V) if rho is None else rho * a**2 / (4 * eps0)
    if omega is None:
        omega = 2 / (1 + np.sin(np.pi / max(nx, ny)))
    I, J = np.indices(V.shape)
    colors = (upd & ((I + J) % 2 == 0), upd & ((I + J) % 2 == 1))
    for it in range(1, max_iter + 1):
        biggest = 0.0
        for color_mask in colors:          # red cells first, then black cells
            gs = V.copy()
            gs[1:-1, 1:-1] = 0.25 * (V[2:, 1:-1] + V[:-2, 1:-1] +
                                     V[1:-1, 2:] + V[1:-1, :-2]) + src[1:-1, 1:-1]
            delta = np.where(color_mask, omega * (gs - V), 0.0)
            V += delta
            biggest = max(biggest, np.abs(delta).max())
        if biggest < tol:
            break
    return V, it


def efield(V, a=1.0):
    Ey, Ex = np.gradient(-V, a)            # axis 0 = y (rows), axis 1 = x (columns)
    return Ex, Ey


# ----------------------------------------------------------------------------
# Demo 1: parallel-plate capacitor
# ----------------------------------------------------------------------------
def capacitor_demo():
    N = 101
    V0 = np.zeros((N, N))
    fixed = np.zeros((N, N), bool)
    V0[35, 25:76] = +10
    fixed[35, 25:76] = True
    V0[65, 25:76] = -10
    fixed[65, 25:76] = True

    t = time.time(); V, n_sor = sor(V0, fixed);    t_sor = time.time() - t
    t = time.time(); Vj, n_jac = jacobi(V0, fixed); t_jac = time.time() - t
    print(f"SOR: {n_sor} its ({t_sor:.2f}s) | Jacobi: {n_jac} its ({t_jac:.2f}s) "
          f"| max diff: {np.abs(V - Vj).max():.2e}")

    Ex, Ey = efield(V)
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, V, 40, cmap="RdBu_r")
    plt.colorbar(label="V")
    s = 3
    plt.quiver(X[::s, ::s], Y[::s, ::s], Ex[::s, ::s], Ey[::s, ::s])
    plt.gca().set_aspect("equal")
    plt.title("Parallel-plate capacitor: potential and E-field")
    plt.tight_layout()
    plt.savefig(OUT / "capacitor.png", dpi=150)


# ----------------------------------------------------------------------------
# Validation: 2D line charge near a grounded wall (method of images)
# Physical geometry is fixed: box of side L, charge at (x0, y0) in physical units.
# ----------------------------------------------------------------------------
def image_benchmark(a, L=200.0, x0=40.0, y0=100.0, lam=1.0, eps0=1.0,
                    tol=1e-9, exclude=8.0):
    """Returns (max error away from charge, iterations, V_num, err)."""
    N = int(round(L / a)) + 1
    ix0, iy0 = int(round(x0 / a)), int(round(y0 / a))
    X, Y = np.meshgrid(np.arange(N) * a, np.arange(N) * a)
    r1 = np.hypot(X - x0, Y - y0)
    r2 = np.hypot(X + x0, Y - y0)
    with np.errstate(divide="ignore"):
        V_exact = lam / (2 * np.pi * eps0) * np.log(r2 / r1)

    V0 = np.zeros((N, N))
    fixed = np.zeros((N, N), bool)
    for e in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        V0[e] = V_exact[e]                       # exact boundary values
    rho = np.zeros((N, N))
    rho[iy0, ix0] = lam / a**2                   # charge in one cell

    V_num, its = sor(V0, fixed, rho=rho, a=a, eps0=eps0, tol=tol)
    err = V_num - V_exact
    err[r1 < exclude] = np.nan                   # same physical region excluded for every a
    return np.nanmax(np.abs(err)), its, V_num, err


def validation():
    # error map at a = 1
    e, its, _, err = image_benchmark(a=1.0)
    print(f"Image benchmark a=1: {its} its, max err (r>8) = {e:.3e}")
    plt.figure(figsize=(6, 5))
    plt.imshow(err, origin="lower", cmap="RdBu")
    plt.colorbar(label="V_num - V_exact")
    plt.title("Error map (charge neighbourhood masked)")
    plt.tight_layout()
    plt.savefig(OUT / "image_error_map.png", dpi=150)

    # order of convergence: halve a with geometry held fixed
    spacings = [4.0, 2.0, 1.0, 0.5]
    errs = []
    for a in spacings:
        e, its, _, _ = image_benchmark(a)
        errs.append(e)
        print(f"  a = {a:<4}  N = {int(round(200 / a)) + 1:<4}  its = {its:<6}  max err = {e:.3e}")
    for i in range(len(spacings) - 1):
        p = np.log(errs[i] / errs[i + 1]) / np.log(spacings[i] / spacings[i + 1])
        print(f"  observed order between a={spacings[i]} and a={spacings[i+1]}: p = {p:.2f}")
    slope = np.polyfit(np.log(spacings), np.log(errs), 1)[0]
    print(f"  log-log fit slope = {slope:.2f}  (expect about 2)")

    plt.figure(figsize=(5, 4))
    plt.loglog(spacings, errs, "o-")
    plt.xlabel("grid spacing a"); plt.ylabel("max error")
    plt.title(f"Convergence, slope ~ {slope:.2f}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "convergence_order.png", dpi=150)


# ----------------------------------------------------------------------------
# Iteration counts: vs N (Jacobi ~ N^2, SOR ~ N) and vs omega
# ----------------------------------------------------------------------------
def _capacitor_problem(N):
    V0 = np.zeros((N, N))
    fixed = np.zeros((N, N), bool)
    r1, r2 = int(0.35 * N), int(0.65 * N)
    c1, c2 = int(0.25 * N), int(0.75 * N)
    V0[r1, c1:c2] = +10; fixed[r1, c1:c2] = True
    V0[r2, c1:c2] = -10; fixed[r2, c1:c2] = True
    return V0, fixed


def iteration_scaling():
    Ns = [21, 41, 61, 81, 101]
    nj, ns = [], []
    for N in Ns:
        V0, fixed = _capacitor_problem(N)
        nj.append(jacobi(V0, fixed)[1])
        ns.append(sor(V0, fixed)[1])
        print(f"  N = {N:<4} Jacobi = {nj[-1]:<6} SOR = {ns[-1]}")
    sj = np.polyfit(np.log(Ns), np.log(nj), 1)[0]
    ss = np.polyfit(np.log(Ns), np.log(ns), 1)[0]
    print(f"  slopes: Jacobi {sj:.2f} (expect ~2), SOR {ss:.2f} (expect ~1)")

    plt.figure(figsize=(5, 4))
    plt.loglog(Ns, nj, "o-", label=f"Jacobi (slope {sj:.2f})")
    plt.loglog(Ns, ns, "s-", label=f"SOR (slope {ss:.2f})")
    plt.xlabel("N"); plt.ylabel("iterations"); plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "iterations_vs_N.png", dpi=150)

    # iterations vs omega at N = 61
    N = 61
    V0, fixed = _capacitor_problem(N)
    w_opt = 2 / (1 + np.sin(np.pi / N))
    omegas = np.linspace(1.0, 1.98, 25)
    its = [sor(V0, fixed, omega=w)[1] for w in omegas]
    plt.figure(figsize=(5, 4))
    plt.plot(omegas, its, "o-")
    plt.axvline(w_opt, color="k", ls="--", label=f"omega_opt ~ {w_opt:.3f}")
    plt.xlabel("omega"); plt.ylabel("iterations"); plt.legend()
    plt.title(f"SOR iterations vs omega (N={N})")
    plt.tight_layout()
    plt.savefig(OUT / "iterations_vs_omega.png", dpi=150)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("== Capacitor ==")
    capacitor_demo()
    print("== Validation (method of images) ==")
    validation()
    print("== Iteration scaling ==")
    iteration_scaling()
    print(f"Plots saved in: {OUT}")
    plt.show()