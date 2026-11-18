"""
N - number of grid points
D - diffusivity
α - conversion rate from folder to misfolder
c0 - starting value of girst grid point
T - simulation time
dt - time step
scheme - ...
matsolver - ... (LU, jacobi, gauss-seidel)
Tol - convergence tolerance
"""
from kind import Scheme, MatrixSolver
import numpy as np
import time


def power_method(L, tol=1e-10, max_iter=1000):
    n = L.shape[0]
    x = np.ones(n) # initial eigenvector guess
    x = x / np.linalg.norm(x)

    eigenvalue_old = 0.0

    for _ in range(max_iter):
        y = L @ x
        x = y / np.linalg.norm(y)
        eigen_val = x @ L @ x

        if abs(eigen_val - eigenvalue_old) < tol:
            return eigen_val

        eigenvalue_old = eigen_val

    return eigenvalue_old


def setup(N):
    h = 1.0 / N
    L = np.zeros((N, N))

    np.fill_diagonal(L, 2.0) # diag = 2
    np.fill_diagonal(L[1:, :], -1.0) # off-diag = -1
    np.fill_diagonal(L[:, 1:], -1.0) # off-diag = -1
    L[0, 0] = L[N-1, N-1] = 1.0 # boundary entry

    L = L / h**2 # matrix scaling by 1/h^2

    return h, L


def helper(c, D, alpha, L):
    return -D * (L @ c) + alpha * c * (1.0 - c)


def forward_euler(c, dt, D, alpha, L):
    return c + dt * helper(c, D, alpha, L)


def rk4(c, dt, D, alpha, L):
    k1 = helper(c, D, alpha, L)
    k2 = helper(c + dt * k1 / 2, D, alpha, L)
    k3 = helper(c + dt * k2 / 2, D, alpha, L)
    k4 = helper(c + dt * k3, D, alpha, L)

    return c + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def backwards_euler(c, dt, D, alpha, L, matsolver: MatrixSolver, tol=1e-8, l_tol=1e-10, max_newton_iter=50, max_iter=1000):
    g = c.copy()
    total_lin_iter = 0 #total linear iterations
    newton_lin_iter = 0

    for newton_iterations in range(max_newton_iter):
        F = fsolver(g, c, dt, D, alpha, L)
        J = jacobian(g, dt, D, alpha, L)

        match matsolver:
            case MatrixSolver.LU:
                delta, lin_iter = lu_solver(J, -F)
            case MatrixSolver.JACOBI:
                delta, lin_iter = jacobi_solver(J, -F, tol=l_tol, max_iter=max_iter)
            case MatrixSolver.GAUSS_SEIDEL:
                delta, lin_iter = gauss_seidel_solver(J, -F, tol=l_tol, max_iter=max_iter)
            case _:
                raise ValueError("Maglagay kayo ng matsolver")

        total_lin_iter += lin_iter            
        newton_lin_iter += 1
        g = g + delta #numpy works!

        if np.max(np.abs(delta)) < tol:
            #Infinity Norm Check
            break

    return g, newton_lin_iter, total_lin_iter


def lu_solver(J, b): 
    n = J.shape[0]

    #initialize nxn matrix    
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)

    #Calculate First Row of U
    for j in range(n):
        U[0, j] = J[0, j]

    for i in range(n):
        L[i, i] = 1.0

    for i in range(1, n):
        L[i, 0] = J[i, 0]/ U[0, 0]

    for k in range(1, n):
        for j in range(k, n):
            U[k, j] = J[k, j] - sum(L[k, m] * U[m, j] for m in range(k))

        for i in range(k + 1, n):
            L[i, k] = (J[i, k] - sum(L[i, m] * U[m, k] for m in range(k))) / U[k, k]

    y = np.zeros(n, dtype=float)
    y[0] = b[0]

    for k in range(1, n):
        y[k] = b[k] - sum(L[k, j] * y[j] for j in range(k))

    x = np.zeros(n, dtype=float)
    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for k in range(n - 1, -1, -1):
        x[k] = (y[k] - sum(U[k, j] * x[j] for j in range(k + 1, n))) / U[k, k]

    return x, 0


def jacobi_solver(J, b, tol, max_iter):

    #x_next = -D^(-1)(L+U)x + D^(-1)b

    d = np.diag(J)

    LU = J - np.diag(d)

    T_J = -LU/ d[:, None]
    c_J = b / d

    #Initial Guess
    x = np.zeros_like(b, dtype=float)

    for i in range(1, max_iter + 1):
        x_next = (T_J @ x) + c_J

        if np.max(np.abs(x_next - x)) < tol:
            return x_next, i

        x = x_next

    return x, max_iter


def gauss_seidel_solver(J, b, tol, max_iter):
    #x_next = -(D + L)^(-1) * Ux + (D + L)^(-1) b

    DL = np.tril(J)

    U = np.triu(J, k=1)
    DL_inv = np.linalg.inv(DL)
    
    T_GS = -DL_inv @ U
    c_GS = DL_inv @ b

    #Initial Guess
    x = np.zeros_like(b, dtype=float)

    for i in range(1, max_iter + 1):
        x_next = T_GS @ x + c_GS

        if np.max(np.abs(x_next - x)) < tol:
            return x_next, i

        x = x_next

    return x, max_iter


def fsolver(g, c, dt, D, alpha, L):
    return g - c + (dt*D*(L@g)) - (dt*alpha*g*(1.0-g))


def jacobian(g, dt, D, alpha, L):
    n = len(g)
    I = np.eye(n)

    return I + (dt*D*L) - (dt*alpha*np.diag(1.0 - 2.0*g))


def front_position(c, h, threshold=0.5):
    loc = np.flatnonzero(c >= threshold)

    if len(loc) == 0:
        return float("nan")

    i = loc[-1]

    if i == len(c) -1:
        return float(i * h)

    frac = (threshold - c[i]) / (c[i + 1] - c[i])

    return float((i + frac) * h)


def simulate(N, D, alpha, c0, T, dt, scheme: Scheme, matsolver: MatrixSolver, Tol):
    h, L = setup(N)
    max_eigenvalue = power_method(L)
    max_timestop = 2.0 / (D * max_eigenvalue)

    c = np.zeros(N, dtype=float)
    c[0] = c0

    dataset = []
    current_time = 0.0
    step = 0

    for _ in range(int(T / dt)): # assume that T/dt is an int
        start = time.perf_counter()

        newton_iterations = 0
        linear_iterations = 0
        
        match scheme:
            case Scheme.FORWARD_EULER:
                c = forward_euler(c, dt, D, alpha, L)
            case Scheme.RK4:
                c = rk4(c, dt, D, alpha, L)
            case Scheme.BACKWARDS_EULER:
                c, newton_lin_iter, lin_iter = backwards_euler(c, dt, D, alpha, L, matsolver)
                linear_iterations = lin_iter
                newton_iterations = newton_lin_iter
            case _:
                raise ValueError("Invalid iteration scheme")

        elapsed = time.perf_counter() - start
        step += 1
        current_time += dt

        dataset.append({
            "step": step,
            "time": current_time,
            "mean_c": float(np.mean(c)),
            "front_position": front_position(c, h),
            "newton_iterations": newton_iterations,
            "linear_iterations": linear_iterations,
            "run_time": elapsed,
            "max_c": float(np.max(c))
        })

        if np.max(c) > 2 or not np.all(np.isfinite(c)):
            break

    return dataset, c


if __name__ == "__main__":
    results, final_c = simulate(
        N=50,
        D=1.0,
        alpha=1.0,
        c0=0.1,
        T=10,
        dt=0.2,
        scheme=Scheme.FORWARD_EULER,
        matsolver=MatrixSolver.LU,
        Tol=1e-10
    )

    print("Dataset result:")
    print(results[-1])
    # for data in results:
    #     print(data)

    print("Final c:")
    print(final_c)