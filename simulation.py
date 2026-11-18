from algo import forward_euler, simulate
from kind import Scheme, MatrixSolver


def test_simulation(T: int, scheme: Scheme, matsolver: MatrixSolver):
        data = []

        for i in range(10):
            # we run each simulation 10x since values changes per simulation
            simulation_data, _ = simulate(
                    N=50,
                    D=1.0,
                    alpha=1.0,
                    c0=0.1,
                    T=T,
                    dt=0.2,
                    scheme=scheme,
                    matsolver=matsolver,
                    Tol=1e-10
                ) # we just need the last information
            last_data = simulation_data[-1]
            # print(last_data)

            data.append((last_data["run_time"], last_data["mean_c"], last_data["max_c"]))

        if scheme == Scheme.BACKWARDS_EULER:
            file.write(f"Scheme: {scheme} with MatSolver: {matsolver}\n")
        else:
            file.write(f"Scheme: {scheme}\n")

        file.write(f"T = {T}\n")
        file.write("run time        mean_c          max_c\n")

        for row in data:
            file.write(
                f"{row[0]:.10f}\t{row[1]:.10f}\t{row[2]:.10f}\n"
            )

        averages = [
            sum(row[j] for row in data) / 10
            for j in range(3)
        ]

        file.write(
            f"\nAverage: {averages[0]:.10f}\t"
            f"{averages[1]:.10f}\t{averages[2]:.10f}\n\n"
        )

# each simulation is done 10x
if __name__ == "__main__":
    with open("output.txt", "w") as file:
        # Forward Euler
        for T in [10, 25, 50]:
            test_simulation(T, Scheme.FORWARD_EULER, MatrixSolver.LU)

        # RK4
        for T in [10, 25, 50]:
            test_simulation(T, Scheme.RK4, MatrixSolver.LU)

        # Backwards Euler with LU
        for T in [10, 25, 50]:
            test_simulation(T, Scheme.BACKWARDS_EULER, MatrixSolver.LU)

        
        # Backwards Euler with Jacobi
        for T in [10, 25, 50]:
            test_simulation(T, Scheme.BACKWARDS_EULER, MatrixSolver.JACOBI)
            
        # Backwards Euler with Gauss-Seidel
        for T in [10, 25, 50]:
            test_simulation(T, Scheme.BACKWARDS_EULER, MatrixSolver.GAUSS_SEIDEL)