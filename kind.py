from dataclasses import dataclass
from enum import Enum


class Scheme(Enum):
    BACKWARDS_EULER = "Backwards_Euler"
    FORWARD_EULER = "Forward_Euler"
    RK4 = "RK4"


class MatrixSolver(Enum):
    LU = "LU"
    JACOBI = "Jacobi"
    GAUSS_SEIDEL = "Gauss_Seidel"