import copy
from Pylinal.matrix import Matrix
from Pylinal.vector import Vector

def _ensure_matrix(A):
    if isinstance(A, Matrix):
        return True
    else:
        raise TypeError(f"Expected Matrix, got {type(A)}")


def _ensure_vector(A):
    if isinstance(A, Vector):
        return True
    else:
        raise TypeError(f"Expected Vector, got {type(A)}")

def _get_mutable_matrix(A):
    if _ensure_matrix(A):
        return [list(row) for row in A]
    return None
def _get_mutable_vector(A):
    if _ensure_vector(A):
        return [elem for elem in A]


def trace(A):
    if A.m != A.n:
        raise ValueError("Trace is defined only for square matrices")
    return sum(A[i, i] for i in range(A.m))


def rref(matrix):
    if _ensure_matrix(matrix):

        M = _get_mutable_matrix(matrix)

        rows = matrix.m
        cols = matrix.n

        pivot_row = 0

        for c in range(cols):
            if pivot_row >= rows:
                break

            max_row_idx = pivot_row

            for r in range(pivot_row + 1, rows):
                if abs(M[r][c]) > abs(M[max_row_idx][c]):
                    max_row_idx = r


            if abs(M[max_row_idx][c]) < 1e-9:
                continue


            M[pivot_row], M[max_row_idx] = M[max_row_idx], M[pivot_row]

            pivot_val = M[pivot_row][c]
            M[pivot_row] = [x / pivot_val for x in M[pivot_row]]


            for r in range(rows):
                if r != pivot_row:
                    factor = M[r][c]
                    M[r] = [current - factor * pivot_el for current, pivot_el in zip(M[r], M[pivot_row])]


            pivot_row += 1

        return Matrix(M)
    return None

def rank(matrix):
    if _ensure_matrix(matrix):
        rref_matrix = rref(matrix)

        rank = 0

        for row in rref_matrix:
            if any(abs(x) > 1e-9 for x in row):
                rank += 1
        return rank
    return None

def solve(mat_a, vec_b):
    return mat_a.inverse() @ vec_b
