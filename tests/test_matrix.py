import pytest
from Pylinal.matrix import Matrix
from Pylinal.vector import Vector

# --------------------------------------------------------
# 1. Initialization Tests
# --------------------------------------------------------

def test_create_matrix():
    m = Matrix([[1, 2], [3, 4]])
    assert m.m == 2
    assert m.n == 2
    assert m.mat == ((1.0, 2.0), (3.0, 4.0))


def test_copy_constructor():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix(m1)
    assert m1 == m2


def test_empty_matrix_error():
    with pytest.raises(ValueError):
        Matrix([])


def test_invalid_type():
    with pytest.raises(TypeError):
        Matrix(10)


# --------------------------------------------------------
# 2. Representation & Indexing
# --------------------------------------------------------

def test_repr():
    m = Matrix([[1, 20], [300, 4]])
    s = repr(m)

    # Expected actual output format of your __repr__
    # Example:
    # [  1.0  20.0]
    # [300.0   4.0]

    assert "[  1.0  20.0]" in s
    assert "[300.0   4.0]" in s



def test_indexing():
    m = Matrix([[10, 20], [30, 40]])
    assert m[0] == (10.0, 20.0)
    assert m[1][0] == 30.0


def test_iteration():
    m = Matrix([[1, 2], [3, 4]])
    assert list(m) == [(1.0, 2.0), (3.0, 4.0)]


# --------------------------------------------------------
# 3. Arithmetic Tests
# --------------------------------------------------------

def test_addition():
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    assert (m1 + m2) == Matrix([[6, 8], [10, 12]])


def test_subtraction():
    m1 = Matrix([[10, 10], [5, 5]])
    m2 = Matrix([[3, 2], [1, 1]])
    assert (m1 - m2) == Matrix([[7, 8], [4, 4]])


def test_hadamard_multiplication():
    m1 = Matrix([[2, 3], [4, 5]])
    m2 = Matrix([[1, 2], [3, 4]])
    assert (m1 * m2) == Matrix([[2, 6], [12, 20]])


def test_scalar_multiplication():
    m = Matrix([[1, -2], [3, 4]])
    assert (m * 3) == Matrix([[3, -6], [9, 12]])
    assert (3 * m) == Matrix([[3, -6], [9, 12]])


def test_scalar_division():
    m = Matrix([[10, 20]])
    assert (m / 10) == Matrix([[1, 2]])
    with pytest.raises(ZeroDivisionError):
        m / 0


# --------------------------------------------------------
# 4. Transpose Tests
# --------------------------------------------------------

def test_transpose():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.T == Matrix([[1, 4], [2, 5], [3, 6]])


# --------------------------------------------------------
# 5. Identity Test
# --------------------------------------------------------

def test_identity():
    I = Matrix.identity(3)
    assert I == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    with pytest.raises(ValueError):
        Matrix.identity(0)


# --------------------------------------------------------
# 6. Matrix Multiplication Tests
# --------------------------------------------------------

def test_matrix_multiplication():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[2, 0], [1, 2]])
    assert (A @ B) == Matrix([[4, 4], [10, 8]])


def test_matmul_dim_error():
    A = Matrix([[1, 2]])
    B = Matrix([[1, 2]])
    with pytest.raises(ValueError):
        A @ B  # dim mismatch


# --------------------------------------------------------
# 7. Linear Transformation (Matrix @ Vector)
# --------------------------------------------------------

def test_linear_transformation():
    M = Matrix([[1, 0], [0, 2]])
    v = Vector([3, 4])
    assert M @ v == Vector([3, 8])


def test_linear_transformation_dim_error():
    M = Matrix([[1, 0], [0, 1]])
    v = Vector([1, 2, 3])
    with pytest.raises(ValueError):
        M @ v
