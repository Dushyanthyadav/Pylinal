import pytest
import math
from Pylinal.matrix import Matrix
from Pylinal.vector import Vector

# --------------------------------------------------------
# 1. Initialization & Representation Tests
# --------------------------------------------------------

def test_create_matrix():
    """Test that we can create a matrix and verify dims & stored values."""
    m = Matrix([[1, 2], [3, 4]])
    assert m.m == 2
    assert m.n == 2
    assert m.mat == ((1.0, 2.0), (3.0, 4.0))


def test_copy_constructor():
    """Matrix(Matrix_obj) should return an exact copy."""
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix(m1)
    assert m1 == m2


def test_empty_matrix_error():
    """Test that creating an empty matrix raises ValueError."""
    with pytest.raises(ValueError):
        Matrix([])


def test_invalid_matrix_type():
    """Test that invalid input types raise TypeError."""
    with pytest.raises(TypeError):
        Matrix(10)

    with pytest.raises(TypeError):
        Matrix("abc")


def test_unequal_row_length_error():
    """Test that matrices with uneven row lengths raise ValueError."""
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3, 4, 5]])


def test_matrix_repr():
    """Test matrix `__repr__` formatting output."""
    m = Matrix([[1, 20], [300, 4]])
    s = repr(m)
    assert "[  1 20]" in s
    assert "[300  4]" in s


# --------------------------------------------------------
# 2. Indexing & Iteration Tests
# --------------------------------------------------------

def test_indexing():
    """Test matrix indexing using row/column."""
    m = Matrix([[10, 20], [30, 40]])
    assert m[0][1] == 20.0
    assert m[1, 0] == 30.0


def test_iteration():
    """Test iteration over matrix rows."""
    m = Matrix([[1, 2], [3, 4]])
    assert list(m) == [(1.0, 2.0), (3.0, 4.0)]
    assert tuple(m) == ((1.0, 2.0), (3.0, 4.0))


# --------------------------------------------------------
# 3. Arithmetic Tests
# --------------------------------------------------------

def test_addition():
    """Test element-wise addition of matrices."""
    m1 = Matrix([[1, 2], [3, 4]])
    m2 = Matrix([[5, 6], [7, 8]])
    assert (m1 + m2) == Matrix([[6, 8], [10, 12]])


def test_subtraction():
    """Test element-wise subtraction of matrices."""
    m1 = Matrix([[10, 10], [5, 5]])
    m2 = Matrix([[3, 2], [1, 1]])
    assert (m1 - m2) == Matrix([[7, 8], [4, 4]])


def test_scalar_multiplication():
    """Test matrix * scalar and scalar * matrix."""
    m = Matrix([[1, -2], [3, 4]])
    assert (m * 3) == Matrix([[3, -6], [9, 12]])
    assert (3 * m) == Matrix([[3, -6], [9, 12]])


def test_elementwise_multiplication():
    """Test element-wise multiplication (Hadamard product)."""
    m1 = Matrix([[2, 3], [4, 5]])
    m2 = Matrix([[1, 2], [3, 4]])
    assert (m1 * m2) == Matrix([[2, 6], [12, 20]])


def test_scalar_division():
    """Test dividing a matrix by a scalar."""
    m = Matrix([[10, 20]])
    with pytest.raises(ZeroDivisionError):
        m / 0
    assert (m / 10) == Matrix([[1, 2]])


# --------------------------------------------------------
# 4. Transpose & Apply Tests
# --------------------------------------------------------

def test_transpose():
    """Test matrix transpose operation."""
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.T == Matrix([[1, 4], [2, 5], [3, 6]])


def test_apply_function():
    """Test applying a function element-wise."""
    m = Matrix([[1, 4], [9, 16]])
    result = m.apply(math.sqrt)
    assert result == Matrix([[1, 2], [3, 4]])


# --------------------------------------------------------
# 5. Determinant & Minor Tests
# --------------------------------------------------------

def test_determinant_2x2():
    """Test determinant of a 2x2 matrix."""
    m = Matrix([[4, 6], [3, 8]])
    assert m.determinant() == (4*8 - 6*3)


def test_determinant_3x3():
    """Test determinant of a 3x3 matrix."""
    m = Matrix([
        [1, 2, 3],
        [0, 4, 5],
        [1, 0, 6]
    ])
    assert m.determinant() == 22


def test_minor():
    """Test computing minor of a matrix."""
    m = Matrix([[1, 2], [3, 4]])
    assert m.minor(0, 0) == Matrix([[4]])


# --------------------------------------------------------
# 6. Identity & Zeros Tests
# --------------------------------------------------------

def test_identity_matrix():
    """Test generating identity matrix of given size."""
    I = Matrix.identity(3)
    assert I == Matrix([[1,0,0], [0,1,0], [0,0,1]])


def test_identity_bad_size():
    """Test identity() rejects invalid size."""
    with pytest.raises(ValueError):
        Matrix.identity(0)


def test_zeros_matrix():
    """Test generating a zero-filled matrix."""
    z = Matrix.zeros(2, 3)
    assert z == Matrix([[0,0,0], [0,0,0]])


# --------------------------------------------------------
# 7. Matrix Multiplication (@) Tests
# --------------------------------------------------------

def test_matrix_multiplication():
    """Test classical matrix multiplication."""
    A = Matrix([[1, 2],
                [3, 4]])
    B = Matrix([[2, 0],
                [1, 2]])
    assert (A @ B) == Matrix([[4, 4], [10, 8]])


def test_matmul_dimension_error():
    """Test @ operator raises ValueError for mismatched dims."""
    A = Matrix([[1]])
    B = Matrix([[1, 2]])
    with pytest.raises(ValueError):
        A @ B


# --------------------------------------------------------
# 8. Linear Transformation Tests
# --------------------------------------------------------

def test_linear_transformation():
    """Test matrix @ vector linear transformation."""
    M = Matrix([[1, 0], [0, 2]])
    v = Vector([3, 4])
    assert M @ v == Vector([3, 8])


def test_linear_transformation_dim_error():
    """Test mismatched dims in matrix-vector multiplication."""
    M = Matrix([[1, 0], [0, 1]])
    v = Vector([1, 2, 3])
    with pytest.raises(ValueError):
        M @ v


# --------------------------------------------------------
# 9. Error Handling Tests
# --------------------------------------------------------

def test_dimension_mismatch_add_sub():
    """Test that mismatched dims for add/sub raise ValueError."""
    A = Matrix([[1,2]])
    B = Matrix([[1,2],[3,4]])

    with pytest.raises(ValueError):
        A + B

    with pytest.raises(ValueError):
        A - B


def test_dimension_mismatch_elementwise_mul():
    """Test Hadamard product fails on mismatched dims."""
    A = Matrix([[1, 2]])
    B = Matrix([[1, 2, 3]])

    with pytest.raises(ValueError):
        A * B


def test_determinant_non_square():
    """Test determinant() requires square matrix."""
    with pytest.raises(ValueError):
        Matrix([[1,2,3]]).determinant()
