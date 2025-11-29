import pytest
import math
from Pylinal.vector import Vector
from Pylinal.matrix import Matrix

# 1. Initialization & Representation Tests

def test_create_vector():
    """Test that we can create a vector from a list and tuple."""
    v1 = Vector([1, 2, 3])
    v2 = Vector((1, 2, 3))
    assert v1.vec == (1.0, 2.0, 3.0)
    assert v2.vec == (1.0, 2.0, 3.0)
    assert len(v1) == 3


def test_string_representation():
    """Test the __repr__ output."""
    v = Vector([1.5, 2.0])
    assert repr(v) == "Vector([1.5 2.0])"


def test_indexing_and_iteration():
    """Test __getitem__ and __iter__."""
    v = Vector([10, 20, 30])
    assert v[1] == 20.0
    assert list(v) == [10.0, 20.0, 30.0]
    assert tuple(v) == (10.0, 20.0, 30.0)

# 2. Arithmetic Tests (Parametrized)

def test_negation():
    v1 = Vector([1,-2, 3])
    v2 = -v1
    assert v2 == Vector([-1, 2, -3])

# We use parametrization to test multiple scenarios in one function
@pytest.mark.parametrize("v1_data, v2_data, expected", [
    ([1, 2], [3, 4], [4, 6]),  # Basic Integers
    ([1.5, 2.5], [0.5, 0.5], [2.0, 3.0]),  # Floats
    ([1, 2], [-1, -2], [0, 0]),  # Negatives
])
def test_addition(v1_data, v2_data, expected):
    v1 = Vector(v1_data)
    v2 = Vector(v2_data)
    result = v1 + v2
    assert result == Vector(expected)


def test_subtraction():
    v1 = Vector([10, 10])
    v2 = Vector([2, 3])
    assert (v1 - v2) == Vector([8, 7])


def test_scalar_multiplication():
    v = Vector([1, -2])
    assert (v * 3) == Vector([3, -6])
    assert (3 * v) == Vector([3, -6])  # Tests __rmul__


def test_elementwise_multiplication():
    """Tests the Hadamard product (vector * vector)."""
    v1 = Vector([2, 3])
    v2 = Vector([4, 5])
    assert (v1 * v2) == Vector([8, 15])


def test_division():
    v = Vector([10, 20])
    with pytest.raises(ZeroDivisionError, match="Cannot divide by Zero"):
        v / 0
    assert (v / 2) == Vector([5, 10])

# 3. Linear Algebra Logic Tests

def test_magnitude():
    # 3-4-5 triangle logic
    v = Vector([3, 4])
    assert v.magnitude == 5.0
    assert abs(v) == 5.0


def test_dot_product():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert v1.dot(v2) == 32


def test_cross_product():
    """Cross-product is strictly for 3D vectors."""
    i = Vector([1, 0, 0])
    j = Vector([0, 1, 0])
    k = Vector([0, 0, 1])

    # i x j should equal k
    assert i.cross(j) == k
    # j x i should equal -k
    assert j.cross(i) == -k

    a = Vector([2,3,4,21])
    b = Vector([2,3,4,21])

    with pytest.raises(ValueError, match=f"Dimension {a.dim}. Cross product is only possible for vector of dim 3"):
        a.cross(b)

def test_normalization_unit_vector():
    v = Vector([3, 0])
    u = v.unit
    assert u == Vector([1, 0])
    assert u.magnitude == 1.0


def test_angle_calculation():
    """Uses pytest.approx because PI is irrational."""
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])

    # Angle between X and Y axis is 90 degrees (pi/2 radians)
    rads = v1.angle(v2, degrees=False)
    degs = v1.angle(v2, degrees=True)

    assert rads == pytest.approx(math.pi / 2)
    assert degs == pytest.approx(90.0)


def test_project_onto():
    """
    Tests projecting vector A onto vector B.
    Formula: (A . B_unit) * B_unit
    """
    # Case 1: Projecting (3, 3) onto the X-axis (1, 0)
    # The shadow should be exactly (3, 0)
    v = Vector([3, 3])
    x_axis = Vector([1, 0])

    projection = v.project_onto(x_axis)

    assert projection.vec == (3.0, 0.0)

    # Case 2: Projecting (3, 4) onto the Y-axis (0, 1) -> Should be (0, 4)
    v2 = Vector([3, 4])
    y_axis = Vector([0, 1])
    assert v2.project_onto(y_axis).vec == (0.0, 4.0)

    # Case 3: Complex projection (requires approximate comparison)
    # Projecting (10, 0) onto (1, 1).
    # Result should be (5, 5).
    v3 = Vector([10, 0])
    target = Vector([1, 1])
    proj = v3.project_onto(target)

    assert proj[0] == pytest.approx(5.0)
    assert proj[1] == pytest.approx(5.0)


def test_outer_product():
    """
    Tests the outer product of two vectors.
    Result should be a Matrix.
    """
    u = Vector([1, 2])  # 2x1
    v = Vector([3, 4, 5])  # 3x1 (treated as 1x3 in outer product logic)

    # Expected result:
    # [ 1*3  1*4  1*5 ]   [ 3  4  5 ]
    # [ 2*3  2*4  2*5 ] = [ 6  8 10 ]

    result = u.outer_product(v)

    # Check it returns a Matrix
    assert isinstance(result, Matrix)

    # Check dimensions
    assert result.m == 2
    assert result.n == 3

    # Check values
    assert result == Matrix([[3, 4, 5], [6, 8, 10]])

# 4. Error Handling Tests (The "Crash" Path)

def test_dimension_mismatch_error():
    v2d = Vector([1, 2])
    v3d = Vector([1, 2, 3])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        v2d + v3d


def test_division_by_zero():
    v = Vector([1, 2])
    with pytest.raises(ZeroDivisionError):
        v / 0


def test_cross_product_dimension_error():
    v = Vector([1, 2])  # 2D vector
    with pytest.raises(ValueError):
        v.cross(v)


def test_normalize_zero_vector():
    z = Vector([0, 0, 0])
    with pytest.raises(ValueError, match="Cannot Normalize"):
        _ = z.unit


