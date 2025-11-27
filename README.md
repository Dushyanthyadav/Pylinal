# Linear Algebra Module

A implementation of linear algebra concepts in Python.

## Usage

```python
from vector import Vector

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

# Vector Addition
V3 = v1 + v2
print(v3)

# Scalar Multiplication
v3 = 2*v2
print(v3)

# Dot product
print(v1.dot(v2)) 

# Cross product
print(v1.cross(v2)) 
