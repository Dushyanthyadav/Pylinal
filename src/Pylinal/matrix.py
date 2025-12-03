from .vector import Vector
class Matrix:

    __slots__ = ("mat", "m", "n")

    def __init__(self, matrix):

        if isinstance(matrix, Matrix):
            self.mat = matrix.mat
            self.m = matrix.m
            self.n = matrix.n
            return
        
        if not isinstance(matrix, (list, tuple, Matrix)):
            raise TypeError(f"{type(matrix)} can not be a Matrix")
        
        if not matrix or not matrix[0]:
            raise ValueError("Matrix cannot be empty")
        col_len = len(matrix[0])

        clean_matrix = []
        try:
            for row in matrix:
                if len(row) != col_len:
                    raise ValueError("Matrix rows must have equal length.")
                clean_matrix.append(tuple(float(x) for x in row))
        except (TypeError, ValueError):
            raise TypeError(f"All element should be int or float")

        self.mat = tuple(clean_matrix)
        self.m = len(matrix)
        self.n = len(matrix[0])

    def __repr__(self):
        str_matrix = [[str(item) for item in row] for row in self.mat]
        max_len = max(len(item) for row in str_matrix for item in row)
        lines = []
        for row in str_matrix:
            formatted_row = [f"{item:>{max_len}}" for item in row]
            lines.append(f"[{' '.join(formatted_row)}]")
        output = "\n".join(lines)
        
        return output

    def __iter__(self):
        return iter(self.mat)

    def __getitem__(self, index):
        return self.mat[index]
    
    def _ensure_matrix(self, other): # Matrix can take in list and tuple
        if isinstance(other, Matrix):
            return other
        if isinstance(other, (list, tuple, Matrix)):
            return Matrix(other)
        raise TypeError(f"{type(other)} can not be a matrix")
        
    def _ensure_dim(self, other):
        if (self.m, self.n) != (other.m, other.n):
            raise ValueError(f"({self.m}, {self.n}) vs ({other.m}, {other.n}) Dimension miss match")
        return other

    def __eq__(self, other):
        other = self._ensure_dim(self._ensure_matrix(other))
        if (self.m, self.n) != (other.m, other.n):
            return False
        return all([row_a == row_b for row_a, row_b in zip(self, other)])
    
    def __add__(self, other):
        other = self._ensure_dim(self._ensure_matrix(other))
        return Matrix([[a+b for a,b in zip(self_row, other_row)] for self_row, other_row in zip(self.mat, other.mat)])     

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._ensure_dim(self._ensure_matrix(other))
        return Matrix([[a-b for a,b in zip(self_row, other_row)] for self_row, other_row in zip(self.mat, other.mat)])     

    def __rsub__(self, other):
        other = self._ensure_dim(self._ensure_matrix(other))
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int,float)):
            return Matrix([[i*other for i in row] for row in self])
        other = self._ensure_dim(self._ensure_matrix(other))
        return Matrix([[a*b for a, b in zip(row_A,row_b)] for row_A, row_b in zip(self, other)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Matrix division by non-scalar {type(other)} is not supported.")
        if other == 0:
            raise ZeroDivisionError("Cannot divide by Matrix by zero")

        return self.__mul__(1.0 /other)
        
    @property
    def T(self):
        return Matrix(tuple(zip(*self.mat)))
    
    @classmethod
    def identity(cls, size):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a Positive integer.")
        return cls([[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)])
        
    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self.lin_trans(other)

        other = self._ensure_matrix(other)

        if self.n != other.m:
            raise ValueError(f"({self.m}x{self.n}) vs ({other.m}x{other.n}) Dimension mismatch")

        other_T = other.T
        
        return Matrix([[sum(a*b for a, b in zip(row_A, col_B)) for col_B in other_T] for row_A in self])
                        
    def lin_trans(self, other):
        other = Vector(other)
        if self.n == other.dim:
            return Vector([sum(a*b for a,b in zip(rows, other)) for rows in self.mat])
        else:
            raise ValueError(f"({self.m}x{self.n}) vs (1x{other.dim}) Dimension mismatch")