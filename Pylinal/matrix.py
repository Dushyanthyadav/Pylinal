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
        for row in matrix:
            if len(row) != col_len:
                raise ValueError("Matrix rows must have equal length.")
            clean_matrix.append(tuple(row))

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
    
    def __add__(self, other):
        other = self._ensure_dim(self._ensure_matrix(other))
        return Matrix([[a+b for a,b in zip(self_row, other_row)] for self_row, other_row in zip(self.mat, other.mat)])     

    def __radd__(self, other):
        return self.__add__(other)
        
    @property
    def T(self):
        return Matrix(tuple(zip(*self.mat)))

    def lin_trans(self, other):
        other = Vector(other)
        if self.n == other.dim:
            return Vector([sum(a*b for a,b in zip(rows, other)) for rows in self.mat])
        else:
            raise ValueError(f"({self.m}x{self.n}) vs (1x{other.dim}) Dimension mismatch")