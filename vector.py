class Vector:

    __slots__ = ('vec', 'dim')
    
    def __init__(self, vector):
        self.vec = tuple(vector)
        self.dim = len(vector)
        
    def __repr__(self):
        return f"Vector({self.vec})"

    def __add__(self, other):
        if isinstance(other, (tuple, list)):
            other = Vector(other)
        if isinstance(other, Vector):
            if self.dim == other.dim:
                return Vector([a+b for a, b in zip(self.vec, other.vec)])
            raise ValueError(f"Incompitable Dimensions {self.dim} vs {other.dim}")
        return NotImplemented
        
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (tuple, list)):
            other = Vector(other)
        if isinstance(other, (int, float)):
            return Vector([other*i for i in self.vec])
        if isinstance(other, Vector):
            if other.dim != self.dim:
                raise ValueError(f"Incompitable Dimensions {self.dim} vs {other.dim}")
            return Vector([a*b for a, b in zip(self.vec, other.vec)])
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other):
        if isinstance(other, (tuple,list)):
            other = Vector(other)
        if isinstance(other, Vector):
            if other.dim != self.dim:
                raise ValueError(f"Incompitable Dimensions {self.dim} vs {other.dim}")
            return sum([a*b for a,b in zip(self.vec, other.vec)])
        else: 
            return NotImplemented

    def cross(self, other):
        if isinstance(other, (tuple, list)):
            other = Vector(other)
        if isinstance(other, Vector):
            if other.dim == 3 and self.dim == 3:
                a = (self.vec[1]*other.vec[2])-(self.vec[2]*other.vec[1])
                b = (self.vec[0]*other.vec[2])-(self.vec[2]*other.vec[0])
                c = (self.vec[0]*other.vec[1])-(self.vec[1]*other.vec[0])
                return Vector([a, -b, c])
            else: 
                raise ValueError(f"Incompitable Dimensions {self.dim} vs {other.dim}")
        else:
            return NotImplemented
            