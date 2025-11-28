import math
class Vector:

    __slots__ = ('vec', 'dim')
    
    def __init__(self, vector):
        self.vec = tuple(float(x) for x in vector)
        self.dim = len(vector)
        
    def __repr__(self):
        return f"Vector([{' '.join(str(x) for x in self.vec)}])"

    def __iter__(self):
        return iter(self.vec)

    def __len__(self):
        return self.dim

    def __getitem__(self, index):
        return self.vec[index]

    def _ensure_vector(self, other):
        if isinstance(other, Vector):
            return other
        if isinstance(other, (list, tuple)):
            return Vector(other)
        raise TypeError(f"{type(other)} Can not be Vector")

    def _ensure_dim(self, other):
        if self.dim == other.dim:
            return other
        raise ValueError(f"{self.dim} vs {other.dim} Dimension mismatch")

    def __neg__(self):
        return Vector([-x for x in self])
        
    @property
    def magnitude(self):
        return self.__abs__()

    def __eq__(self, other):
        other = self._ensure_dim(self._ensure_vector(other))        
        return self.vec == other.vec

    def __abs__(self):
        return math.sqrt(sum(x**2 for x in self))

    def __add__(self, other):
        other = self._ensure_dim(self._ensure_vector(other))
        return Vector([a+b for a, b in zip(self.vec, other.vec)])
        
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._ensure_dim(self._ensure_vector(other))
        return Vector([a-b for a, b in zip(self.vec, other.vec)])

    def __rsub__(self, other):
        other = self._ensure_dim(self._ensure_vector(other))    
        return other.__sub__(self)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector([i*other for i in self.vec])
        other = self._ensure_dim(self._ensure_vector(other))
        return Vector([a*b for a, b in zip(self.vec, other.vec)])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by Zero")
            return Vector([ i / other for i in self])
        else:
            raise TypeError(f"Vector can only be divided by scalar. Given {type(other)}")
            
    @property
    def unit(self):
        magnitude = abs(self)
        if magnitude == 0:
            raise ValueError("Cannot Normalize a Zero Vector.")
        return self/magnitude

    def dot(self, other):
        other = self._ensure_dim(self._ensure_vector(other))
        return sum([a*b for a,b in zip(self.vec, other.vec)])

    def cross(self, other):
        other = self._ensure_dim(self._ensure_vector(other))
        if other.dim != 3:
            raise ValueError(f"Dimension {other.dim}. Cross product is only possible for vector of dim 3")
        a = (self.vec[1]*other.vec[2])-(self.vec[2]*other.vec[1])
        b = (self.vec[0]*other.vec[2])-(self.vec[2]*other.vec[0])
        c = (self.vec[0]*other.vec[1])-(self.vec[1]*other.vec[0])
        return Vector([a, -b, c])

    def outer_product(self, other):
        other = self._ensure_vector(other)
        from matrix import Matrix
        return  Matrix([[a*b for b in other] for a in self])
    
    def angle(self, other, degrees=False):
        other = self._ensure_dim(self._ensure_vector(other))
        dot_prod = self.dot(other)
        mags = self.magnitude * other.magnitude
        cos_theta = max(min(dot_prod / mags, 1.0), -1.0)

        radians = math.acos(cos_theta)
        return math.degrees(radians) if degrees else radians

    def project_onto(self, other):
        other = self._ensure_vector(other)
        b_unit = other.unit
        return b_unit*self.dot(b_unit)