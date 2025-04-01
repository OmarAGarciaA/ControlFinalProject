import numpy as np
from scipy.spatial.transform import Rotation as R

# Quaternion class to mimic the SimpleQuaternion from Julia
class SimpleQuaternion(object):
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z])  # Quaternion: [w, x, y, z]
    
    def norm(self):
        return np.linalg.norm(self.q)
    
    def conj(self):
        return SimpleQuaternion(self.q[0], -self.q[1], -self.q[2], -self.q[3])
    
    def __mul__(self, other):
        # Hamilton product (geometric product for quaternions)          #Check the signs on the matrix
        if isinstance(other, SimpleQuaternion):
            # Hamilton product (geometric product for quaternions)
            w1, x1, y1, z1 = self.q
            w2, x2, y2, z2 = other.q
            return SimpleQuaternion(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            )
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return SimpleQuaternion(
                self.q[0] * other,
                self.q[1] * other,
                self.q[2] * other,
                self.q[3] * other
            )

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return SimpleQuaternion(
                self.q[0] * other,
                self.q[1] * other,
                self.q[2] * other,
                self.q[3] * other
            )
        else:
            raise TypeError("Unsupported multiplication between {} and SimpleQuatewrnion".format(type(other)))


    # def __add__(self, other):
    #     return SimpleQuaternion(self.q[0] + other.q[0],
    #                             self.q[1] + other.q[1],
    #                             self.q[2] + other.q[2],
    #                             self.q[3] + other.q[3])

    def __add__(self, other):
        if isinstance(other, SimpleQuaternion):
            return SimpleQuaternion(self.q[0] + other.q[0],
                                    self.q[1] + other.q[1],
                                    self.q[2] + other.q[2],
                                    self.q[3] + other.q[3])
        elif isinstance(other, (int, float)):
            return SimpleQuaternion(self.q[0] + other, self.q[1], self.q[2], self.q[3])
        else:
            raise TypeError("Unsupported addition between SimpleQuaternion and {}".format(type(other)))                                

    def __sub__(self, other):
        return SimpleQuaternion(self.q[0] - other.q[0],
                                self.q[1] - other.q[1],
                                self.q[2] - other.q[2],
                                self.q[3] - other.q[3])
    
    def __truediv__(self, scalar):
        return SimpleQuaternion(self.q[0] / scalar,
                                self.q[1] / scalar,
                                self.q[2] / scalar,
                                self.q[3] / scalar)

    def __div__(self, scalar):
        if isinstance(scalar, (int, float)):
            return SimpleQuaternion(self.q[0] / scalar,
                                    self.q[1] / scalar,
                                    self.q[2] / scalar,
                                    self.q[3] / scalar)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'SimpleQuaternion' and '{}'".format(type(scalar)))

    __div__ = __truediv__

    def __neg__(self):
        # Implement negation for quaternion
        return SimpleQuaternion(-self.q[0], -self.q[1], -self.q[2], -self.q[3])                                

    def __repr__(self):
        return "SimpleQuaternion({0}, {1}, {2}, {3})".format(self.q[0], self.q[1], self.q[2], self.q[3])


class Multivector:
    def __init__(self, val):
        self.val = val        

# Multivector classes   (Some of these are wrong)
# Summation vs Number and number vs summation
# And also with the sub
class EvenGrade(object):
    def __init__(self, val):
        self.val = val

    def __neg__(self):
        return EvenGrade(-self.val)  # Negate the quaternion value


    def __add__(self, other):
        if isinstance(other, EvenGrade):
            return EvenGrade(self.val + other.val)
        elif isinstance(other, (int, float)):
            return EvenGrade(self.val + other)
        else:
            raise TypeError("Unsupported addition between EvenGrade and {}".format(type(other)))

    def __radd__(self, other):
        # For Number + EvenGrade
        if isinstance(other, (int, float)):
            return EvenGrade(self.val + other)
        else:
            raise TypeError("Unsupported addition between {} and EvenGrade".format(type(other)))



    def __sub__(self, other):
        return EvenGrade(self.val - other.val)

    def __mul__(self, other):
        # Handle multiplication between EvenGrade instances
        if isinstance(other, EvenGrade):
            return EvenGrade(self.val * other.val)  # Regular multiplication for EvenGrade * EvenGrade
        elif isinstance(other, OddGrade):
            return OddGrade(self.val * other.val)  # Handle EvenGrade * OddGrade (typically results in EvenGrade)
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return EvenGrade(self.val * other)    
        else:
            raise TypeError("Unsupported type for multiplication")
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return EvenGrade(self.val * other)
        else:
            raise TypeError("Unsupported multiplication between {} and EvenGrade".format(type(other)))

    def __truediv__(self, other):
        return EvenGrade(self.val / other)   

    def __div__(self, other):    
        return EvenGrade(self.val / other)   
      


    def adjoint(self):
        return EvenGrade(self.val.conj())

    def norm(self):
        return self.val.norm()

    def __repr__(self):
        return "EvenGrade({})".format(self.val)

class OddGrade(object):
    def __init__(self, val):
        self.val = val

    def __add__(self, other):
        return OddGrade(self.val + other.val)

    def __neg__(self):
        return OddGrade(-self.val)  # Negate the quaternion value

    def __sub__(self, other):
        return OddGrade(self.val - other.val)

    def __mul__(self, other):
        # Handle multiplication between OddGrade instances
        if isinstance(other, OddGrade):
            return EvenGrade(-self.val * other.val)  # Negating the quaternion product for OddGrade * OddGrade
        elif isinstance(other, EvenGrade):         
            return OddGrade(self.val * other.val)  # Handle OddGrade * EvenGrade
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return OddGrade(self.val * other)       
        else:
            raise TypeError("Unsupported type for multiplication")
        
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return OddGrade(other * self.val)
        else:
            raise TypeError("Unsupported multiplication between {} and OddGrade".format(type(other)))
                
    def __truediv__(self, other):
        return OddGrade(self.val / other)     

    
    def __div__(self, other):    
        return OddGrade(self.val / other)   

    def adjoint(self):
        return OddGrade(self.val.conj())

    def norm(self):
        return self.val.norm()

    def __repr__(self):
        return "OddGrade({})".format(self.val)

    # This is terribly wrong (uncomplete lines)    

# def exp_even(A):
#     a0 = A.val.q[0]
#     a1_2 = -A.val.q[3]
#     a2_3 = -A.val.q[1]
#     a1_3 = A.val.q[2]
#     magc = -(a1_2**2 + a2_3**2 + a1_3**2)
#     a_p = 0
#     a_m = np.sqrt(-magc)
#     b0 = np.cos(a_m)
#     b12 = np.sinh(a_m) * a1_2
#     b13 = np.sinh(a_m) * a1_3
#     b23 = np.sinh(a_m) * a2_3

#     return EvenGrade(SimpleQuaternion(np.exp(a0) * b0, 0, 0, 0))        

#--------REVERSE----------
def adjoint(A):
    if isinstance(A, OddGrade):  # Check if A is an instance of OddGrade
        return OddGrade(-A.val.conj())  # Return the conjugate with a negative sign for OddGrade
    elif isinstance(A, EvenGrade):  # Check if A is an instance of EvenGrade
        return EvenGrade(A.val.conj())  # Return the conjugate for EvenGrade
    else:
        raise TypeError("Unsupported type: {}".format(type(A))) # Raise an error if the type is neither OddGrade nor EvenGrade
    
    
#------------------GRADE PROJECTIONS--------------------------

# def scalar_part(A):
#     if isinstance(A, OddGrade):
#         return EvenGrade(SimpleQuaternion(0, 0, 0, 0))
#     elif isinstance(A, EvenGrade):
#         return EvenGrade(SimpleQuaternion(A.val.real, 0, 0, 0))
def scalar_part(A):
    if isinstance(A, OddGrade):
        # OddGrade has no scalar part, return zero as an EvenGrade
        return EvenGrade(SimpleQuaternion(0, 0, 0, 0))
    elif isinstance(A, EvenGrade):
        # Extract the scalar part (real part of the quaternion) and return it as an EvenGrade
        scalar_value = A.val.q[0]  # Real part (w-component) of the quaternion
        return EvenGrade(SimpleQuaternion(scalar_value, 0, 0, 0))
    else:
        raise TypeError("Input must be either OddGrade or EvenGrade. Got: {}".format(type(A)))

def vector_part(A):
    if isinstance(A, OddGrade):
        parts = [A.val.q[0], A.val.q[1], A.val.q[2], A.val.q[3]]
        return OddGrade(SimpleQuaternion(0, parts[1], parts[2], parts[3]))
    elif isinstance(A, EvenGrade):
        return OddGrade(SimpleQuaternion(0, 0, 0, 0))


def bivector_part(A):
    if isinstance(A, OddGrade):
        return EvenGrade(SimpleQuaternion(0, 0, 0, 0))
    elif isinstance(A, EvenGrade):
        parts = [A.val.q[0], A.val.q[1], A.val.q[2], A.val.q[3]]
        return EvenGrade(SimpleQuaternion(0, parts[1], parts[2], parts[3]))


def trivector_part(A):
    if isinstance(A, OddGrade):
        return OddGrade(SimpleQuaternion(A.val.q[0], 0, 0, 0))
    elif isinstance(A, EvenGrade):
        return OddGrade(SimpleQuaternion(0, 0, 0, 0))

#---------------------Multivector Components----------------------


def multivector_coeffs(A):
    if isinstance(A, EvenGrade):
        parts = [A.val.q[0], A.val.q[1], A.val.q[2], A.val.q[3]]
        # Extract components from quaternion
        scalar_part = float(parts[0])  # Scalar part
        bivector_12 = float(parts[1])   # Bivector part for 1-2 plane
        bivector_23 = float(parts[2])   # Bivector part for 2-3 plane
        bivector_31 = float(parts[3])   # Bivector part for 3-1 plane
        
        # # Return the coefficients tuple as per the order from the Julia code
        # return (scalar_part, 0, 0, 0, -bivector_12, -bivector_23, -bivector_31, 0)

        return (float(parts[0]),0,0,0,-float(parts[3]),-float(parts[1]),-float(parts[2]),0)

    elif isinstance(A, OddGrade):
        
        parts = [A.val.q[0], A.val.q[1], A.val.q[2], A.val.q[3]]

        # Extract components from quaternion
        vector_x = float(parts[1])
        vector_y = float(parts[2])
        vector_z = float(parts[3])
        scalar_part = float(parts[0])  # Trivector part (as scalar here)
        
        # Return the coefficients tuple as per the order from the Julia code
        return (0, vector_x, vector_y, vector_z, 0, 0, 0, scalar_part)

    else:
        raise TypeError("Argument must be of type EvenGrade or OddGrade")


# ---------------------------- Exponential and log ------------------
def exp_even(A):
    parts = [A.val.q[0], A.val.q[1], A.val.q[2], A.val.q[3]]
    a0 = parts[0]  # Scalar part
    a12 = -parts[3]  # Bivector part 1-2 plane
    a23 = -parts[1]  # Bivector part 2-3 plane
    a13 = parts[2]  # Bivector part 3-1 plane

    #nOT QUITE SURE IF THIS IS THE RIGHT ORDER
    
    # Magnitude squared of the bivector part
    aS = -a12**2 - a23**2 - a13**2
    magc = -aS
    aP = 0
    aM = np.sqrt(-aS)

    b0 = np.cos(aM) * np.cosh(aP)
    b12 = np.cosh(aP) * np.sin(aM) * (aM * a12) + np.sinh(aP) * np.cos(aM) * (aP * a12)
    b13 = -np.cosh(aP) * np.sin(aM) * (-aM * a13) + np.sinh(aP) * np.cos(aM) * (aP * a13)
    b23 = np.cosh(aP) * np.sin(aM) * (aM * a23) + np.sinh(aP) * np.cos(aM) * (aP * a23)

    alpha = np.exp(a0)
    result = EvenGrade(SimpleQuaternion(alpha * b0, 0, 0, 0))

    if magc > 0:
        result = EvenGrade(SimpleQuaternion(alpha * b0, -alpha * b23 / magc, alpha * b13 / magc, -alpha * b12 / magc))

    return result

# Exponential function for the EvenGrade
def exp(A):
    if isinstance(A, EvenGrade):
        return exp_even(A)
    elif isinstance(A, OddGrade):
        # Exponential for odd grade (not provided in original Julia code but should be handled similarly)
        # For now, we return NotImplemented as it's more complex and depends on your exact use case.
        raise NotImplementedError("Exponential for OddGrade is not implemented")
    else:
        raise TypeError("Argument must be of type EvenGrade or OddGrade")

# Logarithm of EvenGrade
def log_even(A):
    # Extract parts from the quaternion representation of A
    parts = [A.val.q[0], A.val.q[1], A.val.q[2], A.val.q[3]]
    norm_A = np.linalg.norm(parts)  # Euclidean norm of the quaternion

    # Extract scalar and bivector parts
    scalar_part = parts[0]  # Assuming parts[0] is the scalar part
    bivector_part = parts[1:]  # Assuming parts[1:] are the bivector components (x, y, z)

    if np.linalg.norm(bivector_part) > 0:
        # Calculate bivector logarithm part
        log_norm = np.arccos(scalar_part / norm_A) / np.linalg.norm(bivector_part)
        bivector_log = np.array(bivector_part) * log_norm
        result = EvenGrade(SimpleQuaternion(np.log(norm_A), *bivector_log))
    else:
        # If the bivector part is zero, return only the scalar logarithm
        result = EvenGrade(SimpleQuaternion(np.log(norm_A), 0, 0, 0))
    
    return result

# Logarithm function for the EvenGrade
def log(A):
    if isinstance(A, EvenGrade):
        return log_even(A)
    elif isinstance(A, OddGrade):
        # Logarithm for odd grade (not provided in original Julia code but should be handled similarly)
        # For now, we return NotImplemented as it's more complex and depends on your exact use case.
        raise NotImplementedError("Logarithm for OddGrade is not implemented")
    else:
        raise TypeError("Argument must be of type EvenGrade or OddGrade")


# -----------------DUAL -----------------------------
# Double check, if may not be outputing Evengrades or Oddgrades as it should (Original may be wrong as well)
def dual(A):
    if isinstance(A, OddGrade):
        # If A is an odd-grade multivector, return an even-grade multivector
        neg_quaternion = OddGrade(SimpleQuaternion(-1, 0, 0, 0))
        return A * neg_quaternion
    
    elif isinstance(A, EvenGrade):
        # If A is an even-grade multivector, return an odd-grade multivector
        neg_quaternion = OddGrade(SimpleQuaternion(-1, 0, 0, 0))
        return A * neg_quaternion


# ------------INVERSE-------(Original not tested)
def inv(A):
    if isinstance(A, OddGrade):
        # If A is an odd-grade multivector, return an odd-grade inverse
        conj_val = A.val.conj()
        norm_squared = A.val.norm() ** 2
        inverse_val = (-conj_val)/norm_squared
        # inverse_val = SimpleQuaternion(-conj_val.scalar / norm_squared,
        #                                -conj_val.i / norm_squared,
        #                                -conj_val.j / norm_squared,
        #                                -conj_val.k / norm_squared)
        return OddGrade(inverse_val)

    elif isinstance(A, EvenGrade):
        # If A is an even-grade multivector, return an even-grade inverse
        conj_val = A.val.conj()
        norm_squared = A.val.norm() ** 2
        inverse_val = conj_val/norm_squared
        # inverse_val = SimpleQuaternion(conj_val.scalar / norm_squared,
        #                                conj_val.i / norm_squared,
        #                                conj_val.j / norm_squared,
        #                                conj_val.k / norm_squared)
        return EvenGrade(inverse_val)


# --------------------- DOT ----------------------

def dot(u, v):
    if isinstance(u, OddGrade) and isinstance(v, OddGrade):  #check to ensure that both inputs, u and v, are instances of the OddGrade class.
        # Extract components of u and v using get_parts
        u_0, u_1, u_2, u_3 = [u.val.q[0], u.val.q[1], u.val.q[2], u.val.q[3]]
        v_0, v_1, v_2, v_3 = [v.val.q[0], v.val.q[1], v.val.q[2], v.val.q[3]]
        
        # Compute the dot product (assumes trivector part is zero)
        return u_1 * v_1 + u_2 * v_2 + u_3 * v_3
    else:
        raise TypeError("Both u and v must be instances of OddGrade.")


# ---------------WEDGE ---------------------------

def wedge(u, v):
    if isinstance(u, OddGrade) and isinstance(v, OddGrade):
        # Extract components of u and v using get_parts
        u_0, u_1, u_2, u_3 = [u.val.q[0], u.val.q[1], u.val.q[2], u.val.q[3]]
        v_0, v_1, v_2, v_3 = [v.val.q[0], v.val.q[1], v.val.q[2], v.val.q[3]]
        
        # Compute the components of the wedge product
        scalar_part = 0
        i_part = (u_3 * v_2 - u_2 * v_3)
        j_part = (u_1 * v_3 - u_3 * v_1)
        k_part = (u_2 * v_1 - u_1 * v_2)
        
        # Return the result as an even-grade multivector
        return EvenGrade(SimpleQuaternion(scalar_part, i_part, j_part, k_part))
    else:
        raise TypeError("Both u and v must be instances of OddGrade.")


# ----------COMMUTATOR --------

def commutator(A, B):
    if isinstance(A, EvenGrade) and isinstance(B, EvenGrade):
        # Compute the commutator as (1/2) * (A * B - B * A)
        return EvenGrade(0.5 * (A * B - B * A))
    elif isinstance(A, EvenGrade) and isinstance(B, OddGrade):
        return OddGrade(0.5 * (A * B - B * A))
    elif isinstance(A, OddGrade) and isinstance(B, EvenGrade):
        return OddGrade(0.5 * (A * B - B * A))
    elif isinstance(A, OddGrade) and isinstance(B, OddGrade):
        return EvenGrade(0.5 * (A * B - B * A))
    else:
        raise TypeError("Both A and B must be instances of Multivector.")


#-------------POWER-----------------

# Probably I need to remove the OddGrade() and EvenGrade() stuff from this functions
#If the power is even, then the Multivector is even, otherwise the Multivector is odd.
def power(A, p):
    if isinstance(A, OddGrade):
        return power_odd(A, p)
    elif isinstance(A, EvenGrade):
        return power_even(A, p)

# The power function for OddGrade
def power_odd(u, p):
    if u.val.norm() < 1e-15 and p != 0:
        return (ONE*0)  # Left to test
    elif p == 0:
        return (ONE)  
    else:
        return OddGrade((u.val.norm()) ** p * u.val) if p % 2 != 0 else EvenGrade((u.val.norm()) ** p)

# The power function for EvenGrade
def power_even(A, p):
    if (A.val.norm()) < 1e-15 and p != 0:
        return (0)
    elif p == 0:
        return (ONE)  
    else:
        bivect = bivector_part(A)
        if bivect.norm() < 1e-15:
            return (A.val.q[0] ** p * ONE)  # Scalar power case
        else:
            return (exp(p * log(A)))
          

# ---------------Testing

ONE = EvenGrade(SimpleQuaternion(1, 0, 0, 0))  # Unit element in EvenGrade
e12 = EvenGrade(SimpleQuaternion(0, 0, 0, -1))  # Bivector e12
e23 = EvenGrade(SimpleQuaternion(0, -1, 0, 0))  # Bivector e23
e31 = EvenGrade(SimpleQuaternion(0, 0, -1, 0))  # Bivector e31

e1 = OddGrade(SimpleQuaternion(0, 1, 0, 0))  # Vector e1 (odd grade)
e2 = OddGrade(SimpleQuaternion(0, 0, 1, 0))  # Vector e2 (odd grade)
e3 = OddGrade(SimpleQuaternion(0, 0, 0, 1))  # Vector e3 (odd grade)

e123 = OddGrade(SimpleQuaternion(1, 0, 0, 0))  # Trivector e123 (odd grade)
print('aqui')
print(e12)
print(type(e12))  #Para python2.7 e12 se convirtio en una instance
print('aqui')

#Even * even
print('Even * even')
print(e12*e23)

#odd * odd
print('odd * odd')
print(e1*e2)
print(e2*e1)
print(e2*e3)
print(e3*e1)

#odd*even
print('odd * even')
print(e1*e31)
print(e1*e12)


#even*odd
print('even *odd')
print(e31 * e1)

#even addition
print('even addition')
print(1 + e12)

#odd adition
print('odd addition')
print(e1+e123)

#scalar adition
print('scalar addition')
print(1 + e12 + 1)


#Grade Projections
print('Grades Projections')
print(scalar_part(e1+e123))
print(scalar_part(1+e12))
print(vector_part(1+e12))
print(vector_part(e1+e123))
print(bivector_part(1+e12+e23))
print(bivector_part(e1+e123))
print(trivector_part(1+e12))
print(trivector_part(e1+e123))

#reverse
print('reverse')
print(adjoint(ONE))   # How comes it is defined as 1' in julia?
print(ONE) 
print(adjoint(ONE)==ONE) 

print(adjoint(e1))
print((e1))
print(adjoint(e1)==e1)   # Do I want to know if this are "python conceptually" equal, or do I want to know if these throw the same value?

print(adjoint(e2))
print((e2))
print(adjoint(e2)==e2)

print(adjoint(e3))
print((e3))
print(adjoint(e3)==e3)

print(adjoint(e12))
print((-e12))
print(adjoint(e12)==-e12)

print(adjoint(e23))
print((-e23))
print(adjoint(e23)==-e23)

print(adjoint(e31))
print((-e31))
print(adjoint(e31)==-e31)

print(adjoint(e123))
print((-e123))
print(adjoint(e123)==-e123)

#multivector
print('multivector')
print(multivector_coeffs(ONE)) 
print(multivector_coeffs(e1))
print(multivector_coeffs(e2))
print(multivector_coeffs(e3))
print(multivector_coeffs(e12))
print(multivector_coeffs(e23))
print(multivector_coeffs(e31))
print(multivector_coeffs(e123))

#Scalar mult and div
print('Scalar mult and div')
print(e12*(np.pi/2))
print((np.pi/2)*e12)

print('aca')
print(e12)
print(type(e12))
print('aca')

print(e12/2)
print(e1/2)


#Even exponential
print('Even exponential')
print(exp(e12*(np.pi/2)))

#Plane Rotation
print('Plane Rotation')
print(e1*exp(e12*(np.pi/2)))
print(exp(e12*(np.pi/2))*e1)

#Rotations   # I believe exponential is working now
print('Rotations')
rotation = exp(-e23*(np.pi/4))*e1*exp(e23*(np.pi/4))
print(rotation)

rotation2 = exp(-e23*(np.pi/4))*e2*exp(e23*(np.pi/4))
print(rotation2)

#Even logarithm   
print('Even logarithm')
B = e12*np.pi/4
print(B)
R = exp(B)
print(R)
print(log(R))

#Inverse
print('Inverse')
print(inv(ONE)) 
print(inv(e1))
print(inv(e2))
print(inv(e3))
print(inv(e12))
print(inv(e23))
print(inv(e31))
print(inv(e123))
print(inv(1/np.sqrt(2)+1/np.sqrt(2)*e12))

#vector dot product
print('vector dot product')
print(dot(e1,e2))
print(dot(e1,e1))

#vector wedge
print('vector wedge')
print(wedge(e1,e2))
print(wedge(e2,e1))
print(wedge(e1,e3))
print(wedge(e3,e2))

 #Dual
print('dual')
print(dual(ONE))
print(dual(e1))
print(dual(e2))
print(dual(e3))
print(dual(e12))
print(dual(e23))
print(dual(e31))
print(dual(e123))

#commutator   #still wrong outputing Even(Even)
print('commutator')
print(commutator(e1,e2))

#vector powers (integer multiples of 1/2) 
print('vector powers')
print(power(e1,2))
print(power(e1,3))
print(power(e1,3/2))
print(power(e1,-1))
print(power(e1,-1/2))
print(power(0.5*e1,2))
print(power(0.5*e1,-1/2))
print(power(0.5*e1,1/2))
print(power(0.5*e1,-1/2)*power(0.5*e1,1/2))
print(power(2*e1,-1/2))
print(power(2*e1,1/2))

#bivector powers 
print('bivector powers ')
R=power((e1*e2),(-1/2))
print(R)    #In the power function do we have to specify when the output its an even grade or an odd grade?
R=power((e2*e1),(-1/2))
print(R)

print(multivector_coeffs(R))

jth = -2*log(R)
print(jth)
print(multivector_coeffs(jth))


