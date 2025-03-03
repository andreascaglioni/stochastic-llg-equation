"""Example of use of the module bdfllg_func to sample sample paths of the random
magnetization from SLLG equation.
The space domain is an axis-aligned box of sides length 1, 1, 0.1
"""

from math import pi
import os
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.fem import Constant, functionspace, Function
from dolfinx.io import XDMFFile, gmshio

from sample_SLLG.parametric_W import param_LC_Brownian_motion
from high_order_bdf_sllg_funs import high_order_bdf_sllg_functions


"""Sample the parametric LLG equation. 
Tangent plane scheme: Solve for velocities magnetization (linear problem).
Impose tangent palne constraint with Lagrange multiplier.
Approximate in time with order 1 BDF and in space with order 1 FEM 
discretization.

Args:
    y (numpy.ndarray[float]): 1D array of real parameters and any length
    
Returns:
    numpy.ndarray[float]: 1D array of dofs of the solution correspodning to
    input parameter.
"""

################### PARAMETERS ###################
# Physics  
T = 1 # final time
alpha = 1.4
# coefficient g (space compponent noise)
# g_values = [0,1,0]  # for constant g (see also gh below)
def g(x):
    sqr = np.square(x[0]) + np.square(x[1])
    C = 1.
    # g0 = C * np.exp(-1./(1-sq_d)) * x[0]
    # g1 = C * np.exp(-1./(1-sq_d)) * x[1]
    g0 = 0.9*np.sin(0.5*pi*sqr)*x[0]
    g1 = 0.9*np.sin(0.5*pi*sqr)*x[1]
    g2 = np.sqrt(1. - np.square(g0) - np.square(g1))  # C**2 * np.exp(-2./(1-sq_d)) * sq_d
    return np.stack((g0, g1, g2))

def m0 (x):
    m00 = 0.9*x[0]
    m01 = 0.9*x[1]
    m02 = np.sqrt(1. - np.square(m00) - np.square(m01))
    return np.stack((m00, m01, m02))
yy =  np.random.normal(0, 1, (1, 1))    

# Numerics 
mesh_filename = os.path.join("meshes_disk", "disk_2D_4.xdmf")
fem_order = 1  # FEM degree
tau = 0.01  # time step
bdf_order = 1  # BDF order
# set_log_level(31)  # dolfin logging TODO find the one for dolfinx
comm = MPI.COMM_SELF  # compute in serial; MPI.COMM_WORLD for parallel

################### COMPUTE ###################
# Load mesh
with XDMFFile(comm, mesh_filename, "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")
# Define function spaces
Pr = element("Lagrange", msh.basix_cell(), fem_order, dtype=default_real_type)
Pr3 = element("Lagrange", msh.basix_cell(), fem_order, shape=(3,), 
              dtype=default_real_type)
V = functionspace(msh, Pr)
V3 = functionspace(msh, Pr3)
tt = np.linspace(0, T, int(T/tau)+1)  # Time steps
W = param_LC_Brownian_motion(tt, yy, T=1)  # sample path Wiener process
m0h = Function(V3)
m0h.interpolate(lambda x : m0(x))
gh = Function(V3)
gh.interpolate(lambda x : g(x))
# gh = Constant(msh, (PETSc.ScalarType(g_values[0]), 
                #    PETSc.ScalarType(g_values[1]), 
                #    PETSc.ScalarType(g_values[2])))  # g constant (see above)
    
# Run BDF-FEM solver
mvec, vvec, lvec=high_order_bdf_sllg_functions(bdf_order, alpha, T, tau, m0h,
                                               V3, V, W, gh, msh, verbose=True)

################### EXPORT ###################
xdmf = XDMFFile(msh.comm, "test_sllg_"+str(yy)+".xdmf", "w")
xdmf.write_mesh(msh)
for i in range(len(mvec)):
    mvec[i].name = "m"
    xdmf.write_function(mvec[i], tt[i])
xdmf.close()

