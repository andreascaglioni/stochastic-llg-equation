from math import pi
from os.path import join
import numpy as np

# Imports for dolfinx
from mpi4py import MPI
from dolfinx import default_real_type
from dolfinx.fem import functionspace, Function
from dolfinx.io import XDMFFile
from basix.ufl import element

from functions.compute_utils import get_H1_matrix, get_L2_matrix


# Physics data
alpha = 1.4
T = 1
def m0 (x):  # IC
    m00 = 0.9*x[0]
    m01 = 0.9*x[1]
    m02 = np.sqrt(1. - np.square(m00) - np.square(m01))
    return np.stack((m00, m01, m02))
def g(x):  # space component noise
    sqr = np.square(x[0]) + np.square(x[1])
    C = 0.9
    g0 = C*np.sin(0.5*pi*sqr)*x[0]
    g1 = C*np.sin(0.5*pi*sqr)*x[1]
    g2 = np.sqrt(1. - np.square(g0) - np.square(g1))
    return np.stack((g0, g1, g2))


# Discretization space and time
mesh_filename = join("meshes", "disk_2D_3.xdmf")
fem_order = 1
bdf_order = 1
tau = 0.01
comm = MPI.COMM_SELF

# Pre-processing: Finite elements, time
with XDMFFile(comm, mesh_filename, "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")
Pr = element("Lagrange", msh.basix_cell(), fem_order, dtype=default_real_type)
Pr3 = element("Lagrange", msh.basix_cell(), fem_order, shape=(3,), 
              dtype=default_real_type)
V = functionspace(msh, Pr)
V3 = functionspace(msh, Pr3)
n_tt = int(T/tau)+1
tt = np.linspace(0, T, n_tt)
dtdt = tt[1:] - tt[:-1]

m0h = Function(V3)
m0h.interpolate(lambda x : m0(x))
gh = Function(V3)
gh.interpolate(lambda x : g(x))

ip_V3 = get_H1_matrix(V3)
ip_V = get_L2_matrix(V)
