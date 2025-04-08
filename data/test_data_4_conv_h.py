"""as example 1, but 
- remove mesh and mesh dependent quantitities to set them at run-time.
- time step tau = 5.e-3"""

from math import pi
import numpy as np
from mpi4py import MPI
from src.parametric_W import param_LC_W

# Physics data
alpha = 1.4
T = 1

def W_fun(t, y):  # Random field (wiener process) as a function of time and parameter
    return param_LC_W(t, y, T)

def m0(x):  # IC
    m00 = 0.9 * x[0]
    m01 = 0.9 * x[1]
    m02 = np.sqrt(1.0 - np.square(m00) - np.square(m01))
    return np.stack((m00, m01, m02))


def g(x):  # space component noise
    sqr = np.square(x[0]) + np.square(x[1])
    C = 0.9
    g0 = C * np.sin(0.5 * pi * sqr) * x[0]
    g1 = C * np.sin(0.5 * pi * sqr) * x[1]
    g2 = np.sqrt(1.0 - np.square(g0) - np.square(g1))
    return np.stack((g0, g1, g2))


# Discretization space and time
# mesh_filename = join("data/meshes", "disk_2D_3.xdmf")
fem_order = 1
bdf_order = 1
tau = 1.e-2  # Halve time step compared to previous simulation (1.e-2)
comm = MPI.COMM_SELF
n_tt = int(T / tau) + 1
tt = np.linspace(0, T, n_tt)

# Data removed because computed in h-refinement loop:
# with XDMFFile(comm, mesh_filename, "r") as xdmf:
#     msh = xdmf.read_mesh(name="Grid")
# Pr = element("Lagrange", msh.basix_cell(), fem_order, dtype=default_real_type)
# Pr3 = element("Lagrange", msh.basix_cell(), fem_order, shape=(3,),
#               dtype=default_real_type)
# V = functionspace(msh, Pr)
# V3 = functionspace(msh, Pr3)
# m0h = Function(V3)
# m0h.interpolate(lambda x : m0(x))
# gh = Function(V3)
# gh.interpolate(lambda x : g(x))
# ip_V3 = get_H1_matrix(V3)
# ip_V = get_L2_matrix(V)

# Import the following!
data = {
    "m0h": None,
    "alpha": alpha,
    "gh": None,
    "W_fun": W_fun,
    "tt": tt,
    "bdf_order": bdf_order,
    "msh": None,
    "V3": None,
    "V": None,
    "ip_V3": None,
    "ip_V": None,
    "fem_order": fem_order,
    "m0": m0,
    "g": g,
}
