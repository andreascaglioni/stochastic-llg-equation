from math import pi
from os.path import join
import numpy as np

# Imports for dolfinx
from mpi4py import MPI
from dolfinx.io import XDMFFile
from src.set_FE_data import set_FE_data

from src.parametric_W import param_LC_W
# Physics data
alpha = 1.4
T = 1
def W_fun(tt, yy):
    return param_LC_W(tt, yy, T= T)


def m0(x):  # IC
    m00 = 0.9 * x[0]
    m01 = 0.9 * x[1]
    # m00 = 0.0 * x[0]
    # m01 = 0.0 * x[1]
    m02 = np.sqrt(1.0 - np.square(m00) - np.square(m01))
    return np.stack((m00, m01, m02))


def g(x):  # space component noise
    sqr = np.square(x[0]) + np.square(x[1])
    C = 0.6
    g0 = C * np.sin(0.5 * pi * sqr) * x[0]
    g1 = C * np.sin(0.5 * pi * sqr) * x[1]
    # g0 = x[0]
    # g1 = 0.*x[1]
    g2 = np.sqrt(1.0 - np.square(g0) - np.square(g1))
    return np.stack((g0, g1, g2))


# Discretization space and time
bdf_order = 1
mesh_filename = join("data", "meshes", "disk_2D_3.xdmf")
fem_order = 1
comm = MPI.COMM_SELF

# Finite elements data
with XDMFFile(comm, mesh_filename, "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")

# No time stepping data. To be determined at run-time
# n_tt = int(T/tau)+1
# tt = np.linspace(0, T, n_tt)
# dtdt = tt[1:] - tt[:-1]

data = {"m0": m0,
        "g": g,
        "fem_order": fem_order,
        "bdf_order": bdf_order,
        "T": T,
        "alpha": alpha,
        "W_fun": W_fun
        }
data = set_FE_data(msh, data)
