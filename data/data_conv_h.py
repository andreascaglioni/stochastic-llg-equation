"""Data for h-converngece TPS. Data related to finite elements from mesh and
discretization is removed. The data is set with function set_FE_data() at run time."""

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
    C = 0.6
    g0 = C * np.sin(0.5 * pi * sqr) * x[0]
    g1 = C * np.sin(0.5 * pi * sqr) * x[1]
    g2 = np.sqrt(1.0 - np.square(g0) - np.square(g1))
    # g0 = 0.0 * x[0]
    # g1 = 1.0 + 0.0 * x[1]
    # g2 = 0.0 * x[1]
    return np.stack((g0, g1, g2))


# Discretization space and time
# mesh_filename = join("data/meshes", "disk_2D_3.xdmf")
fem_order = 1
bdf_order = 1
comm = MPI.COMM_SELF

# FE data removed because computed at run-time

# Time stepping data
tau = 1.0e-2
n_tt = int(T / tau) + 1
tt = np.linspace(0, T, n_tt)

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
