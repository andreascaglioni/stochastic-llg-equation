from math import sqrt
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import sys
from dolfinx.io import XDMFFile
from mpi4py import MPI
from math import pi

sys.path.insert(0, "./")
from sample_SLLG.BDF_FEM_TPS import BDF_FEM_TPS
from sample_SLLG.set_FE_data import set_FE_data
from sample_SLLG.parametric_W import param_LC_W
from pod_tps.utils import export_xdmf

##################################################################

# SETTINGS 
comm = MPI.COMM_SELF

# DATA
def m0(x):  # IC
    m00 = 0.7*x[0]
    m01 = 0. * x[1]
    m02 = np.sqrt(1.0 - np.square(m00) - np.square(m01))
    return np.stack((m00, m01, m02))
def g(x):
    sqr = np.square(x[0]) + np.square(x[1])
    C = 0.9
    # g0 = C * np.sin(0.5 * pi * sqr) * x[0]
    # g1 = C * np.sin(0.5 * pi * sqr) * x[1]
    # g2 = np.sqrt(1.0 - np.square(g0) - np.square(g1))

    g0 = 0.*x[0]
    g1 = 1.+0.*x[0]
    g2 = 0.*x[0]
    return np.stack((g0, g1, g2))

def H(x):  # IC
    H0 = 0. * x[0]
    H1 = 0. * x[1]
    H2 = -10. + 0. * x[2]
    return np.stack((H0, H1, H2))
T = 2
n_tt = 251
tt = np.linspace(0, T, n_tt)
np.random.seed(0)
yy = np.random.normal(0,1, (1000,))
W = 100*param_LC_W(tt, yy, T=T)

data = {
    "m0": m0,
    "alpha": 1.4,
    "g": g,
    "tt": tt,
    "bdf_order": 1,
    "fem_order": 1,
    "W": W
}
mesh_filename = join("data", "meshes", "disk_2D_4.xdmf")
with XDMFFile(comm, mesh_filename, "r") as xdmf:
    msh = xdmf.read_mesh(name="Grid")
data = set_FE_data(msh, data)

# COMPUTE
mm, _, _, is_tt = BDF_FEM_TPS(data, verbose=int(tt.size / 5), H_input=H)


export_xdmf(msh, mm, tt, "ex_bdf_tps.xdmf")