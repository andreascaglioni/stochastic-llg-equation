"""Plain tangent plane scheme (TPS) simulation. Appproximate the SLLG dynamics for a fixed instance of the Brownian motion.
Generate .xdmf files for the magnetizaion, its velocity, and the Lagrange multipliers used i the computation. The files can be imported in Paraview for visualization."""

# TODO check that mm is unit-modulus and plot

from math import sqrt
import os
import sys
from datetime import datetime
import shutil
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx.io import XDMFFile

sys.path.insert(0, "./")  # Import from this project
from src.BDF_FEM_TPS import BDF_FEM_TPS
from src.compute_mesh_elems_area import mesh_elems_area as mea
from src.utils import export_xdmf, float_f

# SETTINGS
np.set_printoptions(formatter={"float_kind": float_f})
comm = MPI.COMM_SELF
np.random.seed(0)

# PARAMETERS & DATA
date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
dir_save = join("simulations", "TPS_plain_" + date + "/")
os.makedirs(dir_save)
print("Saving results in:", dir_save)
shutil.copy(__file__, join(dir_save, "script.txt"))
from data.data_direct_simulation import data  # noqa: E402

shutil.copy(join("data", "data_direct_simulation.py"), join(dir_save, "data.txt"))

n_MC = 1
dim_y = 1
MC_sample = np.random.randn(dim_y)
np.savetxt(join(dir_save, "MC_sample.csv"), MC_sample, delimiter=",")
tt = data["tt"]
dt = np.amax(tt[1:] - tt[:-1])
msh = data["msh"]
h = sqrt(np.amin(mea(msh)))

# COMPUTE
print("Sample reference solution")
# Add reference data to dictionary (make a deep copy)
data["W"] = data["W_fun"](tt, MC_sample)
print("Max dt:", float_f(dt))
print("Min mesh size h:", float_f(h))
mm, vv, ll, is_tt_ref = BDF_FEM_TPS(data, verbose=int(tt.size / 10))
export_xdmf(msh, mm, tt, join(dir_save, "m_magnetization.xdmf"))
export_xdmf(msh, vv, tt, join(dir_save, "v_velocity.xdmf"))
export_xdmf(msh, ll, tt, join(dir_save, "l_lagrange_multipliers.xdmf"))
