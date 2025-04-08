"""Convergence of the TPS (tangent plane scheme) with respect to both mesh size h and time step size dt.
Time step and mesh size must satisfy the condition tau < C h to guarantee stability (CFL condition).
The expected convergence rate of the L^{infty}(0,T, H^1(D)) error is O(h + dt) or
\Vert m - m_{h, \tau}\Vert_{L^{infty}(0,T, H^1(D))} \leq C (h+\tau)
"""

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
from src.set_FE_data import set_FE_data
from src.compute_mesh_elems_area import mesh_elems_area as mea
from src.utils import (
    error_space_time,
    compute_data_nonmatch_interpol,
    export_xdmf,
    compute_rate,
)

# SETTINGS
float_f = lambda x: f"{x:.4e}"  # noqa: E731
np.set_printoptions(formatter={"float_kind": float_f})
comm = MPI.COMM_SELF
np.random.seed(0)

# PARAMETERS & DATA
date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
dir_save = join("simulations", "BDF_FEM_conv_h_" + date + "/")
os.makedirs(dir_save)
print("Saving results in:", dir_save)
shutil.copy(__file__, join(dir_save, "script.txt"))
from data.data_conv_h import data as data_nomsh  # noqa: E402
shutil.copy(join("data", "data_conv_h.py"), join(dir_save, "data.txt"))
tt = data_nomsh["tt"]
idxs_meshes = np.arange(1, 6)
print("Indices meshes:", idxs_meshes, "(last used as reference)")
print("")
n_MC = 1
dim_y = 1

# COMPUTE
dtdt = tt[1:] - tt[:-1]
MC_sample = np.random.randn(dim_y)
np.savetxt(join(dir_save, "MC_sample.csv"), MC_sample, delimiter=",")
data_nomsh["W"] = data_nomsh["W_fun"](tt, MC_sample)

print("Sample reference solution")
ref_mesh_filename = join("data", "meshes", f"disk_2D_{idxs_meshes[-1]}.xdmf")
idxs_meshes = idxs_meshes[:-1]  # remove reference mesh from conv test
with XDMFFile(comm, ref_mesh_filename, "r") as xdmf:
    msh_ref = xdmf.read_mesh(name="Grid")
print(
    "Min mesh h:",
    float_f(sqrt(np.min(mea(msh_ref)))),
    "Min dt:",
    float_f(np.amin(tt[1:] - tt[:-1])),
)
print("")
data_ref = set_FE_data(msh_ref, data_nomsh)
ip_V3_ref = data_ref["ip_V3"]
mm_ref, _, _, is_tt_ref = BDF_FEM_TPS(
    data_ref,
    return_inf_sup=False,
    verbose=int(tt.size / 5),  # log 5 times
)
export_xdmf(msh_ref, mm_ref, tt, join(dir_save, "m_ref.xdmf"))

print("Convergence Test:")
n_dofs_V = np.zeros(len(idxs_meshes), dtype=int)
err_tx = np.zeros(len(idxs_meshes))
hh = np.zeros_like(err_tx)
ddt = np.zeros_like(err_tx)
for i, msh_idx in enumerate(idxs_meshes):
    # Load mesh and compute mesh data
    mesh_filename = join("data", "meshes", f"disk_2D_{msh_idx}.xdmf")
    with XDMFFile(comm, mesh_filename, "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")
    data = set_FE_data(msh, data_nomsh)
    hh[i] = sqrt(np.amin(mea(msh)))
    ddt[i] = data["tt"][1] - data["tt"][0]
    n_dofs_V[i] = data["ip_V"].shape[0]
    print("Compute discrete solution h:", float_f(hh[i]), "dt:", float_f(ddt[i]))
    
    # Solve
    mm, _, _, is_tt = BDF_FEM_TPS(data)

    # Compute error
    data_nonmatch = compute_data_nonmatch_interpol(data_ref["V3"], data["V3"])
    err_tx[i], err_tt = error_space_time(
        mm_ref,
        mm,
        dtdt,
        ip_V3_ref,
        matching_x_spaces=False,
        data_nonmatch=data_nonmatch,
        t_error_type="Linf",
    )
    
    print("L^{\infty}(0, T, H^1(D)) error:", float_f(err_tx[i]))
    print("")

    # Export sequence time errors
    np.savetxt(join(dir_save, f"error_tt_{msh_idx}.csv"), err_tt, delimiter=",")
    export_xdmf(msh, mm, tt, join(dir_save, "m_" + str(msh_idx) + ".xdmf"))

# POST-PROCESS
# print
print("#V:", n_dofs_V, "(Reference:", data_ref["ip_V"].shape[0], ")")
print("h: ", hh)
print("dt:", ddt)
print("Error L^inf(H1):", err_tx)
rate = compute_rate(hh, err_tx)
print("Convergence rate:", rate)

# Export data convergence
A = np.vstack((n_dofs_V, hh, ddt, err_tx)).T
np.savetxt(join(dir_save, "conv_data.csv"), A, delimiter=",", header="#V, h, dt, error")

# Plot
plt.figure("error")
plt.title("L^{\infty}(0, T, H^1(D)) Error")
plt.loglog(hh, err_tx, ".-", label="error")
C = err_tx[0] /hh[0]
plt.loglog(hh, C * hh, "k-", label="C*h)")
C = err_tx[0] / (hh[0]**rate[-1])
plt.loglog(hh, C * hh**rate[-1], "k--", label="C*h^"+float_f(rate[-1]))
plt.legend()
plt.xlabel("h")
plt.savefig(join(dir_save, "conv_error.png"))
plt.show()
