import ufl
import dolfinx.fem
def compute_h_max(msh):
    """Compute maximum element size of 2D mesh.

    Args:
        msh (dolfinxmesh.Mesh): Mesh
    
    Returns:
        float: Maximum element size
    """

    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    cell_area_form = dolfinx.fem.form(v*ufl.dx)
    cell_area = dolfinx.fem.assemble_vector(cell_area_form)
    return cell_area.array.max()