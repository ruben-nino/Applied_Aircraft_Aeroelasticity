import os
import json
import numpy as np
from numpy import pi

def load_aero_data():
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, '..', 'Configurations', 'wing_TangDowell.json')

    # Normalize the path
    json_path = os.path.abspath(json_path)
    print(json_path)
    with open(json_path, mode="r", encoding="utf-8") as read_file:
        all_data = json.load(read_file)

        aero_data = all_data["aero"]
        aero_data_description = all_data["_aero"]

        return aero_data, aero_data_description

def find_middle_point(x_mesh, y_mesh, z_mesh):
    """
    Given a rectangular mesh in 3D space returns the mesh of the corresponding middle points for each of the rectangles.
    Within the context of VLM this is useful for finding the control points of the chosen mesh.
    The given meshes are assumed to have been generated with 'ij' indexing

    :param x_mesh: x coordinates of the chosen mesh
    :param y_mesh: y coordinates of the chosen mesh
    :param z_mesh: z coordinates of the chosen mesh
    :return: x, y, z coordinates of the middle points of the chosen mesh
    """

    # number of panels in the span- and chord-wise directions, respectively
    n_v = x_mesh.shape[1] - 1
    m_v = x_mesh.shape[0] - 1

    # Assume mesh is aligned with the x and y directions at first, # todo: implement arbitrary orientation later
    x_control = y_control = np.zeros((n_v, m_v))

    for i in range(m_v):
        for j in range(n_v):
            x_control[i, j] = ( x_mesh[i, j] + x_mesh[i + 1, j] ) / 2
            y_control[i, j] = ( y_mesh[i, j] + y_mesh[i, j + 1] ) / 2

    z_control = np.zeros_like(x_control)
    return x_control, y_control, z_control

# coordinate sys with x as downwind, y right wing and z up
def aero_influence_coeff_mats(bound_mesh, wake_mesh):

    # need to pass lines in consistent order
    def biot_savart_Gam_is_1(control_point: np.ndarray, line:np.ndarray):
        corner_1 = line[0,:]
        corner_2 = line[1,:]

        cross_prod = np.cross((control_point - corner_1), (control_point - corner_2))
        sqr_cross  = np.abs(cross_prod)**2

        last_term  = (control_point - corner_1)/np.abs((control_point - corner_1)) \
                     - (control_point - corner_2)/np.abs((control_point - corner_2))
        dot_prod   = np.dot((corner_2-corner_1), last_term)

        result = 1/4/pi * cross_prod / sqr_cross * dot_prod
        return result

    m_v = bound_mesh.shape[0] - 1 # number of panels, which is one less than the no. of nodes in the considered direction
    n_v = bound_mesh.shape[1] - 1
    if n_v != wake_mesh.shape[1]:
        raise ValueError("Wake mesh dimensions in the y direction do not match")
    m_w = wake_mesh.shape[0]  - 1

    A_b = np.zeros()



