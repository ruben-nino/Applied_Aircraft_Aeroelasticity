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
    x_control = np.zeros((m_v, n_v))
    y_control = np.zeros((m_v, n_v))

    for i in np.arange(m_v):
        for j in np.arange(n_v):
            x_control[i, j] = ( x_mesh[i, j] + x_mesh[i + 1, j] ) / 2
            y_control[i, j] = ( y_mesh[i, j] + y_mesh[i, j + 1] ) / 2

    z_control = np.zeros_like(x_control)
    return x_control, y_control, z_control

# coordinate sys with x as downwind, y right wing and z up
def aero_influence_coeff_mats(bound_mesh, wake_mesh, control_points):

    # need to pass lines in consistent order
    def biot_savart_Gam_is_1(control_point: np.ndarray, corner_1, corner_2):
        """
        Calculates the velocity vector in 3D space from the contribution of one section of vortex line with normalized
        vorticity, according to the Biot-Savart law
        :param control_point: Point in which the velocity vector is calculated, given as np.array([x, y, z])
        :param line: Start and end points of the vortex line, given as np.array([[x, y, z],[x, y, z]])
        :return: Velocity vector in the control point
        """

        cross_prod = np.cross((control_point - corner_1), (control_point - corner_2))
        sqr_cross  = np.abs(cross_prod)**2

        last_term  = (control_point - corner_1)/np.abs((control_point - corner_1)) \
                     - (control_point - corner_2)/np.abs((control_point - corner_2))
        dot_prod   = np.dot((corner_2-corner_1), last_term)

        result = 1/4/pi * cross_prod / sqr_cross * dot_prod
        return result

    x_mesh_b, y_mesh_b, z_mesh_b = [bound_mesh[:,:, i] for i in range(3)]
    x_mesh_w, y_mesh_w, z_mesh_w = [wake_mesh[:,:,i] for i in range(3)]
    x_mesh_c, y_mesh_c, z_mesh_c = [control_points[:,:,i] for i in range(3)]

    m_v = x_mesh_b.shape[0] - 1 # number of panels, which is one less than the no. of nodes in the considered direction
    n_v = x_mesh_b.shape[1] - 1
    if n_v != (x_mesh_w.shape[1] -1):
        raise ValueError("Wake mesh dimensions in the y direction do not match")
    m_w = x_mesh_w.shape[0]  - 1
    no_bound_panels = m_v * n_v
    no_wake_panels  = m_w * n_v

    A_b = np.zeros((3, no_bound_panels, no_bound_panels)) # 3 by (m_v * n_v) by (m_v * n_v) since there are the same number of collocation points as bound panels
    A_w = np.zeros((3, no_bound_panels, no_wake_panels)) # not square because the number of wake panels does not match that of the collocation points
                                                            # 3 because we obtain 3 speed components
    # Switch Notation
    x_mesh_b = x_mesh_b.flatten()
    y_mesh_b = y_mesh_b.flatten()
    z_mesh_b = z_mesh_b.flatten()
    x_mesh_c = x_mesh_c.flatten()
    y_mesh_c = y_mesh_c.flatten()
    z_mesh_c = z_mesh_c.flatten()
    x_mesh_w = x_mesh_w.flatten()
    y_mesh_w = y_mesh_w.flatten()
    z_mesh_w = z_mesh_w.flatten()

    for i in np.arange(no_bound_panels): # cycle through the control points
        control_point = np.array([x_mesh_c[i], y_mesh_c[i], z_mesh_c[i]])

        for j in np.arange(no_bound_panels): # cycle through the bound panels

            # select corners of j-th panel
            index_1 = int(j + np.floor(j/n_v))
            index_2 = int(j + np.floor(j / n_v) + 1)
            index_3 = int(j + np.floor(j / n_v) + n_v + 1)
            index_4 = int(j + np.floor(j / n_v) + n_v + 2) # this should be correct, but double check if the results are wrong

            breakpoint()
            corner_1 = np.array([x_mesh_b[index_1], y_mesh_b[index_1], z_mesh_b[index_1]])
            corner_2 = np.array([x_mesh_b[index_2], y_mesh_b[index_2], z_mesh_b[index_2]])
            corner_3 = np.array([x_mesh_b[index_3], y_mesh_b[index_3], z_mesh_b[index_3]])
            corner_4 = np.array([x_mesh_b[index_4], y_mesh_b[index_4], z_mesh_b[index_4]])

            corner_1_sym = corner_1 * np.array([1, -1, 1])
            corner_2_sym = corner_2 * np.array([1, -1, 1])
            corner_3_sym = corner_3 * np.array([1, -1, 1])
            corner_4_sym = corner_4 * np.array([1, -1, 1])

            # need to pass lines in consistent order => clockwise from -z view
            vel_unit_vorticity = biot_savart_Gam_is_1(control_point, corner_1, corner_2)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_2, corner_3)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_3, corner_4)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_4, corner_1)

            # also clockwise (hopefully correct)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_1_sym, corner_4_sym)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_4_sym, corner_3_sym)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_3_sym, corner_3_sym)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_2_sym, corner_1_sym)

            A_b[:, i, j] = vel_unit_vorticity

        for j in np.arange(no_wake_panels):

            # select corners of j-th panel
            index_1 = int(j + np.floor(j / n_v))
            index_2 = int(j + np.floor(j / n_v) + 1)
            index_3 = int(j + np.floor(j / n_v) + n_v + 1)
            index_4 =int(j + np.floor(
                j / n_v) + n_v + 2)  # this should be correct, but double check if the results are wrong

            corner_1 = np.array([x_mesh_w[index_1], y_mesh_w[index_1], z_mesh_w[index_1]])
            corner_2 = np.array([x_mesh_w[index_2], y_mesh_w[index_2], z_mesh_w[index_2]])
            corner_3 = np.array([x_mesh_w[index_3], y_mesh_w[index_3], z_mesh_w[index_3]])
            corner_4 = np.array([x_mesh_w[index_4], y_mesh_w[index_4], z_mesh_w[index_4]])

            corner_1_sym = corner_1 * np.array([1, -1, 1])
            corner_2_sym = corner_2 * np.array([1, -1, 1])
            corner_3_sym = corner_3 * np.array([1, -1, 1])
            corner_4_sym = corner_4 * np.array([1, -1, 1])

            # need to pass lines in consistent order => clockwise from -z view
            vel_unit_vorticity = biot_savart_Gam_is_1(control_point, corner_1, corner_2)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_2, corner_3)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_3, corner_4)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_4, corner_1)

            # also clockwise (hopefully correct)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_1_sym, corner_4_sym)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_4_sym, corner_3_sym)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_3_sym, corner_3_sym)
            vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_2_sym, corner_1_sym)

            A_w[:, i, j] = vel_unit_vorticity

    return A_b, A_w


