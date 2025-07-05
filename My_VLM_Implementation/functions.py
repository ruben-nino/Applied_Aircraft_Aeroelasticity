import os
import json
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from numba import njit

def load_aero_data():
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, '..', 'Configurations', 'wing_TangDowell.json')

    # Normalize the path
    json_path = os.path.abspath(json_path)
    # print(aero_mats_path)
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

# coordinate sys with x as downwind, y right wing and z up # todo: add njit decorator to this function (performance critical)
@njit
def point_from_mesh(mesh, index):
    point = np.array([mesh[0, index], mesh[1, index], mesh[2, index]])
    return point

@njit
def calc_velocity_panel_and_sym(num_panels_spanwise, control_point_index, panel_index, control_points_mesh, corner_points_mesh):

    # select control point and corresponding symmetric twin
    control_point = point_from_mesh(control_points_mesh, control_point_index)
    control_point_sym = control_point * np.array([1, -1, 1]) # x-z plane symmetry

    # select corners of j-th panel
    index_1 = int(panel_index + np.floor(panel_index / num_panels_spanwise))
    index_2 = int(panel_index + np.floor(panel_index / num_panels_spanwise) + 1)
    index_4 = int(panel_index + np.floor(panel_index / num_panels_spanwise) + num_panels_spanwise + 1)
    index_3 = int(panel_index + np.floor(panel_index / num_panels_spanwise) + num_panels_spanwise + 2)  # this should be
    # correct, but double check if the results are wrong
    # the results were, indeed, wrong due to this line (_4 instead of 3)

    corner_1 = point_from_mesh(corner_points_mesh, index_1)
    corner_2 = point_from_mesh(corner_points_mesh, index_2)
    corner_3 = point_from_mesh(corner_points_mesh, index_3)
    corner_4 = point_from_mesh(corner_points_mesh, index_4)

    # need to pass lines in consistent order => clockwise from -z view
    vel_unit_vorticity = biot_savart_Gam_is_1(control_point, corner_1, corner_2)
    vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_2, corner_3)
    vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_3, corner_4)
    vel_unit_vorticity += biot_savart_Gam_is_1(control_point, corner_4, corner_1)

    vel_unit_vorticity += biot_savart_Gam_is_1(control_point_sym, corner_1, corner_2)
    vel_unit_vorticity += biot_savart_Gam_is_1(control_point_sym, corner_2, corner_3)
    vel_unit_vorticity += biot_savart_Gam_is_1(control_point_sym, corner_3, corner_4)
    vel_unit_vorticity += biot_savart_Gam_is_1(control_point_sym, corner_4, corner_1)

    return vel_unit_vorticity

@njit
def biot_savart_Gam_is_1(control_point: np.ndarray, line_end_1, line_end_2):
    """
    Calculates the velocity vector in 3D space from the contribution of one section of vortex line with normalized
    vorticity, according to the Biot-Savart law
    :param control_point: Point in which the velocity vector is calculated, given as np.array([x, y, z])
    :param line_end_1: Start and end points of the vortex line, given as np.array([[x, y, z],[x, y, z]])
    :return: Velocity vector in the control point
    """

    cross_prod = np.cross((control_point - line_end_1), (control_point - line_end_2))
    sqr_cross  = (cross_prod**2).sum()

    last_term  = (control_point - line_end_1) / np.linalg.norm((control_point - line_end_1)) \
                 - (control_point - line_end_2) / np.linalg.norm((control_point - line_end_2))
    dot_prod   = np.dot((line_end_2 - line_end_1), last_term)

    result = 1/4/pi * cross_prod / sqr_cross * dot_prod

    # Remove since numba does not like it
    # if np.isnan(result).any():
    #     print(f"There are nan entries in the velocity vector obtained from Biot-Savart !!!")
    #     plt.plot(line_end_1[0], line_end_1[1], 'bo')
    #     plt.plot(line_end_2[0], line_end_2[1], 'r*')
    #     plt.plot(control_point[0], control_point[1], '8k')
    #     plt.show()
    #     breakpoint()

    return result

@njit
def aero_influence_coeff_mats(bound_mesh, wake_mesh, control_points):


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

    bound_mesh      = np.stack((x_mesh_b, y_mesh_b, z_mesh_b), axis=0)
    control_mesh    = np.stack((x_mesh_c, y_mesh_c, z_mesh_c), axis=0)
    wake_mesh       = np.stack((x_mesh_w, y_mesh_w, z_mesh_w), axis=0)

    for i in np.arange(no_bound_panels): # cycle through the control points
        for j in np.arange(no_bound_panels): # cycle through the bound panels
            A_b[:, i, j] = calc_velocity_panel_and_sym(n_v, i, j, control_mesh, bound_mesh)
        for j in np.arange(no_wake_panels): # cycle through the wake panels
            A_w[:,i,j] = calc_velocity_panel_and_sym(n_v, i, j, control_mesh, wake_mesh)

    print("Aerodynamic coefficient matrices computed.")
    return A_b, A_w

def solve_steady_aero(alphas, aero_data, A_w, A_b, y_mesh, reduced_wake=True):

    if reduced_wake:
        n_v = A_w.shape[2]
        m_v = int(A_b.shape[1] / n_v)
        P_b = np.concatenate([np.zeros((n_v, (m_v-1) * n_v)), np.eye(n_v)], axis=1)

    """
    with the assumption that all of the panels are aligned with the x and y axes, we can just take the z component of
    the first dimension in the A matrices (which corresponds to the z component of the velocity vectors obtained upon
    multiplication by the respective vorticity vectors)
    """
    uz_component = np.s_[2, :, :]
    A_b = A_b[uz_component]
    A_w = A_w[uz_component]

    combined_mats = A_b + A_w @ P_b
    ones = np.ones((A_b.shape[0], 1))
    delta_y_vec = y_mesh[0:n_v] - y_mesh[1:(n_v+1)]
    delta_y_vec = np.tile(delta_y_vec, m_v)
    G_y = np.eye(m_v*n_v) - np.concatenate(
                                            [np.zeros( (n_v, m_v*n_v) ),
                            np.concatenate( [np.eye((m_v-1)*n_v), np.zeros(((m_v-1)*n_v, n_v))],
                                                           axis=1) ],
                                           axis=0)

    F_a = np.zeros((len(alphas), A_b.shape[1]) ).T

    for i, alpha in enumerate(alphas):
        Gamma_b = - np.linalg.inv(combined_mats) @ ones * np.sin(alpha) * aero_data['v0']
        result = aero_data['v0'] * aero_data['rho'] * np.cos(alpha) * (G_y * delta_y_vec) @ Gamma_b
        F_a[:,i:i+1] = result

    return F_a

# def convergence_study()