import os
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from functions import *

@njit
def wake_AIC_mat(wake_mesh, control_points):

    x_mesh_w, y_mesh_w, z_mesh_w = [wake_mesh[:,:,i] for i in range(3)]
    x_mesh_c, y_mesh_c, z_mesh_c = [control_points[:,:,i] for i in range(3)]

    m_v = x_mesh_b.shape[0] - 1 # number of panels, which is one less than the no. of nodes in the considered direction
    n_v = x_mesh_b.shape[1] - 1
    if n_v != (x_mesh_w.shape[1] -1):
        raise ValueError("Wake mesh dimensions in the y direction do not match")
    m_w = x_mesh_w.shape[0]  - 1
    no_bound_panels = m_v * n_v
    no_wake_panels  = m_w * n_v

    A_w = np.zeros((3, no_bound_panels, no_wake_panels)) # not square because the number of wake panels does not match that of the collocation points
    # 3 because we obtain 3 speed components
    # Switch Notation
    x_mesh_c = x_mesh_c.flatten()
    y_mesh_c = y_mesh_c.flatten()
    z_mesh_c = z_mesh_c.flatten()
    x_mesh_w = x_mesh_w.flatten()
    y_mesh_w = y_mesh_w.flatten()
    z_mesh_w = z_mesh_w.flatten()

    control_mesh    = np.stack((x_mesh_c, y_mesh_c, z_mesh_c), axis=0)
    wake_mesh       = np.stack((x_mesh_w, y_mesh_w, z_mesh_w), axis=0)

    for i in np.arange(no_bound_panels): # cycle through the control points
        for j in np.arange(no_wake_panels): # cycle through the wake panels
            A_w[:,i,j] = calc_velocity_panel_and_sym(n_v, i, j, control_mesh, wake_mesh)
    return  A_w

aero_data, _ = load_aero_data()
chord = aero_data["c"][0]
semi_span = aero_data['s']

# base number of panels for half the wing (we're using symmetry)
n_v = 8*6
m_v = 5*6

reduced_wake = 1
convergence = False
convergence_threshold = 0.1e-2

AoA_deg = 10
AoA = AoA_deg / 180 * pi  # select biggest AoA in order to have a bigger magnitude in the CL, thus bigger change wrt mesh size

results = np.zeros((2, 30))

current_dir = os.path.dirname(__file__)
conv_results_path = os.path.join(current_dir, 'results_conv_study', f'wake_study_n_v_{n_v}.npy')
wake_plot_path = os.path.join(current_dir, 'results_conv_study', f'wake_study_n_v_{n_v}.png')
conv_results_path = os.path.abspath(conv_results_path) # Normalize the path

data = np.load(os.path.join(current_dir,'aero_influence_coeff_mats','nv_48_mv_30_steady.npz'))
A_b = data['bound']

if not os.path.exists(conv_results_path):
    print('Starting convergence study')
    for c_w_factor in range(1,31):
        print(f'Considering wake length of {c_w_factor} times c')
        aero_mats_path = os.path.join(current_dir, 'aero_influence_coeff_mats','wake_study', f'c_w_{c_w_factor}.npy')
        aero_mats_path = os.path.abspath(aero_mats_path)

        y_points = np.cos(
            pi / 2 - np.arange(n_v + 1) * pi / 2 / n_v) * semi_span  # needed to calculate delta y of each row of panels

        if os.path.exists(aero_mats_path):
            A_w = np.load(aero_mats_path)
        else: # calculate the matrix
            x_points = np.linspace(0, chord, m_v + 1)

            # geometric mesh
            x_mesh_g, y_mesh_g = np.meshgrid(x_points, y_points, indexing='ij')
            z_mesh_g = np.zeros_like(x_mesh_g)

            delta_x = chord / m_v
            x_mesh_b = x_mesh_g + delta_x / 4  # apply shift downwind

            # wake mesh
            if reduced_wake:
                x_points_wake = np.array([0, c_w_factor * chord])
            else:
                x_points_wake = np.linspace(0, c_w_factor * chord, c_w_factor * m_v + 1)

            x_points_wake += chord + delta_x / 4  # apply shift downwind
            x_mesh_w, y_mesh_w = np.meshgrid(x_points_wake, y_points, indexing='ij')
            z_mesh_w = np.zeros_like(x_mesh_w)

            x_control, y_control, z_control = find_middle_point(x_mesh_b, y_mesh_g, z_mesh_g)

            wake_mesh = np.stack((x_mesh_w, y_mesh_w, z_mesh_w), axis=2)
            control_mesh = np.stack((x_control, y_control, z_control), axis=2)

            A_w = wake_AIC_mat(wake_mesh, control_mesh)
            np.save(aero_mats_path, A_w)
        L, D = solve_steady_aero(AoA, aero_data['v0'], aero_data['rho'], A_w, A_b, y_points)
        _, _, C_L, C_D = aero_post_process(L, D, aero_data['v0'], aero_data['rho'], chord * 2*semi_span)

        i = c_w_factor - 1
        results[:, i] = [C_L, C_D]

    np.save(conv_results_path, results)
else:
    print('Loading previously computed results for the given convergence tolerance.')
    results = np.load(conv_results_path)
for i in range(1,30):
    diff = abs(results[0, i - 1] - results[0, i]) / results[0, i]
    if diff < convergence_threshold:
        convergence = True
        good_c_w_factor = i+1
        break
if convergence:
    print(f'Convergence achieved for a wake length of {good_c_w_factor} times c')
else:
    print(f'Convergence was not achieved within the given number of maximum iterations')
fig, ax = plt.subplots()
ax.plot( np.arange(1, 31),results[0, :], 'ro--', label = r'$C_L(\alpha=$'+f'{AoA_deg:.0f} deg)' )
ax.plot(np.arange(1, i+2), results[1, :i+1],  'b*--', label = r'$C_D$' )
ax.set_xlim((0.5,30))
ax.set_xlabel('Wake Length Multiplier')
ax.set_ylabel('Aerodynamic coefficient')
ax.grid()
ax.legend()
fig.tight_layout()
# plt.show()
fig.savefig(wake_plot_path)