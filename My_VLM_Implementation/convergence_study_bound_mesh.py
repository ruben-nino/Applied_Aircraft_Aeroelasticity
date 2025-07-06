import os
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from functions import *

convergence_threshold = 0.1e-2
max_iters = 10

aero_data, _ = load_aero_data()
chord = aero_data["c"][0]
semi_span = aero_data['s']

# base number of panels for half the wing (we're using symmetry)
n_v_base = 8
m_v_base = 5

c_w_factor = 10  # to multiply by c to obtain the wake length * aero_data['c'][0]
reduced_wake = 1
convergence = False

scaling_factor = 0
AoA_deg = 10
AoA = AoA_deg / 180 * np.pi  # select biggest AoA in order to have a bigger magnitude in the CL, thus bigger change wrt mesh size

results = np.zeros((2, max_iters))

current_dir = os.path.dirname(__file__)
conv_results_path = os.path.join(current_dir, 'results_conv_study', f'eps_{convergence_threshold}_max_{max_iters}.npy')
plot_path = os.path.join(current_dir, 'results_conv_study', f'plot_eps_{convergence_threshold}_max_{max_iters}.png')
conv_results_path = os.path.abspath(conv_results_path) # Normalize the path
plot_path = os.path.abspath(plot_path)

if __name__ == '__main__':
    if not os.path.exists(conv_results_path):
        while not convergence and scaling_factor < max_iters:

            scaling_factor += 1
            n_v = scaling_factor * n_v_base
            m_v = scaling_factor * m_v_base
            print(f'Solving for mesh with {n_v = } and {m_v = }')

            if reduced_wake:
                aero_mats_path = os.path.join(current_dir, 'aero_influence_coeff_mats', f'nv_{n_v}_mv_{m_v}_steady.npz')
            else:
                aero_mats_path = os.path.join(current_dir, 'aero_influence_coeff_mats', f'nv_{n_v}_mv_{m_v}.npz')

            aero_mats_path = os.path.abspath(aero_mats_path)

            if os.path.exists(aero_mats_path):
                data = np.load(aero_mats_path)
                A_b = data['bound']
                A_w = data['wake']
            else: # calculate the matrices
                x_points = np.linspace(0, chord, m_v + 1)
                y_points = np.cos(
                    pi / 2 - np.arange(
                        n_v + 1) * pi / 2 / n_v) * semi_span  # only mesh half the wing (use symmetry wrt zx plane)

                # geometric mesh
                x_mesh_g, y_mesh_g = np.meshgrid(x_points, y_points, indexing='ij')
                z_mesh_g = np.zeros_like(x_mesh_g)

                # bound mesh
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

                bound_mesh = np.stack((x_mesh_b, y_mesh_g, z_mesh_g), axis=2)
                wake_mesh = np.stack((x_mesh_w, y_mesh_w, z_mesh_w), axis=2)
                control_mesh = np.stack((x_control, y_control, z_control), axis=2)
                # breakpoint() # bound_mesh.shape[:] = (11, 17, 3) for m_v = 10 and n_v = 16

                A_b, A_w = aero_influence_coeff_mats(bound_mesh, wake_mesh, control_mesh)
                np.savez(aero_mats_path, bound=A_b, wake=A_w)

            # Solve for Steady-State solution
            # Assume that n (panel unit vector) is [0, 0, 1] for all panels
            # AOAs = np.array([1, 3, 10]) * (np.pi / 180) #

            y_points = np.cos(
                pi / 2 - np.arange(n_v + 1) * pi / 2 / n_v) * semi_span  # needed to calculate delta y of each row of panels

            L, D = solve_steady_aero(AoA, aero_data['v0'], aero_data['rho'], A_w, A_b, y_points)
            _, _, C_L, C_D = aero_post_process(L, D, aero_data['v0'], aero_data['rho'], chord * 2*semi_span)

            i = scaling_factor - 1
            if i == 0:
                results[:,0] = [C_L, C_D]
            else:
                results[:, i] = [C_L, C_D]

                diff_1 = abs(results[0, i-1] - results[0, i]) / C_L
                diff_2 = abs(results[1, i - 1] - results[1, i]) / C_D

                if diff_1 < convergence_threshold and diff_2 < convergence_threshold:
                    convergence = True

        np.save(conv_results_path, results)
        if convergence:
            print(f'Convergence achieved for a scaling factor of {scaling_factor} applied to the base mesh')
        else:
            print(f'Convergence was not achieved within the given number of maximum iterations')
    else:
        print('Loading previously computed results for the given convergence tolerance.')
        results = np.load(conv_results_path)

    idx = np.argwhere(np.all(results[..., :] == 0, axis=0))
    results = np.delete(results, idx, axis=1)

    fig, ax = plt.subplots()
    ax.plot( np.arange(1, results.shape[1]+1),results[0, :], 'ro--', label = r'$C_L(\alpha=$'+f'{AoA_deg:.0f} deg)' )
    ax.plot(np.arange(1, results.shape[1]+1), results[1, :],  'b*--', label = r'$C_D$' )
    ax.set_xlabel('Scaling factor over base mesh')
    ax.set_ylabel('Aerodynamic coefficient')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    # plt.show()
    fig.savefig(plot_path)