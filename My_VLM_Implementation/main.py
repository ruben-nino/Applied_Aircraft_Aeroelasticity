import os
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from functions import *

test = 1
visualize_mesh = 1
visualize_aero = 1

aero_data, _ = load_aero_data()
chord           = aero_data["c"][0]
semi_span       = aero_data['s']

# %% Test Aero Influence Coeff Mats Generation

if test:
    # number of panels for half the wing (we're using symmetry)
    n_v = 8
    m_v = 5
    c_w_factor = 15 # to multiply by c to obtain the wake length * aero_data['c'][0]
    reduced_wake = 1

    current_dir = os.path.dirname(__file__)
    if reduced_wake:
        aero_mats_path = os.path.join(current_dir, 'aero_influence_coeff_mats', f'nv_{n_v}_mv_{m_v}_cw_{c_w_factor}_steady.npz')
    else:
        aero_mats_path = os.path.join(current_dir, 'aero_influence_coeff_mats', f'nv_{n_v}_mv_{m_v}.npz')

    # Normalize the path
    aero_mats_path = os.path.abspath(aero_mats_path)

    if os.path.exists(aero_mats_path):
        print("Loading previously computed Aerodynamic Influence Coefficients (AIC) matrices")
        data = np.load(aero_mats_path)
        A_b = data['bound']
        A_w = data['wake']
    else: # calculate the matrices
        print("Computing Aerodynamic Influence Coefficients (AIC) matrices")
        start_time = time.time()

        x_points = np.linspace(0, chord, m_v + 1)
        y_points = np.cos(
            pi / 2 - np.arange(n_v + 1) * pi / 2 / n_v) * semi_span  # only mesh half the wing (use symmetry wrt zx plane)

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

        if __name__ == "__main__" and visualize_mesh:
            plt.hlines(y=semi_span, xmin=- 0, xmax=chord, color='b', label='Physical Wing Edges')
            plt.plot(x_mesh_g.flatten(), y_mesh_g.flatten(), '*b', label='Geometric Panel Vertices')
            plt.plot(x_mesh_b.flatten(), y_mesh_g.flatten(), 'or', label='Bound Panel Vertices', markersize=3)
            plt.plot(x_mesh_w.flatten(), y_mesh_w.flatten(), '1g', markersize=5)
            plt.plot(x_control.flatten(), y_control.flatten(), '+k', label='Control Points', markersize=5)
            # plt.xlim(- 0.025, semi_span)
            plt.xlim(- 0.025, 0.15)
            plt.vlines(x=[0, chord], ymin=0, ymax=semi_span, color='b')
            plt.hlines(y=y_points, xmin= chord+delta_x/4, xmax=chord * 10, color='g')
            plt.vlines(x= chord + delta_x/4, ymin=0, ymax=semi_span, color='g', label='Wake Vortices Edges')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.legend(loc='center right')
            plt.tight_layout()
            plt.savefig('Geometry_and_Mesh.png', format='png')
            # plt.show()
            breakpoint()

        bound_mesh = np.stack((x_mesh_b, y_mesh_g, z_mesh_g), axis=2)
        wake_mesh = np.stack((x_mesh_w, y_mesh_w, z_mesh_w), axis=2)
        control_mesh = np.stack((x_control, y_control, z_control), axis=2)
        # breakpoint() # bound_mesh.shape[:] = (11, 17, 3) for m_v = 10 and n_v = 16

        A_b, A_w = aero_influence_coeff_mats(bound_mesh, wake_mesh, control_mesh)
        end_time = time.time()
        print(f"AIC matrices computed in {end_time - start_time:.3f} seconds.")
        np.savez(aero_mats_path, bound=A_b, wake=A_w)

    # %% Run Integration

    # %% Solve for Steady-State solution

    # Assume that n (panel unit vector) is [0, 0, 1] for all panels
    AOAs = np.array([1, 3, 10]) * (np.pi/180)
    y_points = np.cos(pi / 2 - np.arange(n_v + 1) * pi / 2 / n_v) * semi_span # needed to calculate delta y of each row of panels

    F_aero = solve_steady_aero(AOAs, aero_data['v0'], aero_data['rho'], A_w, A_b, y_points)
    # breakpoint() # F_aero.shape[:] = (n_v*m_v, 3)
    if __name__ == "__main__" and visualize_aero:
        ax = plt.figure().add_subplot(projection='3d')
        # just realized that this is not really needed, stick to required points and do this if I have time afterwards

