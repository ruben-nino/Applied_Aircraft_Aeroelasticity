import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from functions import *

visualize = 0

aero_data, _ = load_aero_data()
chord           = aero_data["c"][0]
semi_span       = aero_data['s']

# %% Generate meshes

# number of panels for half the wing (we're using symmetry)
n_v = 16
m_v = 10
c_w_factor = 10 # to multiply by c to obtain the wake length * aero_data['c'][0]

x_points = np.linspace(0, chord, m_v + 1)
y_points = np.cos( pi/2 - np.arange( n_v + 1)  * pi/2 / n_v) * semi_span # only mesh half the wing (use symmetry wrt zx plane)

# geometric mesh
x_mesh_g, y_mesh_g = np.meshgrid(x_points, y_points, indexing='ij')
z_mesh_g = np.zeros_like(x_mesh_g)

# bound mesh
delta_x = chord / m_v
x_mesh_b = x_mesh_g + delta_x / 4

# wake mesh
x_points_wake = np.linspace(0, c_w_factor * aero_data['c'][0], c_w_factor * m_v + 1 )
x_points_wake += chord + delta_x / 4
x_mesh_w, y_mesh_w = np.meshgrid(x_points_wake, y_points)
z_mesh_w = np.zeros_like(x_mesh_w)

x_control, y_control, z_control = find_middle_point(x_mesh_b, y_mesh_g, z_mesh_g)

if __name__ == "__main__" and visualize:
    plt.plot(x_mesh_g, y_mesh_g, '*r')
    plt.plot(x_mesh_b, y_mesh_g, 'ob', markersize=3)
    plt.plot(x_mesh_w, y_mesh_w, '1g')
    plt.plot(x_control, y_control, '8k', markersize=1)

    # plt.xlim(- 0.025, semi_span)
    plt.xlim(- 0.025, 0.1)

    plt.hlines(y = semi_span, xmin= - 0.1, xmax=semi_span)
    plt.show()
    breakpoint()

bound_mesh = np.stack((x_mesh_b, y_mesh_g, z_mesh_g), axis=2)
wake_mesh  = np.stack((x_mesh_w, y_mesh_w, z_mesh_w), axis=2)
# breakpoint() # bound_mesh.shape[:] = (11, 17, 3) for m_v = 10 and n_v = 16

# consider n unit vector in the integration step