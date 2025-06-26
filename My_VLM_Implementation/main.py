import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from functions import *



aero_data, _ = load_aero_data()

# %% Generate meshes
n_v = 16
m_v = 10
c_w_factor = 10 # to multiply by c to obtain the wake length * aero_data['c'][0]

x_points = np.linspace(0, aero_data["c"][0], m_v+1)
y_points = np.cos( pi - (np.arange( n_v + 2) - 1) * pi / n_v) * aero_data['s']

# geometric mesh
x_mesh_g, y_mesh_g = np.meshgrid(x_points, y_points)
# plt.plot(x_mesh_g, y_mesh_g, '*r')
# plt.xlim(- 0.1, aero_data["s"]/2)
# plt.show()

# bound mesh
delta_x = aero_data["c"][0] / m_v
x_mesh_b = x_mesh_g + delta_x / 4

# wake mesh
x_points_wake = np.linspace(0, c_w_factor * aero_data['c'][0], c_w_factor * m_v + 1 )
x_points_wake += aero_data["c"][0] + delta_x / 4
x_mesh_w, y_mesh_w = np.meshgrid(x_points_wake, y_points)

