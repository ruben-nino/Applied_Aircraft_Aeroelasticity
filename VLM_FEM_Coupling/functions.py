import os
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
import scipy.linalg as ln
from pathlib import Path

from My_VLM_Implementation.functions import *
import FEM.fem_linear as fe
import FEM.fem_loads as fl
import FEM.fem_state_space as fss

def solve_deflections(fem, n_dof, n_nd, y_nd, L, ac, chord, semi_span, n_v, m_v, b_u):
    x_points = np.linspace(0, chord, m_v + 1)
    y_points = np.cos( pi / 2 - np.arange(n_v + 1) * pi / 2 / n_v) * semi_span  # needed to calculate delta y of each row of panels

    x_mesh_g, y_mesh_g = np.meshgrid(x_points, y_points, indexing='ij')# geometric mesh
    z_mesh_g = np.zeros_like(x_mesh_g)
    delta_x = chord / m_v
    x_mesh_g = x_mesh_g + delta_x / 4  # apply shift downwind to get bound mesh
    x_control, y_control, _ = find_middle_point(x_mesh_g, y_mesh_g, z_mesh_g) # middle of each bound vortex is the control point

    # Nodal values of distributed structural torque moment, r, shear force, f, and bending moment, q.
    r = np.zeros((n_nd,))  # [Nm/m]
    f = np.zeros((n_nd,))  # [N/m]
    q = np.zeros((n_nd,))  # [Nm/m]

    for i in range(n_v):  # loop through the "columns" of panels in the span-wise direction
        # find closest beam node
        closest_nd_idx = (np.abs(y_nd - y_control[0, i])).argmin()  # 0th element has the same y as any other in the column
        y_closest = y_nd[closest_nd_idx]

        for j in range(m_v):  # loop through the chord-wise panels in the i-th column
            f[closest_nd_idx] += -1 * L[j, i]
            r[closest_nd_idx] += -1 * L[j, i] * (
                        ac * chord - x_control[j, i])  # x_node - x_panel # todo: look up coordinate sys on Mathematica

    lds = np.zeros((n_dof,))
    lds[0::3] = r
    lds[1::3] = f
    lds[2::3] = q

    lds_red = lds[b_u]

    # Solve the structural problem directly:
    KK = fe.mat_stiffness(fem)
    KKinv = ln.inv(KK)

    DDdst = fl.mat_force_dst(fem)

    xi_red = KKinv @ DDdst @ lds_red

    xi = np.zeros((n_dof,))
    xi[b_u] = xi_red
    return xi, f, y_control


@njit
def solve_aero_elastic(alpha0, thetas, v_0, rho, A_w, A_b, y_mesh, reduced_wake=True):

    if reduced_wake: # since we are calculating the steady solution it makes sense that the wake would be reduced (long panels).
        n_v = A_w.shape[2]
        m_v = int(A_b.shape[1] / n_v)
        P_b = np.hstack(
                            (np.zeros((n_v, (m_v-1) * n_v)), np.eye(n_v))
                        )

    """
    with the assumption that all of the panels are aligned with the x and y axes, we can just take the z component of
    the first dimension in the A matrices (which corresponds to the z component of the velocity vectors obtained upon
    multiplication by the respective vorticity vectors)
    """
    # uz_component = np.s_[2, :, :] # removed since slice objects are not supported by njit
    A_b = A_b[2, :, :]
    A_w = A_w[2, :, :]

    #NumbaPerformanceWarning: '@' is faster on contiguous arrays, called on (Array(float64, 2, 'A', False, aligned=True), Array(float64, 2, 'C', False, aligned=True))
    A_b = np.ascontiguousarray(A_b)
    A_w = np.ascontiguousarray(A_w)
    P_b = np.ascontiguousarray(P_b)

    combined_mats = A_b + A_w @ P_b
    ones = np.ones((A_b.shape[0], 1))
    delta_y_vec = y_mesh[0:n_v] - y_mesh[1:(n_v+1)]
    # delta_y_vec = np.tile(delta_y_vec, m_v) # np.tile is not supported either (this is a bit of a pain in the butt)
    full_delta_y = np.empty(n_v * m_v)
    for i in range(m_v):
        full_delta_y[i * n_v: (i + 1) * n_v] = delta_y_vec
    G_y = np.eye(m_v*n_v) - np.vstack(
                                        (np.zeros( (n_v, m_v*n_v) ),
                                        np.hstack(
                                            (np.eye((m_v-1)*n_v), np.zeros(((m_v-1)*n_v, n_v)))
                                                 )
                                        )
                                    )

    # calculate both "lift" and "drag" (quotation marks since lift should be perpendicular to the free stream,
    # see pg 33 of chapter 10 Introduction to nonlinear aeroelasticity)

    Gamma_b = - np.linalg.inv(combined_mats) @  np.sin(ones * alpha0 + thetas) * v_0
    normal      = v_0 * rho * np.cos(ones * alpha0 + thetas) * (G_y * full_delta_y) @ Gamma_b
    in_plane    = v_0 * rho * np.sin(ones * alpha0 + thetas) * (G_y * full_delta_y) @ Gamma_b # I think this is wrong actually

    return normal, in_plane