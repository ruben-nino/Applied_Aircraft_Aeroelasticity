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
from VLM_FEM_Coupling.functions import solve_deflections, solve_aero_elastic

if __name__ == '__main__':
    
    data = load_TangDowell_data()
    data['fem']['KBT'] =  0.1 * (data['fem']['EI'] * data['fem']['GJ'])

    fem = fe.initialise_fem(data['fem'])
    aero_data = data['aero']

    n_nd = fem['n_nd']
    n_dof = fem['n_dof']
    y_nd = fem['y_nd']
    b_u = fem['b_u']


    rho = aero_data['rho']
    chord = aero_data["c"][0]
    b = chord / 2
    semi_span = aero_data['s']
    U = aero_data['v0']
    ac = aero_data['ac'][0] # distance to the beam axis as a fraction of the chord measured from the LE, [-]
    a = ac*chord/b - 1  # distance to the beam axis measured from the mid-chord point in semi-chords, (xf/b -1), [-]

    bss_u = np.append(b_u, b_u) # twice the dimensions in SS form since we also consider nodal velocities in the state vector
    n_dof_red = len(b_u[b_u])

    # results from convergence studies
    m_v = 5 * 6
    n_v = 8 * 6
    c_w_factor = 15

    # get AIC matrices

    aero_mats_path = (Path.cwd().parent / 'My_VLM_Implementation'/ 'aero_influence_coeff_mats' /
                      f'nv_{n_v}_mv_{m_v}_cw_{c_w_factor}_steady.npz')
    mats = np.load(aero_mats_path)
    A_b = mats['bound']
    A_w = mats['wake']

    x_points = np.linspace(0, chord, m_v + 1)
    y_points = np.cos( pi / 2 - np.arange(n_v + 1) * pi / 2 / n_v) * semi_span  # needed to calculate delta y of each row of panels

    aoa0 = 10 * pi / 180.0

    L, _ = solve_steady_aero(aoa0, U, rho, A_w, A_b, y_points)
    L = L.reshape((m_v, n_v))
    # Assemble control points mesh

    xi, f_rigid, y_control = solve_deflections(fem, n_dof, n_nd, y_nd, L, ac, chord, semi_span, n_v, m_v, b_u )

    xi_rigid = xi.copy()

    for i in range(3): # update aerodynamic forces based on structural displacements
        # AoA correction:

        thetas = np.zeros((m_v * n_v, 1))
        for j in range(n_v): # loop through the columns of panels (spanwise)
            # find the closest beam node
            closest_nd_idx = (np.abs(y_nd - y_control[0, j])).argmin()  # 0th element has the same y as any other in the column
            thetas[j::n_v] = xi[closest_nd_idx * 3] * np.ones((m_v,1))  # store the torsional deflections

        L, _ = solve_aero_elastic(aoa0, thetas, U, rho, A_w, A_b, y_points) # pass them to the VLM solver
        L = L.reshape((m_v, n_v))
        xi, f, _ = solve_deflections(fem, n_dof, n_nd, y_nd, L, ac, chord, semi_span, n_v, m_v, b_u)


    # Plot the structural deformation:
    lbl_y = ['Lift [N]','theta, [rad]', 'v, [m]', 'beta, [rad]'] # NOTE beta here is the y derivative of v ( =rotation in EB beams)

    fig, ax = plt.subplots(4, 1, sharex=True, num=10)
    #
    ax[0].plot(y_nd[:], f[:], 'og')
    ax[0].plot(y_nd, f_rigid, '*r')

    for j in range(1,4):
        ax[j].plot(y_nd, xi[j-1::3], '--')
        ax[j].plot(y_nd, xi_rigid[j-1::3], '-.')
        # ax[j].plot(y_nd, xi[n_dof + j::3])
    #     ax[j].plot(y_nd, xi3[n_dof + j::3], '--')
    #     ax[j].plot(y_nd, xi4[n_dof + j::3], 'o--')
    #     ax[j].plot(y_nd, arr_xi2[0][j::3], ':', y_nd, arr_xi2[1][j::3], ':')
    #     ax[j].plot(y_nd, -arr_xi2[0][j::3], ':', y_nd, -arr_xi2[1][j::3], ':')
    #
        if j == 3:
            ax[j].set_xlabel('span, [m]')
    #
        ax[j].set_ylabel(lbl_y[j])
    #
    fig.suptitle(f'{U = :.3f} m/s, { aoa0 = :.3f} rad')
    #
    # plt.show()
    fig.savefig('./Plots/flex_vs_rigid.png', format='png', dpi=300)