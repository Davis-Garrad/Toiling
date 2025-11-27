import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from PIL import Image

R = 4
params = { 'bounds': [[-R, R], [-R, R]],
           'alpha': -np.pi/8,
           'U': 1,
           'beta': np.pi-np.pi*1/16,
           'a': 0.99,
           'zeta_1_arg': np.pi-np.pi/4,
           'zeta_1_amp': 1/1.2,
           'plot_options': { 'resolution': 512, },
          }

def generate_plots(params):

    sim_N = params['plot_options']['resolution']

    #inverse_zhuk = lambda z: z/2 * (1 + np.emath.sqrt(1 - 4/np.square(z)))
    #dinverse_zhuk = lambda z: 1/2 * (1 + np.sign(np.real(z))*z/np.emath.sqrt(np.square(z) - 4))
    #inverse_zhuk2 = lambda z: z/2 * (1 - np.emath.sqrt(1 - 4/np.square(z)))
    #dinverse_zhuk2 = lambda z: 1/2 * (1 - np.sign(np.real(z))*z/np.emath.sqrt(np.square(z) - 4))

    xi_0 = 1 + params['a'] * np.exp(1j * params['beta'])
    gamma = 4*np.pi*params['U'] * params['a'] * np.sin(params['alpha'] - params['beta'])
    
    zeta_1 = 1/2 * (-1 + np.exp(1j * params['zeta_1_arg']))
    zeta_2 = -1-zeta_1
    zeta_mu = zeta_1 * zeta_2

    transform = lambda xi: xi + 1/xi * (1 - zeta_mu) + 1/np.square(xi) * zeta_mu/2

    xi_coords = np.linspace(*params['bounds'][0], sim_N)[None,:] + 1j*np.linspace(*params['bounds'][0], sim_N)
    circle_coords = np.linspace(0,2*np.pi,sim_N)

    foil = lambda theta: xi_0 + params['a'] * np.exp(1j * theta)
    #foil_top = lambda percent: zhuk(foil((theta_c - theta_h)*percent + theta_h))
    #foil_bottom = lambda percent: zhuk(foil((theta_c + 2*np.pi - theta_h)*percent + theta_h))

    #cylinder_complex_potential = lambda xi: (params['U']*np.exp(-1j*params['alpha'])*(xi-xi_0) + params['U']*np.exp(1j*params['alpha'])*np.square(params['a'])/(xi - xi_0) - 1j*gamma/(2*np.pi) * np.log((xi-xi_0)/params['a']))

    #cylinder_complex_velocity = lambda xi: params['U']*np.exp(-1j*params['alpha']) - params['U']*np.exp(1j*params['alpha'])*np.square(params['a']/(xi - xi_0)) - 1j*gamma/(2*np.pi) * 1/(xi-xi_0)


    def complex_points(curve):
        return np.real(curve), np.imag(curve)


    realised_foil_vertices = np.array(complex_points(transform(foil(circle_coords)))).T
    foil_polygon = mpl_patches.Polygon(realised_foil_vertices, closed=True, fill=True, color='k', zorder=100)
    realised_cylinder_vertices = np.array(complex_points(foil(circle_coords))).T
    cylinder_polygon = mpl_patches.Polygon(realised_cylinder_vertices, closed=True, fill=True, color='k', zorder=100)

    fig, [ax_xi, ax_z] = plt.subplots(nrows=1, ncols=2)
    ax_xi.set_aspect('equal')
    ax_xi.scatter(*complex_points(zeta_1), zorder=1000, c='b')
    ax_xi.scatter(*complex_points(zeta_2), zorder=1000, c='b')
    ax_xi.axhline(0, c='k')
    ax_xi.axvline(0, c='k')
    #ax_xi.quiver(real_coords_x, real_coords_y, *complex_points(np.where(np.abs(complex_coords-xi_0)>np.sqrt(params['a']), np.conj(cylinder_complex_velocity(complex_coords)), 0)))
    #ax_xi.streamplot(real_coords_x, real_coords_y, *complex_points(np.conj(cylinder_complex_velocity(complex_coords))), broken_streamlines=False, density=2, start_points=np.array([np.zeros(40), np.linspace(params['bounds'][1][0], params['bounds'][1][1], 40)]).T, arrowstyle='-', color='black')
    ax_xi.add_patch(cylinder_polygon)

    ax_xi.set_xlim(params['bounds'][0])
    ax_xi.set_ylim(params['bounds'][1])


    ax_z.set_aspect('equal')

    #potential_vals = cylinder_complex_potential(inverse_zhuk(complex_coords))
    #potential_vals2 = cylinder_complex_potential(inverse_zhuk2(complex_coords))
    #velocity = lambda z: np.conj(dinverse_zhuk(z) * cylinder_complex_velocity(inverse_zhuk(z)))
    #velocity2 = lambda z: np.conj(dinverse_zhuk2(z) * cylinder_complex_velocity(inverse_zhuk2(z)))

    #thetas = np.linspace(0, 1, 100)
    #bot = foil_bottom(thetas)
    #bottom = np.max(np.imag(bot))

    #velocity_vals = np.where((np.abs(np.real(complex_coords))<2)*(np.imag(complex_coords)>0)*(np.imag(complex_coords)<bottom), velocity2(complex_coords), velocity(complex_coords))
    #normed_velocity_vals = velocity_vals

    #pressures = -np.square(np.abs(velocity_vals))/2
    #mean = 1/2 * np.average(pressures[:10,:] + pressures[-10:,:])
    #print(mean)
    #std = np.std(pressures)
    #print(std)

    #ctr = ax_z.pcolormesh(*complex_points(complex_coords), pressures, vmin=-1, vmax=0, cmap='bwr')
    #fig.colorbar(ctr)
    #ax_z.streamplot(*complex_points(complex_coords), *complex_points(normed_velocity_vals), broken_streamlines=False, density=2, start_points=np.array([np.zeros(30), np.linspace(params['bounds'][1][0], params['bounds'][1][1], 30)]).T, arrowstyle='-', color='black')

    ax_z.add_patch(foil_polygon)

    ax_z.set_xlim(params['bounds'][0])
    ax_z.set_ylim(params['bounds'][1])

    #ax_z.legend([f'$a={params['a']}$\n$\\beta-\\alpha={180/np.pi * (params['beta']-params['alpha']):05.1f}^\\circ$'], loc='upper right')

    #if('savefile' in params['plot_options'].keys()):
    #    fig.savefig(params['plot_options']['savefile'])
    #else:
    plt.show()

    return fig

fig=generate_plots(params)

