import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from PIL import Image
import scipy as sp

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
    dinvtransform = lambda z: np.power(1+((zeta_mu-1)/np.power(z,2) - zeta_mu/np.power(z,3)), -1)

    xi_coords = np.linspace(*params['bounds'][0], sim_N)[None,:] + 1j*np.linspace(*params['bounds'][0], sim_N)[:,None]
    circle_coords = np.linspace(0,2*np.pi,sim_N)
    z_coords = transform(xi_coords)
    inverse_map = lambda z: xi_coords[np.argmin(np.abs(z - z_coords))]

    #print(inverse_map(z_coords) - xi_coords) # should be zeroes.

    foil = lambda theta: xi_0 + params['a'] * np.exp(1j * theta)
    #foil_top = lambda percent: zhuk(foil((theta_c - theta_h)*percent + theta_h))
    #foil_bottom = lambda percent: zhuk(foil((theta_c + 2*np.pi - theta_h)*percent + theta_h))

    cylinder_complex_potential = lambda xi: (params['U']*np.exp(-1j*params['alpha'])*(xi-xi_0) + params['U']*np.exp(1j*params['alpha'])*np.square(params['a'])/(xi - xi_0) - 1j*gamma/(2*np.pi) * np.log((xi-xi_0)/params['a']))

    cylinder_complex_velocity = lambda xi: params['U']*np.exp(-1j*params['alpha']) - params['U']*np.exp(1j*params['alpha'])*np.square(params['a']/(xi - xi_0)) - 1j*gamma/(2*np.pi) * 1/(xi-xi_0)


    def complex_points(curve):
        return np.real(curve), np.imag(curve)


    realised_foil_vertices = np.array(complex_points(transform(foil(circle_coords)))).T
    foil_polygon = mpl_patches.Polygon(realised_foil_vertices, closed=True, fill=True, color='k', zorder=100)
    realised_cylinder_vertices = np.array(complex_points(foil(circle_coords))).T
    cylinder_polygon = mpl_patches.Polygon(realised_cylinder_vertices, closed=True, fill=True, color='k', zorder=100)

    fig, [ax_xi, ax_z] = plt.subplots(nrows=1, ncols=2)
    ax_xi.set_aspect('equal')
    ax_xi.scatter(*complex_points(zeta_1), zorder=1000, c='white')
    ax_xi.scatter(*complex_points(zeta_2), zorder=1000, c='white')
    ax_xi.axhline(0, c='k')
    ax_xi.axvline(0, c='k')
    #ax_xi.quiver(real_coords_x, real_coords_y, *complex_points(np.where(np.abs(complex_coords-xi_0)>np.sqrt(params['a']), np.conj(cylinder_complex_velocity(complex_coords)), 0)))
    ax_xi.streamplot(*complex_points(xi_coords), *complex_points(np.conj(cylinder_complex_velocity(xi_coords))), broken_streamlines=False, density=2, start_points=np.array([np.zeros(40), np.linspace(params['bounds'][1][0], params['bounds'][1][1], 40)]).T, arrowstyle='-', color='black')
    ax_xi.add_patch(cylinder_polygon)

    ax_xi.set_xlim(params['bounds'][0])
    ax_xi.set_ylim(params['bounds'][1])


    ax_z.set_aspect('equal')

    #potential_vals = cylinder_complex_potential(inverse_zhuk(complex_coords))
    #potential_vals2 = cylinder_complex_potential(inverse_zhuk2(complex_coords))
    velocity = lambda z: np.conj(dinvtransform(z) * cylinder_complex_velocity(inverse_map(z)))

    def get_streamline(st, N, z_field, vel_field, scale=1):
        z = st
        zs = np.zeros((N,), dtype=complex)
        for i in range(N):
            zs[i] = z
            zi = np.argmin(np.abs(z - z_field.ravel()[:,None]))
            vel = vel_field.ravel()[zi]
            if(zi > 0):
                dz = z_field.ravel()[zi] - z_field.ravel()[zi-1]
            else:
                dz = z_field.ravel()[zi+1] - z_field.ravel()[zi]
            #if(np.abs(z) > np.abs(st)):
            #    vel = 0
            z += vel * scale
            
        return zs

    num_streamlines = 3*33
    streamline_length = 300
    pressureresolution = 1
    streamlines = np.zeros((num_streamlines, streamline_length), dtype=complex)
    start_i = np.append(np.append(-3.5 + 1j*np.linspace(-3.5, 3.5, num_streamlines//3, endpoint=True), 3.5j + np.linspace(-3.5, 3.5, num_streamlines//3)), -3.5j + np.linspace(-3.5,3.5,num_streamlines//3))
    for xi in range(len(start_i)):
        x = start_i[xi]
        streamlines[xi] = transform(get_streamline(x, streamline_length, xi_coords, np.conj(cylinder_complex_velocity(xi_coords)), scale=2/50))

    velocities = np.zeros((streamlines.shape[0], (streamlines.shape[1]-1)//pressureresolution))
    for i in range(streamlines.shape[0]):
        velocities[i] = (streamlines[i,1::pressureresolution] - streamlines[i,:-1:pressureresolution])
        #ax_z.quiver(*complex_points(streamlines[i,1::pressureresolution]), *complex_points(velocities[i]), scale=1)
    
    for i in range(num_streamlines):
        rel_speed = np.max(np.abs(velocities[i]))/np.max(np.abs(velocities.ravel()))
        ax_z.plot(*complex_points(streamlines[i]), color=(rel_speed, 1-rel_speed, 0.0))
    #pressures = -np.square(np.abs(velocities))/2.0
    #mean = -3e-4#np.average(pressures)
    #std = 3e-4#np.std(pressures.ravel())
    #print(np.average(pressures), np.std(pressures))

    #XX = ax_z.tricontourf(*complex_points(streamlines[:,1::pressureresolution].ravel()), -pressures.ravel(), levels=100, cmap='bwr', vmax=-mean+std, vmin=-mean-std)
    #plt.colorbar(XX)

    #velocity2 = lambda z: np.conj(dinverse_zhuk2(z) * cylinder_complex_velocity(inverse_zhuk2(z)))

    #thetas = np.linspace(0, 1, 100)
    #bot = foil_bottom(thetas)
    #bottom = np.max(np.imag(bot))

    #velocity_vals = np.where((np.abs(np.real(complex_coords))<2)*(np.imag(complex_coords)>0)*(np.imag(complex_coords)<bottom), velocity2(complex_coords), velocity(complex_coords))
    #velocity_vals = velocity(z_coords)/np.abs(velocity(z_coords))

    #regular_z_coords = np.copy(xi_coords)
    #nn = 320

    #interp_velocity_vals = interpolator(np.array(complex_points(regular_z_coords.ravel())).T).reshape(regular_z_coords.shape)
    #interp_velocity_vals = np.where(np.imag(regular_z_coords) < 0.5, np.conj(interp_velocity_vals), interp_velocity_vals)
    #normed_velocity_vals = velocity_vals

    #pressures = -np.square(np.abs(velocity_vals))/2
    #mean = 1/2 * np.average(pressures[:10,:] + pressures[-10:,:])
    #print(mean)
    #std = np.std(pressures)
    #print(std)

    #ctr = ax_z.pcolormesh(*complex_points(complex_coords), pressures, vmin=-1, vmax=0, cmap='bwr')
    #fig.colorbar(ctr)
    #nn = 4
    #ax_z.quiver(*complex_points(z_coords[::nn,::nn]), *complex_points(velocity_vals[::nn,::nn]))
    #ax_z.streamplot(*complex_points(regular_z_coords[::nn,::nn]), *complex_points(interp_velocity_vals[::nn,::nn]))
    #ax_z.streamplot(*complex_points(z_coords), *complex_points(velocity_vals), broken_streamlines=False, density=2, start_points=np.array([np.zeros(30), np.linspace(params['bounds'][1][0], params['bounds'][1][1], 30)]).T, arrowstyle='-', color='black')

    ax_z.add_patch(foil_polygon)

    ax_z.set_xlim(np.array(params['bounds'][0]) * 0.75)
    ax_z.set_ylim(np.array(params['bounds'][1]) * 0.75)

    #ax_z.legend([f'$a={params['a']}$\n$\\beta-\\alpha={180/np.pi * (params['beta']-params['alpha']):05.1f}^\\circ$'], loc='upper right')

    if('savefile' in params['plot_options'].keys()):
        fig.savefig(params['plot_options']['savefile'])
    #plt.ioff()
    #plt.show()

    return fig

images = []
index = 0
for i in np.linspace(-np.pi/4, np.pi/4, 25):
    params['alpha'] = i
    params['plot_options']['savefile'] = f'tmp_vonmises/tmp{index}.png'
    fig=generate_plots(params)
    plt.close(fig)
    images.append(Image.open(f'tmp_vonmises/tmp{index}.png'))
    print(index)
    index += 1
images[0].save('animation.gif', save_all=True, append_images=images, duration=100, loop=0)

