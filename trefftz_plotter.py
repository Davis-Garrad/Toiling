import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

params = {
        'k': 1.95, # in Trefftz map
        'a': 1.05, # radius
        'xi_0': -0.05 + 0.1j, # centre
        'U': 1,
        'alpha': np.pi/8,
        'resolution': 512,
        'streamline_origins': 41,
        'streamline_length': 2048, # MAX length
        }

def cpts(complex_coord):
    return np.real(complex_coord), np.imag(complex_coord)

def get_streamline(st, N, z_field, vel_field, scale=1, rightbound=4, leftbound=-3):
    zs = np.zeros((N,), dtype=complex)
    zs[0] = st
    for i in range(1, N):
        zi = np.argmin(np.abs(zs[i-1] - z_field.ravel()[:,None]))
        vel = vel_field.ravel()[zi]
        if(zi > 0):
            dz = np.abs(z_field.ravel()[zi] - z_field.ravel()[zi-1])
        else:
            dz = np.abs(z_field.ravel()[zi+1] - z_field.ravel()[zi])
        #if(np.abs(z) > np.abs(st)):
        #    vel = 0
        if(np.abs(vel*scale) > 1):
            vel /= np.abs(vel*scale)
        zs[i] = zs[i-1] + vel * scale
        if(np.real(zs[i] > rightbound)):
           zs[i:] = zs[i]
           break
        if(np.real(zs[i]) < leftbound):
           zs[i:] = zs[i]
           break
        
    return zs

def trefftz_plots(params):
    bound = 4

    beta = np.angle(params['xi_0'] - 1)
    gamma = 4*np.pi*params['U'] * params['a'] * np.sin(params['alpha'] - beta)


    foil = lambda theta: params['xi_0'] + params['a'] * np.exp(1j * theta)
    trefftz = lambda xi: params['k'] * (np.power(xi + 1, params['k']) + np.power(xi - 1, params['k'])) / (np.power(xi + 1, params['k']) - np.power(xi - 1, params['k']))

    cylinder_complex_potential = lambda xi: (params['U']*np.exp(-1j*params['alpha'])*(xi-params['xi_0']) + params['U']*np.exp(1j*params['alpha'])*np.square(params['a'])/(xi - params['xi_0']) - 1j*gamma/(2*np.pi) * np.log((xi-params['xi_0'])/params['a']))

    cylinder_complex_velocity = lambda xi: np.conj(params['U']*np.exp(-1j*params['alpha']) - params['U']*np.exp(1j*params['alpha'])*np.square(params['a']/(xi - params['xi_0'])) - 1j*gamma/(2*np.pi) * 1/(xi-params['xi_0']))

    fig, (axl, axr) = plt.subplots(ncols=2)

    # foils
    thetas = np.linspace(0, 2*np.pi, 360, endpoint=True)
    axl.plot(*cpts(foil(thetas)))
    axr.plot(*cpts(trefftz(foil(thetas))))

    realised_xi_vertices = np.array(cpts(foil(thetas))).T
    xi_polygon = mpl_patches.Polygon(realised_xi_vertices, closed=True, fill=True, color='k', zorder=100)
    axl.add_patch(xi_polygon)
    realised_z_vertices = np.array(cpts(trefftz(foil(thetas)))).T
    z_polygon = mpl_patches.Polygon(realised_z_vertices, closed=True, fill=True, color='k', zorder=100)
    axr.add_patch(z_polygon)

    # xi velocity field
    xi_coords = np.linspace(-bound, bound, params['resolution'])[None,:] + 1j*np.linspace(-bound, bound, params['resolution'])[:,None]
    xi_velocity_field = cylinder_complex_velocity(xi_coords)
    
    # streamlines
    #axl.streamplot(*cpts(xi_coords), *cpts(xi_velocity_field))
    #origins = np.append(np.append(
    #    np.linspace(-bound, bound, params['streamline_origins'])*1j - bound,
    #    np.linspace(-bound, bound, params['streamline_origins']) - bound*1j),
    #    np.linspace(-bound, bound, params['streamline_origins']) + bound*1j,
    #    )

    origins = np.linspace(-bound, bound, params['streamline_origins']) * 1j - bound
    origins_rev = np.linspace(-bound, bound, params['streamline_origins']) * 1j + bound
    
    streamlines = np.zeros((len(origins)+len(origins_rev), params['streamline_length']), dtype=complex)
    #for ii in range(len(origins)):
    #    streamline = get_streamline(origins[ii], params['streamline_length'], xi_coords, xi_velocity_field, scale=1e-1)
    #    streamlines[ii] = streamline
    #    print(f'\r{ii} ({ii/(3*params['streamline_origins'])*100:.1f}%)   ', end='')
    for ii in range(len(origins_rev)):
        streamline = get_streamline(origins_rev[ii], params['streamline_length'], xi_coords, xi_velocity_field, scale=-1e-2)
        streamlines[ii+len(origins)] = streamline
        print(f'\r{ii} ({ii/(3*params['streamline_origins'])*100:.1f}%)   ', end='')
    
    xi_velocities = streamlines[:,1:] - streamlines[:,:-1]
    xi_speeds = np.abs(xi_velocities)#/ (np.max(np.abs(xi_velocities.ravel())) - np.min(np.abs(xi_velocities.ravel())))
    xi_minspeed = np.min(xi_speeds.ravel(), where=np.abs(streamlines[:,:-1].ravel()) < 2.5, initial=1e9)
    xi_maxspeed = np.max(xi_speeds.ravel(), where=np.abs(streamlines[:,:-1].ravel()) < 2.5, initial=1e-9)
    z_velocities = trefftz(streamlines[:,1:]) - trefftz(streamlines[:,:-1])
    z_speeds = np.abs(z_velocities)
    z_minspeed = np.min(z_speeds.ravel(), where=np.abs(trefftz(streamlines[:,:-1].ravel())) < 2.5, initial=1e9)
    z_maxspeed = np.max(z_speeds.ravel(), where=np.abs(trefftz(streamlines[:,:-1].ravel())) < 2.5, initial=1e-9)

    for ii in range(streamlines.shape[0]):
        sl = streamlines[ii]

        axl.plot(*cpts(sl[:-1]), c='k')
        axr.plot(*cpts(trefftz(sl[:-1])), c='k')


    axl.set_xlim([-3, 3])
    axl.set_ylim([-3, 3])
    axr.set_xlim([-3, 3])
    axr.set_ylim([-3, 3])

    axr.set_aspect('equal')
    axl.set_aspect('equal')

    fig.tight_layout()

    fig.show()


trefftz_plots(params)
