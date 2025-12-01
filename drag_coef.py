import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 22})

def optimal_angle_direct_wind(a):
    return np.arccos(2/(-a-np.sqrt(np.square(a)+2)))

a_s = np.linspace(2, 10, 256)

fig, ax = plt.subplots()
ax.plot(a_s, optimal_angle_direct_wind(a_s), linewidth=4)
ax.set_xlabel('$a$')
ax.set_ylabel('$\\beta$ (radians)')
ax.set_yticks([5/12 * np.pi, np.pi/2, np.pi * 7/12, 2*np.pi/3], ['$\\frac{5}{12}\\pi$', '$\\frac{1}{2}\\pi$','$\\frac{7}{12}\\pi$','$\\frac{2}{3}\\pi$'])
ax.axhline(np.pi/2, linewidth=4, c='k')
fig.set_size_inches(6, 6)
fig.tight_layout()
plt.show()
