import numpy as np
from scipy.integrate import quad
c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


def get_directivity(wl, P, M, eps_Au):
    px, py, pz = P
    mx, my, mz = M

    def I(phi): return np.abs((mx/c_const + np.sqrt(1/(eps_Au(wl)+1))*py)*np.sin(phi) + (my/c_const - np.sqrt(1/(eps_Au(wl)+1))*px)*np.cos(phi) - np.sqrt(eps_Au(wl)/(eps_Au(wl)+1))*pz)**2
    Imax = quad(I, 0, 2 * np.pi)[0]
    return lambda phi: 2 * np.pi * I(phi) / Imax
