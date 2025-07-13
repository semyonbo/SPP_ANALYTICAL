from MieSppForce import green_func, dipoles, frenel
import numpy as np
from cmath import sqrt
from functools import lru_cache

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


def get_field(wl, eps_interp, alpha, phase, a_angle, stop, eps_particle, R,   r, phi, z, z0 ):
    k = 2*np.pi/wl*1e9
    omega = k*c_const
    amplitude = 1  
    GE = green_func.getGE(wl, eps_interp, z0, r, phi, z, stop)
    GH = green_func.getGH(wl, eps_interp, z0, r, phi, z, stop)
    rotGE = green_func.get_rotGE(wl, eps_interp, z0, r, phi, z, stop)
    rotGH = green_func.get_rotGH(wl, eps_interp, z0, r, phi, z, stop)
    p,m = dipoles.calc_dipoles_v2(wl, eps_interp, [0,0,z0], R, eps_particle, alpha, amplitude, phase, a_angle, stop)

    E =  k**2/eps0_const * GE @ p + 1j*omega*mu0_const*rotGH@m
    H =  k**2 * GH @ m - 1j*omega*rotGE @ p
    
    # S =np.real( 0.5 * np.cross(E[:,0], H[:,0].conj()))
    
    # Sr = np.sqrt(S[0]**2 + S[1]**2)
    
    return E[:,0],H[:,0]