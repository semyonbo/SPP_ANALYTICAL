import numpy as np
import scipy
import frenel
from scipy.integrate import quad


def green_ref_00_integrand(kr, wl, z0, eps_interp):
    k = 2*np.pi/wl/1e-9
    kz = np.sqrt(k**2 - kr**2+0j)
    rpE, rpH, rsE, rsH = frenel.reflection_coeff(wl, eps_interp, kr)
    IntegrandE = np.array([
        [k**2*rsE - kz**2*rpE, 0, 0],
        [0, k**2*rsE-kz**2*rpE, 0],
        [0, 0, 2*kr**2*rpE]
    ], dtype=np.complex128)
    IntegrandH = np.array([
        [k**2*rpH - kz**2*rsH, 0, 0],
        [0, k**2*rpH-kz**2*rsH, 0],
        [0, 0, 2*kr**2*rsH]
    ], dtype=np.complex128)
    if np.abs(kz) == 0:
        return np.zeros((3, 3), dtype=np.complex128), np.zeros((3, 3), dtype=np.complex128)
    return kr*IntegrandE/kz*np.exp(2*1j*kz*z0), kr*IntegrandH/kz*np.exp(2*1j*kz*z0)


def green_ref_00(wl, z0, eps_interp, stop=np.inf, rel_tol=1e-3):
    k = 2*np.pi/wl/1e-9
    IntE = np.zeros((3, 3), dtype=np.complex128)
    IntH = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        IntE[i, i] = quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                          0][i, i], 0, np.inf, epsrel=rel_tol, complex_func=True)[0]
        IntH[i, i] = quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                          1][i, i], 0, np.inf, epsrel=rel_tol, complex_func=True)[0]
    G_ref_E = 1j * IntE/(8*np.pi*k**2)
    G_ref_H = 1j * IntH/(8*np.pi*k**2)
    return G_ref_E, G_ref_H


def rot_green_ref_00_integrand(kr, wl, z0, eps_interp):
    k = 2*np.pi/wl/1e-9
    kz = np.sqrt(k**2 - kr**2+0j)
    rpE, rpH, rsE, rsH = frenel.reflection_coeff(wl, eps_interp, kr)
    Integrand = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]
    ])
    return Integrand*kr*(rsE-rpE)*np.exp(2*1j*kz*z0)


def rot_green_ref_00(wl, z0, eps_interp, stop=np.inf, rel_tol=1e-3):
    IntE01 = quad(lambda kr: rot_green_ref_00_integrand(
        kr, wl, z0, eps_interp)[0, 1], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    IntE10 = quad(lambda kr: rot_green_ref_00_integrand(
        kr, wl, z0, eps_interp)[1, 0], 0, stop, epsrel=rel_tol, complex_func=True)[0]

    IntE = np.zeros((3, 3), dtype=np.complex128)
    IntE[0, 1] = IntE01
    IntE[1, 0] = IntE10
    rot_G_ref_E = IntE/(8*np.pi)
    rot_G_ref_H = -1 * IntE/(8*np.pi)
    return rot_G_ref_E, rot_G_ref_H