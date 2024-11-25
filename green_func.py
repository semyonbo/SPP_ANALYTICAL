import numpy as np
import scipy
import frenel
from scipy.integrate import quad
from cmath import sqrt


def green_ref_00_integrand(kr, wl, z0, eps_interp):
    k = 1
    kz = sqrt(1 - kr**2)
    rpE, rpH, rsE, rsH = frenel.reflection_coeff(wl, eps_interp, kr)
    IntegrandE = np.array([
        [k**2*rsE - kz**2*rpE, 0, 0],
        [0, k**2*rsE-kz**2*rpE, 0],
        [0, 0, 2*kr**2*rpE]
    ], dtype=np.complex128)
    IntegrandH = np.array([
        [k**2*rpE - kz**2*rsE, 0, 0],
        [0, k**2*rpE-kz**2*rsE, 0],
        [0, 0, 2*kr**2*rsE]
    ], dtype=np.complex128)
    return kr*IntegrandE/kz*np.exp(2*1j*kz*z0*2*np.pi/wl), kr*IntegrandH/kz*np.exp(2*1j*kz*z0*2*np.pi/wl)


def green_ref_00(wl, z0, eps_interp, stop=100000, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    IntE = np.zeros((3, 3), dtype=np.complex128)
    IntH = np.zeros((3, 3), dtype=np.complex128)
    for i in range(3):
        IntE[i, i] = quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                          0][i, i], 0, 1, epsrel=rel_tol, complex_func=True)[0]\
            + quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                0][i, i], 1, stop, epsrel=rel_tol, complex_func=True)[0]
        IntH[i, i] = quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                          1][i, i], 0, 1, epsrel=rel_tol, complex_func=True)[0]\
            + quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                1][i, i], 1, stop, epsrel=rel_tol, complex_func=True)[0]
    G_ref_E = 1j * IntE/(8*np.pi)*k
    G_ref_H = 1j * IntH/(8*np.pi)*k
    return G_ref_E, G_ref_H


def rot_green_ref_00_integrand(kr, wl, z0, eps_interp):
    kz = sqrt(1 - kr**2)
    rpE, rpH, rsE, rsH = frenel.reflection_coeff(wl, eps_interp, kr)
    Integrand = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]
    ])
    return Integrand*kr*(rsE-rpE)*np.exp(2*1j*kz*z0*2*np.pi/wl)


def rot_green_ref_00(wl, z0, eps_interp, stop=10, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9

    IntE01 = quad(lambda kr: rot_green_ref_00_integrand(
        kr, wl, z0, eps_interp)[0, 1], 0, 1, epsrel=rel_tol, complex_func=True)[0]\
        + quad(lambda kr: rot_green_ref_00_integrand(
            kr, wl, z0, eps_interp)[0, 1], 1, stop, epsrel=rel_tol, complex_func=True)[0]
    IntE10 = quad(lambda kr: rot_green_ref_00_integrand(
        kr, wl, z0, eps_interp)[1, 0], 0, 1, epsrel=rel_tol, complex_func=True)[0]\
        + quad(lambda kr: rot_green_ref_00_integrand(
            kr, wl, z0, eps_interp)[1, 0], 1, stop, epsrel=rel_tol, complex_func=True)[0]

    IntE = np.zeros((3, 3), dtype=np.complex128)
    IntE[0, 1] = IntE01
    IntE[1, 0] = IntE10
    rot_G_ref_E = IntE/(8*np.pi)*k**2
    rot_G_ref_H = -1 * IntE/(8*np.pi)*k**2
    return rot_G_ref_E, rot_G_ref_H
