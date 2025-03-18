import numpy as np
import scipy
from MieSppForce import frenel
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


def green_ref_00(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    IntE = np.zeros((3, 3), dtype=np.complex128)
    IntH = np.zeros((3, 3), dtype=np.complex128)
    if stop==1:
        for i in range(3):
            IntE[i, i] = quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                            0][i, i], 0, 1, epsrel=rel_tol, complex_func=True)[0]
            IntH[i, i] = quad(lambda kr: green_ref_00_integrand(kr, wl, z0, eps_interp)[
                            1][i, i], 0, 1, epsrel=rel_tol, complex_func=True)[0]
    else:        
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


def rot_green_ref_00(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9

    if stop==1:
        IntE01 = quad(lambda kr: rot_green_ref_00_integrand(
            kr, wl, z0, eps_interp)[0, 1], 0, 1, epsrel=rel_tol, complex_func=True)[0]
        IntE10 = quad(lambda kr: rot_green_ref_00_integrand(
            kr, wl, z0, eps_interp)[1, 0], 0, 1, epsrel=rel_tol, complex_func=True)[0]
    else:      
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


def dy_green_integrand(kr, wl, eps_interp, z0):
    kz = sqrt(1 - kr**2) 
    rpE, _, rsE, _ = frenel.reflection_coeff(wl, eps_interp, kr)
    return rpE * kr**3 * np.exp(2*1j*kz*z0*2*np.pi/wl), rsE*kr**3*np.exp(2*1j*kz*z0*2*np.pi/wl)


def dy_green_E_H(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    Int_dy_dG_E_yz = quad(lambda kr: dy_green_integrand(kr, wl, eps_interp, z0)[
                          0], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    Int_dy_dG_H_yz = quad(lambda kr: dy_green_integrand(kr, wl, eps_interp, z0)[
                          1], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    
    dy_green_E = np.zeros((3, 3), dtype=np.complex128)
    dy_green_H = np.zeros((3, 3), dtype=np.complex128)
    dy_green_E[1,2] = Int_dy_dG_E_yz/(8*np.pi)*k**2
    dy_green_E[2,1] = -1*Int_dy_dG_E_yz/(8*np.pi)*k**2
    dy_green_H[1,2] = Int_dy_dG_H_yz/(8*np.pi)*k**2
    dy_green_H[2,1] = -1*Int_dy_dG_H_yz/(8*np.pi)*k**2
    
    return dy_green_E, dy_green_H


def dy_rot_green_H_integrand(kr, wl, eps_interp, z0):
    kz = sqrt(1 - kr**2)
    rpE, _, rsE, _ = frenel.reflection_coeff(wl, eps_interp, kr)
    # Inegrand = np.array([
    #     [0, 0, rsE],
    #     [0, 0, 0],
    #     [-rpE, 0, 0]
    # ])
    return kr**3*rsE*np.exp(2*1j*kz*z0*2*np.pi/wl)/kz, -1*kr**3*rpE*np.exp(2*1j*kz*z0*2*np.pi/wl)/kz


def dy_rot_green_E_H(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    if stop == 1:
        Int_dy_rotG_H_xz = quad(lambda kr: dy_rot_green_H_integrand(kr, wl, eps_interp, z0)[0], 0, 1, epsrel=rel_tol, complex_func=True)[
        0]
        Int_dy_rotG_H_zx = quad(lambda kr: dy_rot_green_H_integrand(kr, wl, eps_interp, z0)[1], 0, 1, epsrel=rel_tol, complex_func=True)[
        0]
    else:
        Int_dy_rotG_H_xz = quad(lambda kr: dy_rot_green_H_integrand(kr, wl, eps_interp, z0)[0], 0, 1, epsrel=rel_tol, complex_func=True)[
            0]+quad(lambda kr: dy_rot_green_H_integrand(kr, wl, eps_interp, z0)[0], 1, stop, epsrel=rel_tol, complex_func=True)[0]
        Int_dy_rotG_H_zx = quad(lambda kr: dy_rot_green_H_integrand(kr, wl, eps_interp, z0)[1], 0, 1, epsrel=rel_tol, complex_func=True)[
            0]+quad(lambda kr: dy_rot_green_H_integrand(kr, wl, eps_interp, z0)[1], 1, stop, epsrel=rel_tol, complex_func=True)[0]
    dy_drotG_H = np.zeros((3, 3), dtype=np.complex128)
    dy_drotG_E = np.zeros((3, 3), dtype=np.complex128)
    dy_drotG_H[0,2] = -1j*Int_dy_rotG_H_xz/(8*np.pi)*k**3
    dy_drotG_H[2,0] = -1j*Int_dy_rotG_H_zx/(8*np.pi)*k**3
    
    dy_drotG_E[0,2] = 1j*Int_dy_rotG_H_zx/(8*np.pi)*k**3
    dy_drotG_E[2,0] = 1j*Int_dy_rotG_H_xz/(8*np.pi)*k**3
    
    return dy_drotG_E, dy_drotG_H





def dx_green_integrand(kr, wl, eps_interp, z0):
    kz = sqrt(1 - kr**2)
    rpE, _, rsE, _ = frenel.reflection_coeff(wl, eps_interp, kr)
    return rpE*kr**3*np.exp(2*1j*kz*z0*2*np.pi/wl), rsE*kr**3*np.exp(2*1j*kz*z0*2*np.pi/wl)

def dx_green_E_H(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    Int_dx_dG_E_xz = quad(lambda kr: dx_green_integrand(kr, wl, eps_interp, z0)[
                          0], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    Int_dx_dG_H_xz = quad(lambda kr: dx_green_integrand(kr, wl, eps_interp, z0)[
                          1], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    
    dx_green_E = np.zeros((3, 3), dtype=np.complex128)
    dx_green_H = np.zeros((3, 3), dtype=np.complex128)
    
    dx_green_E[0,2] = Int_dx_dG_E_xz/(8*np.pi)*k**2
    dx_green_E[2,0] = -1*Int_dx_dG_E_xz/(8*np.pi)*k**2
    dx_green_H[0,2] = Int_dx_dG_H_xz/(8*np.pi)*k**2
    dx_green_H[2,0] = -1*Int_dx_dG_H_xz/(8*np.pi)*k**2
    
    
    return dx_green_E, dx_green_H

def dx_rot_green_H_integrand(kr, wl, eps_interp, z0):
    kz = sqrt(1 - kr**2)
    rpE, _, rsE, _ = frenel.reflection_coeff(wl, eps_interp, kr)
    # Inegrand = np.array([
    #     [0, 0, 0],
    #     [0, 0, -rsE],
    #     [0, rpE, 0]
    # ])
    return -1*kr**3*rsE*np.exp(2*1j*kz*z0*2*np.pi/wl)/kz, kr**3*rpE*np.exp(2*1j*kz*z0*2*np.pi/wl)/kz

def dx_rot_green_E_H(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    if stop==1:
        Int_dx_rotH_zy = quad(lambda kr: dx_rot_green_H_integrand(kr, wl, eps_interp, z0)[0], 0, 1, epsrel=rel_tol, complex_func=True)[0]
        Int_dx_rotH_yz = quad(lambda kr: dx_rot_green_H_integrand(kr, wl, eps_interp, z0)[1], 0, 1, epsrel=rel_tol, complex_func=True)[0]
    else:
        Int_dx_rotH_zy = quad(lambda kr: dx_rot_green_H_integrand(kr, wl, eps_interp, z0)[0], 0, 1, epsrel=rel_tol, complex_func=True)[
            0]+quad(lambda kr: dx_rot_green_H_integrand(kr, wl, eps_interp, z0)[0], 1, stop, epsrel=rel_tol, complex_func=True)[0]
        Int_dx_rotH_yz = quad(lambda kr: dx_rot_green_H_integrand(kr, wl, eps_interp, z0)[1], 0, 1, epsrel=rel_tol, complex_func=True)[
            0]+quad(lambda kr: dx_rot_green_H_integrand(kr, wl, eps_interp, z0)[1], 1, stop, epsrel=rel_tol, complex_func=True)[0]
    dx_drotG_H = np.zeros((3, 3), dtype=np.complex128)
    dx_drotG_E = np.zeros((3, 3), dtype=np.complex128)
    dx_drotG_H[1,2] = -1j*Int_dx_rotH_zy/(8*np.pi)*k**3
    dx_drotG_H[2,1] = -1j*Int_dx_rotH_yz/(8*np.pi)*k**3
    
    dx_drotG_E[1,2] = 1j*Int_dx_rotH_yz/(8*np.pi)*k**3
    dx_drotG_E[2,1] = 1j*Int_dx_rotH_zy/(8*np.pi)*k**3
    
    return dx_drotG_E, dx_drotG_H


def dz_green_integrand(kr, wl, eps_interp, z0):
    kz = sqrt(1 - kr**2) 
    rpE, _, rsE, _ = frenel.reflection_coeff(wl, eps_interp, kr)
    return (rsE - rpE*kz**2) * kr * np.exp(2*1j*kz*z0*2*np.pi/wl), 2*rpE*kr**3*np.exp(2*1j*kz*z0*2*np.pi/wl), (rpE - rsE*kz**2) * kr * np.exp(2*1j*kz*z0*2*np.pi/wl), 2*rsE*kr**3*np.exp(2*1j*kz*z0*2*np.pi/wl)

def dz_green_E_H(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    
    Int_dz_green_E_xx = quad(lambda kr: dz_green_integrand(kr, wl, eps_interp, z0)[0], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    Int_dz_green_E_zz = quad(lambda kr: dz_green_integrand(kr, wl, eps_interp, z0)[1], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    
    Int_dz_green_H_xx = quad(lambda kr: dz_green_integrand(kr, wl, eps_interp, z0)[2], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    Int_dz_green_H_zz = quad(lambda kr: dz_green_integrand(kr, wl, eps_interp, z0)[3], 0, stop, epsrel=rel_tol, complex_func=True)[0]
    
    
    dz_dG_H = np.zeros((3, 3), dtype=np.complex128)
    dz_dG_E = np.zeros((3, 3), dtype=np.complex128)
    
    dz_dG_E[0,0] = -1*Int_dz_green_E_xx *k**2 /(8*np.pi)
    dz_dG_E[1,1] = -1*Int_dz_green_E_xx *k**2 /(8*np.pi)
    dz_dG_E[2,2] = -1*Int_dz_green_E_zz *k**2 /(8*np.pi)
    
    dz_dG_H[0,0] = -1*Int_dz_green_H_xx *k**2 /(8*np.pi)
    dz_dG_H[1,1] = -1*Int_dz_green_H_xx *k**2 /(8*np.pi)
    dz_dG_H[2,2] = -1*Int_dz_green_H_zz *k**2 /(8*np.pi)
    
    return dz_dG_E, dz_dG_H

def dz_rot_green_integrand(kr, wl, eps_interp, z0):
    kz = sqrt(1 - kr**2) 
    rpE, _, rsE, _ = frenel.reflection_coeff(wl, eps_interp, kr)
    return kz*kr*(rsE-rpE) * np.exp(2*1j*kz*z0*2*np.pi/wl)
    
def dz_rot_green_E_H(wl, z0, eps_interp, stop, rel_tol=1e-8):
    k = 2*np.pi/wl/1e-9
    
    Int_dz__rot_green_E = quad(lambda kr: dz_rot_green_integrand(kr, wl, eps_interp, z0), 0, stop, epsrel=rel_tol, complex_func=True)[0]
    
    dz_drotG_H = np.zeros((3, 3), dtype=np.complex128)
    dz_drotG_E = np.zeros((3, 3), dtype=np.complex128)
    
    dz_drotG_E[0,1] = Int_dz__rot_green_E*1j/(8*np.pi)*k**3
    dz_drotG_E[1,0] = -1*Int_dz__rot_green_E*1j/(8*np.pi)*k**3
    dz_drotG_H[0,1] = -1*Int_dz__rot_green_E*1j/(8*np.pi)*k**3
    dz_drotG_H[1,0] = Int_dz__rot_green_E*1j/(8*np.pi)*k**3
    
    return dz_drotG_E, dz_drotG_H
    
    
    
    