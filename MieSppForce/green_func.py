import numpy as np
import scipy
from MieSppForce import frenel
from scipy.integrate import quad
from cmath import sqrt
from functools import lru_cache
from scipy.special import j0, j1
from scipy.special import jn
import warnings
from scipy.integrate import IntegrationWarning
from numpy import sin,cos

# warnings.filterwarnings("ignore", category=IntegrationWarning)

eps_val = np.finfo(float).eps

def green_ref_00_integrand(kr, wl, z0, eps_interp):
    k = 1
    kz = sqrt(1 - kr**2)
    rpE, _, rsE, _ = frenel.reflection_coeff(wl, eps_interp, kr)
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)    
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


@lru_cache(maxsize=None) 
def get_GE_int(wl, eps_interp, h_nm, r_nm, stop):
    k = 2*np.pi/wl*1e9
    h=h_nm*1e-9
    r=r_nm*1e-9
    #j2 = lambda x: 2*j1(x)/x - j0(x)
    j2 = lambda x: jn(2,x)
    rs = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[2]
    rp = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[0]
    kz = lambda kr: k*sqrt(1 - kr**2)
    exp_fac = lambda kr: np.exp(1j*kz(kr)*h)
    
    #k_spp = np.sqrt(eps_interp(wl)/(eps_interp(wl)+1)).real
    k_spp =1
    
    int_GExx1 = k*quad(lambda kr: exp_fac(kr)/kz(kr) * (kz(kr)**2 * rp(kr) + k**2 * rs(kr) ) * j1(kr*k*r), 0, stop, complex_func=True, points=[1,k_spp])[0]
    int_GExx2 = k*quad(lambda kr: exp_fac(kr) * kz(kr) * kr*k*j0(kr*k*r) *rp(kr), 0, stop, complex_func=True, points=[1,k_spp])[0]
    int_GExx3 = k*quad(lambda kr: exp_fac(kr)/kz(kr) * kr*k*j0(kr*k*r) * k**2 * rs(kr),0, stop, complex_func=True, points=[1,k_spp])[0]
    int_GExy = k*quad(lambda kr: kr*k/kz(kr) * (kz(kr)**2 * rp(kr) + k**2 * rs(kr))*j2(kr*k*r)*exp_fac(kr),0, stop, complex_func=True, points=[1,k_spp])[0]
    int_GExz = k*quad(lambda kr: exp_fac(kr)*(kr*k)**2 *rp(kr)* j1(kr*k*r),0, stop, complex_func=True, points=[1,k_spp])[0]
    int_GEzz = k*quad(lambda kr: exp_fac(kr)*(kr*k)**3*rp(kr)*j0(kr*r*k)/kz(kr),0, stop, complex_func=True, points=[1,k_spp])[0]
    
    return int_GExx1, int_GExx2, int_GExx3, int_GExy, int_GExz, int_GEzz

def getGE(wl, eps_interp, z0_nm, r_nm, phi, z_nm, stop):
    assert r_nm != 0, "Ошибка: радиус r не должен быть равен нулю (используй частный случай)."
    k = 2*np.pi/wl*1e9
    h_nm = z_nm+z0_nm
    r = r_nm*1e-9
    r_safe = np.maximum(r, eps_val)
    int_GExx1, int_GExx2, int_GExx3, int_GExy, int_GExz, int_GEzz = get_GE_int(wl, eps_interp, h_nm, r_nm, stop)
    
    GExx = -1j/(4*np.pi*k**2) *( -int_GExx1 * np.cos(2*phi)/r_safe  + int_GExx2 * np.cos(phi)**2 -int_GExx3*np.sin(phi)**2)
    GExy = 1j*np.sin(2*phi)/(8*np.pi*k**2) * int_GExy
    GExz = np.cos(phi)/(4*np.pi*k**2) * int_GExz
    GEyx = GExy
    GEyy = -1j/(4*np.pi*k**2)* (int_GExx1*np.cos(2*phi)/r_safe + int_GExx2 * np.sin(phi)**2  - int_GExx3*np.cos(phi)**2)
    GEyz = np.sin(phi)/(4*np.pi*k**2) * int_GExz
    GEzx = - GExz
    GEzy = - GEyz
    GEzz = 1j/(4*np.pi*k**2) * int_GEzz
    
    GE = np.array([[GExx, GExy, GExz],
                   [GEyx, GEyy, GEyz],
                   [GEzx, GEzy, GEzz]], dtype=np.complex128)
    return GE
    
    
def cal_GE_slow(wl, eps_interp, z0_nm, r_nm, phi, z_nm, stop):
    k = 2*np.pi/wl*1e9
    Pi = np.pi
    r = r_nm*1e-9
    r_safe = np.maximum(r, eps_val)
    h_nm = z_nm+z0_nm
    h=h_nm*1e-9
    r=r_nm*1e-9
    
    #k_spp = np.sqrt(eps_interp(wl)/(eps_interp(wl)+1)).real
    k_spp =1
    
    j2 = lambda x: jn(2,x)
    rs = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[2]
    rp = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[0]
    kz = lambda kr: k*sqrt(1 - kr**2)
    
    
    exp_fac = lambda kr: np.exp(1j*kz(kr)*h)

 
 
    intGExx = lambda kr: -1j*exp_fac(kr)/(4*Pi*k**2*kz(kr)*r_safe) *(-1*((kz(kr)**2 * rp(kr) + k**2 * rs(kr)) *j1(kr*k*r_safe) * cos(2*phi) ) + kr*k*r_safe*j0(kr*k*r_safe) * (kz(kr)**2 * rp(kr)*cos(phi)**2- k**2 * rs(kr)*sin(phi)**2))

    intGExy = lambda kr: 1j*exp_fac(kr) * kr*k*(kz(kr)**2 * rp(kr) + k**2 * rs(kr)) * j2(kr*k*r_safe) * sin(2*phi) / (8 * k**2 * kz(kr)*Pi)
    
    intGExz = lambda kr: exp_fac(kr) * (kr*k)**2 * rp(kr) * j1(kr*k*r_safe)  * cos(phi)/ (4 * k**2 * Pi)
    
    intGEyx = lambda kr: 1j * exp_fac(kr) * kr*k * (kz(kr)**2 * rp(kr)+k**2*rs(kr)) * j2(kr*k*r_safe) * sin(2*phi) / (8 * k**2 *Pi * kz(kr))
    
    intGEyy = lambda kr: -1j*exp_fac(kr)/(4*Pi*k**2*kz(kr)*r_safe) * (( kz(kr)**2 * rp(kr) + k**2 * rs(kr)) * j1(kr*k*r_safe)*cos(2*phi)+ kr*k*r_safe*j0(kr*k*r_safe) * (-k**2*rs(kr)*cos(phi)**2+kz(kr)**2*rp(kr)*sin(phi)**2) )
    
    intGEyz = lambda kr: exp_fac(kr)* (kr*k)**2 *rp(kr) * j1(kr*k*r_safe) * sin(phi)/(4*Pi*k**2)
    
    intGEzx = lambda kr: -1 * exp_fac(kr) * (kr*k)**2 * rp(kr) * j1(kr*k*r_safe) * cos(phi) / (4 * k**2 *Pi)
    
    intGEzy = lambda kr: -1 * exp_fac(kr) * (kr*k)**2 * rp(kr) * j1(kr*k*r_safe) * sin(phi) / (4*k**2 * Pi)
    
    intGEzz = lambda kr: 1j*exp_fac(kr) * (kr*k)**3 * rp(kr) * j0(kr*k*r_safe) /(4*k**2 * kz(kr)*Pi)
    
    GExx = k*quad( intGExx, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GExy = k*quad( intGExy, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GExz = k*quad( intGExz, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GEyx = k*quad( intGEyx, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GEyy = k*quad( intGEyy, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GEyz = k*quad( intGEyz, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GEzx = k*quad( intGEzx, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GEzy = k*quad( intGEzy, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    GEzz = k*quad( intGEzz, 0, stop, epsrel=1e-8, complex_func=True, points=[1,k_spp])[0]
    

    GE = np.array([[GExx, GExy, GExz],
                   [GEyx, GEyy, GEyz],
                   [GEzx, GEzy, GEzz]], dtype=np.complex128)
    return GE
    

@lru_cache(maxsize=None) 
def get_GH_int(wl, eps_interp, h_nm, r_nm, stop):
    k = 2*np.pi/wl*1e9
    h=h_nm*1e-9
    r=r_nm*1e-9
    #j2 = lambda x: 2*j1(x)/x - j0(x)
    j2 = lambda x: jn(2,x)
    rs = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[2]
    rp = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[0]
    kz = lambda kr: k*sqrt(1 - kr**2)
    exp_fac = lambda kr: np.exp(1j*kz(kr)*h)
    
    int_GHxx1 = k*quad(lambda kr: exp_fac(kr)/kz(kr) * (kz(kr)**2 * rs(kr) + k**2 * rp(kr) ) * j1(kr*k*r), 0, stop, complex_func=True)[0]
    int_GHxx2 = k*quad(lambda kr: exp_fac(kr) * kz(kr) * kr*k*j0(kr*k*r) *rs(kr), 0, stop, complex_func=True)[0]
    int_GHxx3 = k*quad(lambda kr: exp_fac(kr)/kz(kr) * kr*k*j0(kr*k*r) * k**2 * rp(kr),0, stop, complex_func=True)[0]
    int_GHxy = k*quad(lambda kr: kr*k/kz(kr) * (kz(kr)**2 * rs(kr) + k**2 * rp(kr))*j2(kr*k*r)*exp_fac(kr),0, stop, complex_func=True)[0]
    int_GHxz = k*quad(lambda kr: exp_fac(kr)*(kr*k)**2 *rs(kr)* j1(kr*k*r),0, stop, complex_func=True)[0]
    int_GHzz = k*quad(lambda kr: exp_fac(kr)*(kr*k)**3*rs(kr)*j0(kr*r*k)/kz(kr),0, stop, complex_func=True)[0]
    
    return int_GHxx1, int_GHxx2, int_GHxx3, int_GHxy, int_GHxz, int_GHzz

def getGH(wl, eps_interp, z0_nm, r_nm, phi, z_nm, stop):
    assert r_nm != 0, "Ошибка: радиус r не должен быть равен нулю (используй частный случай)."
    k = 2*np.pi/wl*1e9
    h_nm = z_nm+z0_nm
    r = r_nm*1e-9
    r_safe = np.maximum(r, eps_val)
    int_GHxx1, int_GHxx2, int_GHxx3, int_GHxy, int_GHxz, int_GHzz = get_GH_int(wl, eps_interp, h_nm, r_nm, stop)
    
    GHxx = -1j/(4*np.pi*k**2) *( -int_GHxx1 * np.cos(2*phi)/r_safe  + int_GHxx2 * np.cos(phi)**2 -int_GHxx3*np.sin(phi)**2)
    GHxy = 1j*np.sin(2*phi)/(8*np.pi*k**2) * int_GHxy
    GHxz = np.cos(phi)/(4*np.pi*k**2) * int_GHxz
    GHyx = GHxy
    GHyy = -1j/(4*np.pi*k**2)* (int_GHxx1*np.cos(2*phi)/r_safe + int_GHxx2 * np.sin(phi)**2  - int_GHxx3*np.cos(phi)**2)
    GHyz = np.sin(phi)/(4*np.pi*k**2) * int_GHxz
    GHzx = - GHxz
    GHzy = - GHyz
    GHzz = 1j/(4*np.pi*k**2) * int_GHzz
    
    GH = np.array([[GHxx, GHxy, GHxz],
                   [GHyx, GHyy, GHyz],
                   [GHzx, GHzy, GHzz]], dtype=np.complex128)
    return GH


@lru_cache(maxsize=None) 
def get_rotGE_int(wl, eps_interp, h_nm, r_nm, stop):
    k = 2*np.pi/wl*1e9
    h=h_nm*1e-9
    r=r_nm*1e-9
    j2 = lambda x: jn(2,x)
    rs = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[2]
    rp = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[0]
    kz = lambda kr: k*sqrt(1 - kr**2)
    exp_fac = lambda kr: np.exp(1j*kz(kr)*h)
    
    int1 = k*quad(lambda kr: exp_fac(kr)*kr*k*rp(kr)*j2(kr*r*k), 0, stop, complex_func=True)[0]
    int2 = k*quad(lambda kr: exp_fac(kr)*kr*k*rs(kr)*j2(kr*k*r), 0, stop, complex_func=True)[0]
    int3 = k*quad(lambda kr: exp_fac(kr)*(rp(kr)+rs(kr))*j1(kr*r*k), 0, stop, complex_func=True)[0]
    int4 = k*quad(lambda kr: exp_fac(kr)*kr*k*rs(kr)*j0(kr*k*r) , 0, stop, complex_func=True)[0]
    int5 = k*quad(lambda kr: exp_fac(kr)*kr*k*rp(kr)*j0(kr*k*r), 0, stop, complex_func=True)[0]
    int6 = k*quad(lambda kr: exp_fac(kr)*(kr*k)**2/kz(kr)*rp(kr)*j1(kr*k*r), 0, stop, complex_func=True)[0]
    int7 = k*quad(lambda kr: exp_fac(kr)*(kr*k)**2/kz(kr)*rs(kr)*j1(kr*k*r), 0, stop, complex_func=True)[0]
    
    return int1, int2, int3, int4, int5, int6, int7

def get_rotGE(wl, eps_interp, z0_nm, r_nm, phi, z_nm, stop):
    k = 2*np.pi/wl*1e9
    h_nm = z_nm+z0_nm
    r = r_nm*1e-9
    r_safe = np.maximum(r, eps_val)
    
    int1, int2, int3, int4, int5, int6, int7 = get_rotGE_int(wl, eps_interp, h_nm, r_nm, stop)
    
    rotGExx = np.sin(2*phi)/(8*np.pi) * (int1+int2)
    rotGExy = -1*np.cos(2*phi)/(4*np.pi*r_safe) * int3 + np.cos(phi)**2/(4*np.pi) *int4 - np.sin(phi)**2/(4*np.pi) * int5
    rotGExz = - np.sin(phi)*1j/(4*np.pi) * int6
    rotGEyx = 1/(8*np.pi)*(int5-int4) - np.cos(2*phi)/(8*np.pi)*(int1+int2)
    rotGEyy = -np.sin(2*phi)/(8*np.pi)*(int1+int2)
    rotGEyz = 1j*np.cos(phi)/(4*np.pi) * int6
    rotGEzx = 1j*np.sin(phi)/(4*np.pi) * int7
    rotGEzy = - 1j*np.cos(phi)/(4*np.pi) * int7
    
    rotGE = np.array( [[rotGExx, rotGExy, rotGExz],
                       [rotGEyx, rotGEyy, rotGEyz],
                       [rotGEzx, rotGEzy, 0]], dtype=complex)

    return rotGE


def get_rotGH(wl, eps_interp, z0_nm, r_nm, phi, z_nm, stop):
    k = 2*np.pi/wl*1e9
    h_nm = z_nm+z0_nm
    r = r_nm*1e-9
    r_safe = np.maximum(r, eps_val)
    
    int1, int2, int3, int4, int5, int6, int7 = get_rotGE_int(wl, eps_interp, h_nm, r_nm, stop)
    
    rotGExx = np.sin(2*phi)/(8*np.pi) * (int1+int2)
    rotGExy = -1*np.cos(2*phi)/(4*np.pi*r_safe) * int3 + np.cos(phi)**2/(4*np.pi) *int5- np.sin(phi)**2/(4*np.pi) * int4
    rotGExz = - np.sin(phi)*1j/(4*np.pi) * int7
    rotGEyx = 1/(8*np.pi)*(int4-int5) - np.cos(2*phi)/(8*np.pi)*(int1+int2)
    rotGEyy = -np.sin(2*phi)/(8*np.pi)*(int1+int2)
    rotGEyz = 1j*np.cos(phi)/(4*np.pi) * int7
    rotGEzx = 1j*np.sin(phi)/(4*np.pi) * int6
    rotGEzy = - 1j*np.cos(phi)/(4*np.pi) * int6
    
    rotGE = np.array( [[rotGExx, rotGExy, rotGExz],
                       [rotGEyx, rotGEyy, rotGEyz],
                       [rotGEzx, rotGEzy, 0]], dtype=complex)

    return rotGE

# @lru_cache(maxsize=None) 
# def get_rotGEs_int(wl, eps_interp, h_nm, r_nm, stop):
#     k0 = 2*np.pi/wl*1e9
#     h=h_nm*1e-9
#     r=r_nm*1e-9
#     j2 = lambda x: 2*j1(x)/x - j0(x)
#     rs = lambda kr : frenel.reflection_coeff(wl, eps_interp, kr)[2]
#     kz = lambda kr: k0*sqrt(1 - kr**2)
#     exp_fac = lambda kr: np.exp(1j*kz(kr)*h)

#     int_rotGE_xx = quad( lambda kr: exp_fac(kr)* kr*k0*rs(kr) * j2(kr*r*k0) , 0, stop, complex_func=True)[0]
#     int_rotGE_xy1 = quad( lambda kr: exp_fac(kr)* rs(kr) * j1(kr*r*k0) , 0, stop, complex_func=True)[0]
#     int_rotGE_xy2 = quad( lambda kr: exp_fac(kr)*kr*k0*rs(kr)*j0(kr*r*k0)  , 0, stop, complex_func=True)[0]
#     int_rotGE_zx = quad( lambda kr: exp_fac(kr)*(kr*k0)**2*rs(kr)*j1(kr*r*k0)/kz(kr) , 0, stop, complex_func=True)[0]

#     return int_rotGE_xx, int_rotGE_xy1, int_rotGE_xy2, int_rotGE_zx

# def get_rotGEs(wl, eps_interp, z0_nm, r_nm, phi, z_nm, stop):
#     h_nm = z_nm+z0_nm
#     r = r_nm*1e-9
#     k0 = 2*np.pi/wl*1e9
#     int_rotGE_xx, int_rotGE_xy1, int_rotGE_xy2, int_rotGE_zx = get_rotGEs_int(wl, eps_interp, h_nm, r_nm, stop)
    
#     return None
    
