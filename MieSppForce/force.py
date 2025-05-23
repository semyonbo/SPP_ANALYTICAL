import numpy as np
import scipy as sc
from MieSppForce import green_func
from MieSppForce import frenel
from MieSppForce import dipoles
from functools import lru_cache


c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7

# #add cache
@lru_cache(maxsize=None)
def cached_green_functions(wl, z0, eps_Au, stop):
    dx_G_E, dx_G_H = green_func.dx_green_E_H(wl, z0, eps_Au, stop)
    dx_rot_G_E, dx_rot_G_H = green_func.dx_rot_green_E_H(wl, z0, eps_Au, stop)
    
    dy_G_E, dy_G_H = green_func.dy_green_E_H(wl, z0, eps_Au, stop)
    dy_rot_G_E, dy_rot_G_H = green_func.dy_rot_green_E_H(wl, z0, eps_Au, stop)
    
    dz_G_E, dz_G_H = green_func.dz_green_E_H(wl, z0, eps_Au, stop)
    dz_rot_G_E, dz_rot_G_H = green_func.dz_rot_green_E_H(wl, z0, eps_Au, stop)
    return (dx_G_E, dx_G_H, dx_rot_G_E, dx_rot_G_H,
            dy_G_E, dy_G_H, dy_rot_G_E, dy_rot_G_H,
            dz_G_E, dz_G_H, dz_rot_G_E, dz_rot_G_H)
    


def F(wl, eps_Au, point,R, eps_si, alpha, amplitude, phase, a_angle ,stop, full_output=False, stop_dipoles=None):
    mu=1
    eps=1
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    x0,y0,z0=point
    
    if stop_dipoles != None:
        dip = dipoles.calc_dipoles_v2(wl,eps_Au, point,R,eps_si, alpha, amplitude, phase, a_angle, stop_dipoles)
    else:
        dip = dipoles.calc_dipoles_v2(wl,eps_Au, point,R,eps_si, alpha, amplitude, phase, a_angle, stop)
        
    p = dip[0][:,0]
    m = dip[1][:,0]
    E0, H0 = dipoles.initial_field(wl, alpha, amplitude, eps_Au, point, phase, a_angle)
    E0 = E0[:,0]
    H0 = H0[:,0]
    kx = k*np.sin(alpha)
    rp, rs = frenel.reflection_coeff_v2(wl, eps_Au, alpha)
    zm0 = z0*1e-9
    A = np.sin(a_angle)
    B = np.cos(a_angle)
    Phase = np.exp(1j*phase)
    kz = k * np.cos(alpha)
    
    dz_dEx = B * Phase *amplitude * np.cos(alpha) * (-1j*kz * np.exp(-1j*kz*zm0) - rp * 1j* kz*np.exp(1j*kz*zm0))
    dz_dEy = A * amplitude * (-1j*kz*np.exp(-1j*kz*zm0) + rs * np.exp(1j*kz*zm0)*1j*kz)
    dz_dEz = B * Phase * amplitude * np.sin(alpha) * (-1j*kz * np.exp(-1j*kz*zm0) + rp * 1j* kz*np.exp(1j*kz*zm0))
    
    dz_dE = np.array([[dz_dEx],
                     [dz_dEy],
                     [dz_dEz]])
    
    dz_dHx = k* A *amplitude*np.cos(alpha) / omega/mu0_const *(-1j*kz*np.exp(-1j*kz*zm0) - rs * np.exp(1j*kz*zm0)*1j*kz)
    dz_dHy = -k * B * Phase * amplitude  / omega / mu0_const *(-1j*kz*np.exp(-1j*kz*zm0) + rp * np.exp(1j*kz*zm0)*1j*kz)
    dz_dHz = A * k*amplitude*np.sin(alpha) / omega/mu0_const *(-1j*kz*np.exp(-1j*kz*zm0) + rs * np.exp(1j*kz*zm0)*1j*kz )
    
    dz_dH = np.array([[dz_dHx],
                     [dz_dHy],
                     [dz_dHz]])
    
    dz_dE=dz_dE[:,0]
    dz_dH=dz_dH[:,0]

    (dx_G_E, dx_G_H, dx_rot_G_E, dx_rot_G_H,
     dy_G_E, dy_G_H, dy_rot_G_E, dy_rot_G_H,
     dz_G_E, dz_G_H, dz_rot_G_E, dz_rot_G_H) = cached_green_functions(wl, z0, eps_Au, stop)
    
    F_crest = - k**4/(12*np.pi*c_const*eps0_const) * np.real( np.cross(p, np.conj(m)))
    
    Fy_e1 = 0.5*mu*k**2/eps0_const * np.real( np.conj(p) @ (dy_G_E @ p))
    Fy_e2 = 0.5*omega*mu*mu0_const * np.real( 1j* np.conj(p) @ (dy_rot_G_H @ m))
    Fy_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real( np.conj(m) @ (dy_G_H @ m))
    Fy_m2 = - 0.5 * mu*mu0_const * omega * np.real(1j* np.conj(m) @ (dy_rot_G_E @ p) )

    F_y = Fy_e1 + Fy_e2 + Fy_m1 + Fy_m2 + F_crest[1]
    
    Fx_e1 = 0.5*mu*k**2/eps0_const * np.real(np.conj(p) @ (dx_G_E @ p))
    Fx_e2 = 0.5*omega*mu*mu0_const * np.real(1j*np.conj(p) @ (dx_rot_G_H @ m))
    Fx_e0 = -0.5*np.imag(np.conj(p)@ E0)*kx
    Fx_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real(np.conj(m)@ (dx_G_H @ m))
    Fx_m2 = -0.5*mu*mu0_const*omega*np.real(1j*np.conj(m) @ (dx_rot_G_E @ p))
    Fx_m0 = -0.5*mu*mu0_const*np.imag(np.conj(m)@ H0)*kx
    
    F_x = Fx_e1 + Fx_e2 + Fx_e0 + Fx_m0 + Fx_m1 + Fx_m2 + F_crest[0]
    
    
    
    Fz_e1 = 0.5*mu*k**2/eps0_const * np.real(np.conj(p) @ (dz_G_E @ p))
    Fz_e2 = 0.5*omega*mu*mu0_const * np.real(1j*np.conj(p) @ (dz_rot_G_H @ m))
    Fz_e0 = 0.5*np.real(np.conj(p) @ dz_dE)
    Fz_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real(np.conj(m)@ (dz_G_H @ m))
    Fz_m2 = -0.5*mu*mu0_const*omega*np.real(1j*np.conj(m) @ (dz_rot_G_E @ p))
    Fz_m0 = 0.5*np.real(np.conj(m) @ dz_dH)*mu*mu0_const
    
    F_z = Fz_e1 + Fz_e2 + Fz_e0 + Fz_m0 + Fz_m1 + Fz_m2 + F_crest[2]
    
    if full_output==False:
        return F_x, F_y, F_z
    else:
        Fx = [F_x, Fx_e0, Fx_e1, Fx_e2, Fx_m0, Fx_m1, Fx_m2, F_crest[0]]
        Fy = [F_y, 0, Fy_e1, Fy_e2, 0, Fy_m1, Fy_m2, F_crest[1]]
        Fz = [F_z, Fz_e0, Fz_e1, Fz_e2, Fz_m0, Fz_m1, Fz_m2, F_crest[2]]
        return np.array([Fx, Fy, Fz])
