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
    
    
def field_dx(field, wl, alpha, amplitude, eps_Au, point, phase, a_angle, h=1):
    dp = np.array([h / 2, 0, 0])
    
    E_plus, H_plus = field(wl, alpha, amplitude, eps_Au, point + dp, phase, a_angle)
    E_minus, H_minus = field(wl, alpha, amplitude, eps_Au, point - dp, phase, a_angle)

    dE_dx = (E_plus[:,0] - E_minus[:,0]) / h * 1e09
    dH_dx = (H_plus[:,0] - H_minus[:,0]) / h * 1e09

    return dE_dx, dH_dx

def field_dz(field, wl, alpha, amplitude, eps_Au, point, phase, a_angle, h=1e-3):
    dp = np.array([0, 0, h / 2])
    
    E_plus, H_plus = field(wl, alpha, amplitude, eps_Au, point + dp, phase, a_angle)
    E_minus, H_minus = field(wl, alpha, amplitude, eps_Au, point - dp, phase, a_angle)

    dE_dz = (E_plus[:,0] - E_minus[:,0]) / h * 1e09
    dH_dz = (H_plus[:,0] - H_minus[:,0]) / h * 1e09

    return dE_dz, dH_dz
    


def F(wl, eps_Au, point,R, eps_si, alpha, amplitude, phase, a_angle ,stop, full_output=False, initial_field_type=None):
    mu=1
    eps=1
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    _,_,z0=point
    
    dip = dipoles.calc_dipoles_v2(wl,eps_Au, point,R,eps_si, alpha, amplitude, phase, a_angle, initial_field_type=initial_field_type)
        
    p = dip[0][:,0]
    m = dip[1][:,0]
    
    if initial_field_type == 'two_beam':
        field0 = dipoles.field_two_beam_setup
    elif initial_field_type == 'plane_wave':
        field0 = dipoles.initial_field
    elif initial_field_type == 'custom':
        field0 = dipoles.custom_field
    else:
        raise ValueError("Invalid initial_field_type. Choose from 'plane_wave', 'two_beam', or 'custom'.")

    dE_dx, dH_dx = field_dx(field0, wl, alpha, amplitude, eps_Au, point, phase, a_angle)
    dE_dz, dH_dz = field_dz(field0, wl, alpha, amplitude, eps_Au, point, phase, a_angle)


    (dx_G_E, dx_G_H, dx_rot_G_E, dx_rot_G_H,
     dy_G_E, dy_G_H, dy_rot_G_E, dy_rot_G_H,
     dz_G_E, dz_G_H, dz_rot_G_E, dz_rot_G_H) = cached_green_functions(wl, z0, eps_Au, stop)
    
    F_crest = - k**4/(12*np.pi*c_const*eps0_const) * np.real( np.cross(p, np.conj(m)))
    
    Fy_e1 = 0.5*mu*k**2/eps0_const * np.real( np.conj(p) @ (dy_G_E @ p))
    Fy_e2 = 0.5*omega*mu*mu0_const * np.real( 1j* np.conj(p) @ (dy_rot_G_H @ m))
    Fy_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real( np.conj(m) @ (dy_G_H @ m))
    Fy_m2 = - 0.5 * mu*mu0_const * omega * np.real(1j* np.conj(m) @ (dy_rot_G_E @ p) )
    F_y = Fy_e1 + Fy_e2 + Fy_m1 + Fy_m2 + F_crest[1]
    
    Fx_e0 = 0.5*np.real(np.conj(p) @ dE_dx)
    Fx_e1 = 0.5*mu*k**2/eps0_const * np.real(np.conj(p) @ (dx_G_E @ p))
    Fx_e2 = 0.5*omega*mu*mu0_const * np.real(1j*np.conj(p) @ (dx_rot_G_H @ m))
    Fx_m0 = 0.5*np.real(np.conj(m) @ dH_dx)*mu*mu0_const    
    Fx_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real(np.conj(m)@ (dx_G_H @ m))
    Fx_m2 = -0.5*mu*mu0_const*omega*np.real(1j*np.conj(m) @ (dx_rot_G_E @ p))
    F_x = Fx_e1 + Fx_e2 + Fx_e0 + Fx_m0 + Fx_m1 + Fx_m2 + F_crest[0]
    
    
    
    Fz_e1 = 0.5*mu*k**2/eps0_const * np.real(np.conj(p) @ (dz_G_E @ p))
    Fz_e2 = 0.5*omega*mu*mu0_const * np.real(1j*np.conj(p) @ (dz_rot_G_H @ m))
    Fz_e0 = 0.5*np.real(np.conj(p) @ dE_dz)
    Fz_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real(np.conj(m)@ (dz_G_H @ m))
    Fz_m2 = -0.5*mu*mu0_const*omega*np.real(1j*np.conj(m) @ (dz_rot_G_E @ p))
    Fz_m0 = 0.5*np.real(np.conj(m) @ dH_dz)*mu*mu0_const
    F_z = Fz_e1 + Fz_e2 + Fz_e0 + Fz_m0 + Fz_m1 + Fz_m2 + F_crest[2]
    
    if full_output==False:
        return F_x, F_y, F_z
    else:
        Fx = [F_x, Fx_e0, Fx_e1, Fx_e2, Fx_m0, Fx_m1, Fx_m2, F_crest[0]]
        Fy = [F_y, 0, Fy_e1, Fy_e2, 0, Fy_m1, Fy_m2, F_crest[1]]
        Fz = [F_z, Fz_e0, Fz_e1, Fz_e2, Fz_m0, Fz_m1, Fz_m2, F_crest[2]]
        return np.array([Fx, Fy, Fz])
