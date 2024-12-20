import numpy as np
import scipy as sc
import green_func
import frenel
import dipoles

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


def F(wl, eps_Au, point,R, eps_si, alpha, amplitude, phase,a,stop):
    mu=1
    eps=1
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    x0,y0,z0=point
    dip = dipoles.calc_dipoles_v2(wl,eps_Au, point,R,eps_si, alpha, amplitude, phase,a,stop)
    p = dip[0]
    m = dip[1]
    p = p[:,0]
    m = m[:,0]
    E0, H0 = dipoles.initial_field(wl, alpha, amplitude, eps_Au, point, phase, a)
    E0 = E0[:,0]
    H0 = H0[:,0]
    kx = k*np.sin(alpha)
    rp, rs = frenel.reflection_coeff_v2(wl, eps_Au, alpha)
    zm0 = z0*1e-9
    A = np.sqrt(a)
    B = np.sqrt(1-a)
    Phase = np.exp(1j*phase)
    kz = k * np.cos(alpha)
    
    # dx_dEx = B * Phase *amplitude * np.cos(alpha) * ( np.exp(-1j*kz*zm0) - rp * np.exp(1j*kz*zm0))*1j*kx
    # dx_dEy = A * amplitude * (np.exp(-1j*kz*zm0) + rs * np.exp(1j*kz*zm0))*1j*kx
    # dx_dEz = B * Phase * amplitude * np.sin(alpha) * ( np.exp(-1j*kz*zm0) + rp *np.exp(1j*kz*zm0))*1j*kx
    
    # dx_dE = np.array([[dx_dEx],
    #                  [dx_dEy],
    #                  [dx_dEz]])
    
    # dx_dHx = k* A *amplitude*np.cos(alpha) / omega/mu0_const *(np.exp(-1j*kz*zm0) - rs * np.exp(1j*kz*zm0))*1j*kx
    # dx_dHy = -k * B * Phase * amplitude  / omega / mu0_const *(np.exp(-1j*kz*zm0) + rp * np.exp(1j*kz*zm0))*1j*kx
    # dx_dHz = A * k*amplitude*np.sin(alpha) / omega/mu0_const *(np.exp(-1j*kz*zm0) + rs * np.exp(1j*kz*zm0))*1j*kx
    
    # dx_dH = np.array([[dx_dHx],
    #                  [dx_dHy],
    #                  [dx_dHz]])
    
    # dx_dE=dx_dE[:,0]
    # dx_dH=dx_dH[:,0]
    
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
    
    
    dx_G_E, dx_G_H = green_func.dx_green_E_H(wl, z0, eps_Au, stop)
    dx_rot_G_E, dx_rot_G_H = green_func.dx_rot_green_E_H(wl, z0, eps_Au, stop)
    
    dy_G_E, dy_G_H = green_func.dy_green_E_H(wl, z0, eps_Au, stop)
    dy_rot_G_E, dy_rot_G_H = green_func.dy_rot_green_E_H(wl, z0, eps_Au, stop)
    
    dz_G_E, dz_G_H = green_func.dz_green_E_H(wl, z0, eps_Au, stop)
    dz_rot_G_E, dz_rot_G_H = green_func.dz_rot_green_E_H(wl, z0, eps_Au, stop)
    
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
    
    
    
    return Fx_e0, Fx_e1, Fx_e2, Fx_m0, Fx_m1, Fx_m2, F_crest[0]

