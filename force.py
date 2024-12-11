import numpy as np
import scipy as sc
import green_func
import frenel
import dipoles

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


# def F(wl, eps_Au, point,R, eps_si, alpha, amplitude, phase,a,stop):
#     mu=1
#     eps=1
#     k = 2*np.pi/wl/1e-9
#     omega = 2*np.pi*c_const/wl/1e-9
#     x0,y0,z0=point
#     dip = dipoles.calc_dipoles_v2(wl,eps_Au, point,R,eps_si, alpha, amplitude, phase,a,stop)
#     px,py,pz = dip[0]
#     mx,my,mz = dip[1]
#     E0, H0 = dipoles.initial_field(wl, alpha, amplitude, eps_Au, point, phase, a)
    
    
#     dxdG_E_xz, dxdG_H_xz = green_func.dx_green_E_H_xz(wl, z0, eps_Au, stop)
#     dx_rot_G_H_yz_zy = green_func.dx_rot_green_H_yz_zy(wl, z0, eps_Au, stop)
#     dx_rot_G_H_yz = dx_rot_G_H_yz_zy[1,2]
#     dx_rot_G_H_zy = dx_rot_G_H_yz_zy[2,1]
    
#     Fx_e1 = mu*k**2/eps0_const*np.imag(np.conj(px)*pz)*np.imag(dxdG_E_xz)
#     Fx_m1 = mu**2*mu0_const*eps*k**2*np.imag(np.conj(mx)*mz)*np.imag(dxdG_H_xz)
#     Fx_crest = - k**4/(12*np.pi*c_const*eps0_const)*np.real(py*np.conj(mz)-pz*np.conj(my))
#     Fx_2 = omega*mu*mu0_const*(np.real(np.conj(py)*mz)*np.imag(dx_rot_G_H_yz)+np.real(np.conj(pz)*my)*np.imag(dx_rot_G_H_zy))
#     Fx_e0 = 0.5*np.imag(np.conj(px)*E0[0]+np.conj(py)*E0[1]+np.conj(pz)*E0[2])*k*np.sin(alpha)
#     Fx_m0 = 0.5*np.imag(np.conj(mx)*H0[0]+np.conj(my)*H0[1]+np.conj(mz)*H0[2])*k*np.sin(alpha)*mu*mu0_const
    
#     F_x = Fx_e1+Fx_m1+Fx_crest+Fx_2+Fx_e0+Fx_m0
    
#     dydG_E_yz, dydG_H_yz = green_func.dy_green_E_H_yz(wl, z0, eps_Au, stop)    
#     dy_rot_G_H_zx_xz = green_func.dy_rot_green_H_zx_xz(wl, z0, eps_Au, stop)
#     dy_rot_G_H_zx = dy_rot_G_H_zx_xz[2,0]
#     dy_rot_G_H_xz = dy_rot_G_H_zx_xz[0,2]
    
#     Fy_e1 = mu*k**2/eps0_const*np.imag(np.conj(py)*pz)*np.imag(dydG_E_yz)
#     Fy_m1 = mu**2*mu0_const*eps*k**2*np.imag(np.conj(my)*mz)*np.imag(dydG_H_yz)
#     Fy_crest = - k**4/(12*np.pi*c_const*eps0_const)*np.real(pz*np.conj(mx)-px*np.conj(mz))
#     Fy_2 = omega*mu*mu0_const*(np.imag( dy_rot_G_H_zx)*np.real(np.conj(pz)*mx)+ np.imag(dy_rot_G_H_xz)*np.real(np.conj(px)*mz))
#     F_y = Fy_e1+Fy_m1+Fy_crest+Fy_2

#     return F_x, F_y


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
    
    dx_G_E, dx_G_H = green_func.dx_green_E_H(wl, z0, eps_Au, stop)
    dx_rot_G_E, dx_rot_G_H = green_func.dx_rot_green_E_H(wl, z0, eps_Au, stop)
    dy_G_E, dy_G_H = green_func.dy_green_E_H(wl, z0, eps_Au, stop)
    dy_rot_G_E, dy_rot_G_H = green_func.dy_rot_green_E_H(wl, z0, eps_Au, stop)
    
    F_crest = - k**4/(12*np.pi*c_const*eps0_const) * np.real( np.cross(p, np.conj(m)))
    
    Fy_e1 = 0.5*mu*k**2/eps0_const * np.real( np.conj(p) @ (dy_G_E @ p))
    Fy_e2 = 0.5*omega*mu*mu0_const * np.real( 1j* np.conj(p) @ (dy_rot_G_H @ m))
    Fy_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real( np.conj(m) @ (dy_G_H @ m))
    Fy_m2 = - 0.5 * mu*mu0_const * omega * np.real(1j* np.conj(m) @ (dy_rot_G_E @ p) )

    F_y = Fy_e1 + Fy_e2 + Fy_m1 + Fy_m2 + F_crest[1]
    
    Fx_e1 = 0.5*mu*k**2/eps0_const * np.real(np.conj(p) @ (dx_G_E @ p))
    Fx_e2 = 0.5*omega*mu*mu0_const * np.real(1j*np.conj(p) @ (dx_rot_G_H @ m))
    Fx_e0 = 0.5*np.imag(np.conj(p)@E0)*kx
    Fx_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real(np.conj(m)@ (dx_G_H @ m))
    Fx_m2 = -0.5*mu*mu0_const*omega*np.real(1j*np.conj(m) @ (dx_rot_G_E @ p))
    Fx_m0 = 0.5*mu*mu0_const*np.imag(np.conj(m)@ H0)*kx
    F_x = Fx_e1 + Fx_e2 + Fx_e0 + Fx_m0 + Fx_m1 + Fx_m2 + F_crest[0]
    
    return F_x, F_y

