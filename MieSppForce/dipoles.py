import numpy as np
from MieSppForce import green_func
from MieSppForce import frenel
from scipy.special import spherical_jn, spherical_yn
from cmath import sqrt
from functools import lru_cache

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7

@lru_cache(maxsize=None)
def cached_green_functions(wl, z0, eps_Au, stop):
    G_ref_E, G_ref_H = green_func.green_ref_00(wl, z0, eps_Au, stop)
    rot_G_ref_E, rot_G_ref_H = green_func.rot_green_ref_00(
        wl, z0, eps_Au, stop)
    return (G_ref_E, G_ref_H, rot_G_ref_E, rot_G_ref_H)



def spherical_hn(n, x, derivative=False):
    if derivative == True:
        return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    return spherical_jn(n, x) + 1j * spherical_yn(n, x)


def alpha_v2(wl, R, eps_Si):
    if type(eps_Si) != int:
        eps3 = eps_Si(wl)
    else:
        eps3 = eps_Si
    n3 = np.sqrt(eps3+0j)
    x0 = R*2*np.pi/wl
    x1 = R*2*np.pi/wl*n3
    j1_x0 = spherical_jn(1, x0)
    j1_x1 = spherical_jn(1, x1)
    h1_x0 = spherical_hn(1, x0)
    h1_x1 = spherical_hn(1, x1)
    j1_x0_d = spherical_jn(1, x0, derivative=True)
    j1_x1_d = spherical_jn(1, x1, derivative=True)
    h1_x0_d = spherical_hn(1, x0, derivative=True)
    h1_x1_d = spherical_hn(1, x1, derivative=True)

    dx0dj1x0 = j1_x0 + x0 * j1_x0_d
    dx1dj1x1 = j1_x1 + x1 * j1_x1_d
    dx0dh1x0 = h1_x0 + x0 * h1_x0_d
    dx1dh1x1 = h1_x1 + x1 * h1_x1_d

    a1 = (n3**2 * dx0dj1x0 * j1_x1 - dx1dj1x1 * j1_x0) / \
        (n3**2 * dx0dh1x0 * j1_x1 - dx1dj1x1 * h1_x0)
    b1 = (dx0dj1x0 * j1_x1 - dx1dj1x1 * j1_x0) / \
        (dx0dh1x0 * j1_x1 - dx1dj1x1 * h1_x0)

    alpha_e = 6*1j*np.pi * a1 / (2*np.pi/wl/1e-9)**3
    alpha_m = 6*1j*np.pi * b1 / (2*np.pi/wl/1e-9)**3

    return alpha_e, alpha_m


def get_alpha(Rnm, eps_interp_particle, wl):
    R = Rnm*1e-9
    k = 2*np.pi/wl/1e-9
    eps_p = eps_interp_particle(wl)

    alpha_e_0 = 4*np.pi*eps0_const*R**3 * (eps_p - 1)/(eps_p + 2)
    alpha_e = alpha_e_0/(1 - 1j * k**3 * alpha_e_0 / (6*np.pi*eps0_const))

    alpha_m_0 = 4*np.pi * mu0_const / (k**3) * (eps_p - 1) * (k*R)**5/30
    alpha_m = alpha_m_0/(1 - 1j * k**3 * alpha_m_0 /
                         (6*np.pi*mu0_const))/mu0_const
    return alpha_e, alpha_m


def initial_field(wl, alpha, amplitude, eps_interp, point, phase, a_angle):
    rp, rs = frenel.reflection_coeff_v2(wl, eps_interp, alpha)
    xnm, ynm, znm = point
    x = xnm*1e-9
    z = znm*1e-9
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    exp = np.exp(1j*k*np.sin(alpha)*x-1j*k*np.cos(alpha)*z)
    exp_r = np.exp(1j*k*np.sin(alpha)*x+1j*k*np.cos(alpha)*z)

    Ex_tm = amplitude * np.cos(alpha) * (exp - exp_r * rp)
    Ez_tm = amplitude * np.sin(alpha) * (exp + exp_r * rp)
    Hy_tm = -k * amplitude * exp / omega / mu0_const - \
        rp * k * amplitude * exp_r / omega / mu0_const
    Hx_te = k*amplitude*np.cos(alpha) * exp / omega/mu0_const - \
        rs * k*amplitude*np.cos(alpha) * exp_r / omega/mu0_const
    Hz_te = k*amplitude*np.sin(alpha) * exp / omega/mu0_const + \
        rs * k*amplitude*np.sin(alpha) * exp_r / omega/mu0_const
    Ey_te = amplitude * exp + rs * amplitude * exp_r
    A = np.sin(a_angle)
    B = np.cos(a_angle)
    Phase = np.exp(1j*phase)
    E0 = np.array([[Ex_tm*B*Phase],
                   [Ey_te*A],
                   [Ez_tm*B*Phase]])
    H0 = np.array([[Hx_te*A],
                   [Hy_tm*B*Phase],
                   [Hz_te*A]])

    return E0, H0


# ТУТ ОШИБКА ГДЕ-ТО - ИЛИ НЕТ
# def calc_dipoles(wl, eps_interp, stop, R, eps_interp_particle, point, alpla, amplitude=1, phase=0, a=0):
#     x0, y0, z0 = point
#     mu = 1
#     eps = 1
#     k = 2*np.pi/wl/1e-9
#     omega = 2*np.pi*c_const/wl/1e-9
#     E0, H0 = initial_field(wl, alpla, amplitude, eps_interp, point, phase, a)
#     G_ref_E, G_ref_H = green_func.green_ref_00(wl, z0, eps_interp, stop)
#     rot_G_ref_E, rot_G_ref_H = green_func.rot_green_ref_00(
#         wl, z0, eps_interp, stop)
#     alpha_e, alpha_m = get_alpha(R, eps_interp_particle, wl)
#     I = np.eye(3, dtype=np.complex128)
#     E_eff = np.linalg.inv(I - mu*k**2/eps0_const * G_ref_E * alpha_e - omega**2*mu*mu0_const * rot_G_ref_H * alpha_m
#                           @ np.linalg.inv(I - eps * mu * k**2 * G_ref_H * alpha_m) @ rot_G_ref_E * alpha_e) \
#         @ (E0+1j*omega*mu*mu0_const*rot_G_ref_H*alpha_m @ np.linalg.inv(I - eps*mu*k**2*G_ref_H*alpha_m)@H0)

#     H_eff = np.linalg.inv(I - eps*mu*k**2 * G_ref_H * alpha_m - omega**2*mu*mu0_const*rot_G_ref_E*alpha_e
#                           @ np.linalg.inv(I - mu*k**2/eps0_const * G_ref_E * alpha_e)
#                           @ rot_G_ref_H * alpha_m) @ (H0 - 1j*omega*rot_G_ref_E * alpha_e @ np.linalg.inv(I - mu*k**2/eps0_const * G_ref_E * alpha_e)@E0)

#     p = alpha_e * E_eff
#     m = alpha_m * H_eff
#     return p, m


def calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, alpla, amplitude, phase, a_angle, stop):
    mu = 1
    eps = 1
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    x0, y0, z0 = point
    alpha_e, alpha_m = alpha_v2(wl, R, eps_Si)
    
    #alpha_e, alpha_m = get_alpha(R,eps_Si, wl)
    
    (G_ref_E, G_ref_H, rot_G_ref_E, rot_G_ref_H) = cached_green_functions(
        wl, z0, eps_Au, stop)

    E0, H0 = initial_field(wl, alpla, amplitude, eps_Au, point, phase, a_angle)

    G_ee = mu*k**2/eps0_const * G_ref_E
    G_em = 1j*omega*mu*mu0_const * rot_G_ref_H
    G_me = -1j*omega*rot_G_ref_E
    G_mm = eps*mu*k**2*G_ref_H

    I = np.eye(3, dtype=np.complex128)

    A = I - eps0_const * alpha_e * G_ee - eps0_const * alpha_e * \
        G_em @ np.linalg.inv(I - alpha_m * G_mm) * alpha_m @ G_me
    B = I - alpha_m * G_mm - alpha_m * \
        G_me @ np.linalg.inv(I - eps0_const * alpha_e *
                             G_ee) * eps0_const * alpha_e @ G_em

    Am1 = np.linalg.inv(A)
    Bm1 = np.linalg.inv(B)

    alpha_ee = Am1 * alpha_e
    alpha_mm = Bm1 * alpha_m

    alpha_em = Am1 * eps0_const * \
        alpha_e @ G_em @ np.linalg.inv(I - alpha_m * G_mm) * alpha_m
    alpha_me = Bm1 * \
        alpha_m @ G_me @ np.linalg.inv(I - eps0_const *
                                       alpha_e * G_ee) * eps0_const * alpha_e

    p = eps0_const * alpha_ee @ E0 + alpha_em @ H0
    m = alpha_mm @ H0 + alpha_me @ E0

    return p, m
