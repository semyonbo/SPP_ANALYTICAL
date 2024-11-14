import numpy as np
import green_func
import frenel


c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


def get_alpha(R, eps_interp_particle, wl):
    k = 2*np.pi/wl/1e-9
    eps_p = eps_interp_particle(wl)

    alpha_e_0 = 4*np.pi*eps0_const*R**3 * (eps_p - 1)/(eps_p + 2)
    alpha_e = alpha_e_0/(1 - 1j * k**3 * alpha_e_0 / (6*np.pi*eps0_const))

    alpha_m_0 = 4*np.pi * mu0_const / (k**3) * (eps_p - 1) * (k*R)**5/30
    alpha_m = alpha_m_0/(1 - 1j * k**3 * alpha_m_0 /
                         (6*np.pi*mu0_const))/mu0_const

    return alpha_e, alpha_m


def initial_field(wl, alpha, amplitude, eps_interp, point, phase=0, a=0):
    rp, rs = frenel.reflection_coeff_v2(wl, eps_interp, alpha)
    x, y, z = point
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
    A = np.sqrt(a)
    B = np.sqrt(1-a)
    Phase = np.exp(1j*phase)
    E0 = np.array([[Ex_tm*B*Phase],
                   [Ey_te*A],
                   [Ez_tm*B*Phase]])
    H0 = np.array([[Hx_te*A],
                   [Hy_tm*B*Phase],
                   [Hz_te*A]])

    return E0, H0


def calc_dipoles(wl, eps_interp, stop, R, eps_interp_particle, point, alpla, amplitude=1, phase=0, a=0):
    x0, y0, z0 = point
    mu = 1
    eps = 1
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    E0, H0 = initial_field(wl, alpla, amplitude, eps_interp, point, phase, a)
    G_ref_E, G_ref_H = green_func.green_ref_00(wl, z0, eps_interp, stop)
    rot_G_ref_E, rot_G_ref_H = green_func.rot_green_ref_00(wl, z0, eps_interp, stop)
    alpha_e, alpha_m = get_alpha(R, eps_interp_particle, wl)
    I = np.eye(3, dtype=np.complex128)
    E_eff = np.linalg.inv(I - mu*k**2/eps0_const * G_ref_E * alpha_e - omega**2*mu*mu0_const * rot_G_ref_H * alpha_m @ np.linalg.inv(I - eps * mu * k**2 * G_ref_H * alpha_m) @ rot_G_ref_E * alpha_e)\
        @ (E0+1j*omega*mu*mu0_const*rot_G_ref_H*alpha_m @ np.linalg.inv(I - eps*mu*k**2*G_ref_H*alpha_m)@H0)
    H_eff = np.linalg.inv(I - eps*mu*k**2 * G_ref_H * alpha_m - omega**2*mu*mu0_const*rot_G_ref_E*alpha_e @ np.linalg.inv(I - mu*k**2/eps0_const * G_ref_E * alpha_e) @ rot_G_ref_H * alpha_m)\
        @ (H0 - 1j*omega*rot_G_ref_E * alpha_e @ np.linalg.inv(I - mu*k**2/eps0_const * G_ref_E * alpha_e)@E0)

    p = alpha_e * E_eff
    m = alpha_m * H_eff
    return p, m
