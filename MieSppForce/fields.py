from MieSppForce import green_func, dipoles, frenel, green_func_v2
import numpy as np

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


def get_field(wl, eps_interp, alpha, phase, a_angle, eps_particle, R,   r, phi, z, z0, field_type = None, amplitude=1, initial_field_type=None):
    
    assert z>= 0, "z should be >=0"
    assert z0>0, "z0 should be >0"
    
    k = 2*np.pi/wl*1e9
    omega = k*c_const
    
    GEres =np.zeros((3,3), dtype=complex)
    rotGHres = np.zeros_like(GEres)
    GHres = np.zeros_like(GEres)
    rotGEres = np.zeros_like(GEres)
    
    p,m = dipoles.calc_dipoles_v2(wl, eps_interp, [0,0,z0], R, eps_particle, alpha, amplitude, phase, a_angle, initial_field_type=initial_field_type)

    G0, rotG0 = green_func_v2.G0(wl, z0, r, phi, z)
    GE_spp, rotGH_spp, GH_spp, rotGE_spp = green_func_v2.getG(wl, eps_interp, z+z0, r, phi, 'spp')
    GE_reg, rotGH_reg, GH_reg, rotGE_reg = green_func_v2.getG(wl, eps_interp, z+z0, r, phi, 'reg')
    # GE, rotGH, GH, rotGE = green_func_v2.getG(wl, eps_interp, z+z0, r, phi)
    
    if field_type == 'spp':
        GEres, rotGHres, GHres, rotGEres = GE_spp, rotGH_spp, GH_spp, rotGE_spp
    elif field_type == 'sc':
        GEres, rotGHres, GHres, rotGEres = GE_reg, rotGH_reg, GH_reg, rotGE_reg
    elif field_type == 'air':
        GEres, rotGHres, GHres, rotGEres = G0, rotG0, G0, rotG0
    elif field_type == 'reg':
        GEres, rotGHres, GHres, rotGEres = GE_reg+G0, rotGH_reg+rotG0, GH_reg+G0, rotGE_reg+rotG0
    else:
        GEres, rotGHres, GHres, rotGEres = GE_spp+GE_reg+G0, rotGH_reg+rotGH_spp+rotG0, GH_reg+GH_spp+G0, rotGE_reg+rotGE_spp+rotG0


    E =  k**2/eps0_const * GEres @ p + 1j*omega*mu0_const* rotGHres @m
    H =  k**2 * GHres  @ m - 1j*omega*rotGEres @ p
    return E[:,0],H[:,0]

