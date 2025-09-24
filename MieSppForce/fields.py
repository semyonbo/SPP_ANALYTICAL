from MieSppForce import green_func, dipoles, frenel, green_func_v2
import numpy as np

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7


def get_field(wl, eps_interp, alpha, phase, a_angle, stop, eps_particle, R,   r, phi, z, z0, field_type = None, amplitude=1 ):
    
    assert z>= 0, "z should be >=0"
    assert z0>0, "z0 should be >0"
    
    k = 2*np.pi/wl*1e9
    omega = k*c_const
    
    GEres =np.zeros((3,3), dtype=complex)
    rotGHres = np.zeros_like(GEres)
    GHres = np.zeros_like(GEres)
    rotGEres = np.zeros_like(GEres)
    
    p,m = dipoles.calc_dipoles_v2(wl, eps_interp, [0,0,z0], R, eps_particle, alpha, amplitude, phase, a_angle, stop)

    G0, rotG0 = green_func_v2.G0(wl, z0, r, phi, z)
    
    GErp, rotGHrs, GHrs, rotGErp = green_func_v2.getG(wl, eps_interp, z+z0, r, phi, 'p')
    
    GErs, rotGHrp, GHrp, rotGErs = green_func_v2.getG(wl, eps_interp, z+z0, r, phi, 's')
        
    if field_type == 'spp':
        GEres, rotGHres, GHres, rotGEres = GErp, rotGHrp, GHrp, rotGErp
    elif field_type == 'sc':
        GEres, rotGHres, GHres, rotGEres = GErs, rotGHrs, GHrs, rotGErs
    elif field_type =='air':
        GEres, rotGHres, GHres, rotGEres = G0, rotG0, G0, rotG0
    elif field_type == 'reg':
        GEres, rotGHres, GHres, rotGEres = GErs+G0, rotGHrs+rotG0, G0+GHrs, rotG0+rotGErs
    else:
        GEres, rotGHres, GHres, rotGEres = GErs+GErp+G0, rotGHrs+rotGHrp+rotG0, GHrs+GHrp+G0, rotGErs+rotGErp+rotG0

        


    E =  k**2/eps0_const * GEres @ p + 1j*omega*mu0_const* rotGHres @m
    H =  k**2 * GHres  @ m - 1j*omega*rotGEres @ p

    
    return E[:,0],H[:,0]

def get_H_spp(wl, eps_interp, alpha, phase, a_angle, stop, eps_particle, R,   r, phi, z, z0 ):
    k = 2*np.pi/wl*1e9
    omega = k*c_const
    amplitude = 1  
    GHp = green_func.getGHp(wl, eps_interp, z0, r, phi, z, stop)
    rotGEp = green_func.get_rotGEp(wl, eps_interp, z0, r, phi, z, stop)
    p,m = dipoles.calc_dipoles_v2(wl, eps_interp, [0,0,z0], R, eps_particle, alpha, amplitude, phase, a_angle, stop)

    Hspp =  k**2 * GHp @ m - 1j*omega*rotGEp @ p
    
    return Hspp[:,0]


