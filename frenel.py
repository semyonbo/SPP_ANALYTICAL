import numpy as np
from scipy.interpolate import interp1d
import pickle
import os
import functools

def reflection_coeff(wl, eps_interp, kr):
    """_summary_

    Args:
        wl (_type_): wavelenght
        eps_interp (_type_): substrate material interpolation
        kr (_type_): k_r value

    Returns:
        _type_: _description_
    """
    eps1=1
    eps2 = eps_interp(wl)
    #eps2=-1.5+1j*0.5
    k1 = np.sqrt(eps1+0j)
    k2 = np.sqrt(eps2+0j)
    kz1 = np.sqrt(k1**2 - kr**2+0j)
    kz2 = np.sqrt(k2**2 - kr**2+0j)
    rpE = (eps2*kz1 - eps1*kz2)/(eps2*kz1 + eps1*kz2)
    rpH = (kz1-kz2)/(kz1+kz2)
    rsE = rpH
    rsH = rpE 
    return rpE, rpH, rsE, rsH


def reflection_coeff_v2(wl, eps_interp, angle):
    """ Calculate refraction coefficents

    Args:
        wl ( int): wavelenght
        mat (_type_): _description_
        angle ( float): _description_

    Returns:
        _type_: _description_
    """
    eps1=1
    eps2 = eps_interp(wl)
    #eps2=-1.5+1j*0.5
    k1 = 2*np.pi/wl/1e-9 * np.sqrt(eps1 +0j)
    k2 = k1 * np.sqrt(eps2 + 0j)
    kx = k1*np.sin(angle)
    kz1 = np.sqrt(k1**2 - kx**2+0j)
    kz2 = np.sqrt(k2**2 - kx**2+0j)
    rp = (eps2*kz1 - eps1*kz2)/(eps2*kz1 + eps1*kz2)
    rs = (kz1-kz2)/(kz1+kz2)
    return rp,  rs


def get_interpolate(mat):
    """_summary_

    Args:
        mat (str): Material name

    Returns:
        numpy.complex128: epsilon
    """
    if mat == 'Au':
        data = np.loadtxt('nkAu.csv', delimiter=',', skiprows=1)
    elif mat == 'Si':
        data = np.loadtxt('nkSi.csv', delimiter=',', skiprows=1)

    n_interp = interp1d(data[:,0], data[:,1])
    k_interp = interp1d(data[:,0], data[:,2])
    
    return lambda wl: (n_interp(wl*1e-3) + 1j* k_interp(wl*1e-3))**2