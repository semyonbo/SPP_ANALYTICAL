from functools import lru_cache
import numpy as np
from scipy import integrate
from scipy.special import jn, j0, j1
from numpy import sqrt, exp, sin, cos, tan, pi
from MieSppForce import frenel

_integrals_cache = {}


def _make_cache_key(wl, h, r, forH, eps_val, field_type):
    """Создание уникального ключа для кэша."""
    eps_key = (round(eps_val.real, 10), round(eps_val.imag, 10)) if isinstance(eps_val, complex) else round(eps_val, 10)
    return (round(wl, 6), round(h, 6), round(r, 6), forH, eps_key, field_type)


def get_integrals_cache():
    """Получить текущий кэш интегралов."""
    return _integrals_cache


def set_integrals_cache(cache_dict):
    """Установить кэш интегралов (для передачи между процессами)."""
    global _integrals_cache
    _integrals_cache = cache_dict


def clear_integrals_cache():
    """Очистить кэш интегралов."""
    global _integrals_cache
    _integrals_cache = {}


def integrator(f, field_type=None):
    """Вычисление интеграла с подстановкой."""
    def f_subst_reg(t):
        kr = tan(t)
        return f(kr) * (1 / cos(t)**2)
    
    def f_subst_spp(t):
        kr = 1/t
        return f(kr) / (t**2)

    if field_type == 'spp':
        start, end = 0, 1
        points = [0]
        f_subst = f_subst_spp
    elif field_type == 'reg':
        start, end = 0, pi/4
        points = [pi/4]
        f_subst = f_subst_reg
    else:
        raise ValueError(f"Unknown field_type: {field_type}")
        
    I, err = integrate.quad(lambda t: f_subst(t), start, end, points=points, 
                           complex_func=True, limit=8000, epsrel=1e-6)
    return I


def precompute_integrals(wl, h, r, forH, eps_val, field_type=None):
    """
    Вычисление интегралов для функций Грина с кэшированием.
    
    Интегралы зависят только от:
    - wl: длина волны
    - h: высота (z + z0)  
    - r: радиальная координата
    - forH: True для H-функций, False для E-функций
    - eps_val: значение диэлектрической проницаемости (число!)
    - field_type: 'spp' или 'reg'
    
    НЕ зависят от параметров поляризации (psi, chi, a_angle, phase).
    """
    # Проверка кэша
    cache_key = _make_cache_key(wl, h, r, forH, eps_val, field_type)
    if cache_key in _integrals_cache:
        return _integrals_cache[cache_key]
    k = 2*np.pi/wl

    def rp(kr): return frenel.reflection_coeff(wl, lambda _: eps_val, kr)[2 if forH else 0]
    def rs(kr): return frenel.reflection_coeff(wl, lambda _: eps_val, kr)[0 if forH else 2]
    
    def kz(kr): return k*sqrt(1 - kr**2+0j)
    def exp_fac(kr): return exp(1j*kz(kr)*h)
    
    funcs = [
        lambda kr: rp(kr) * kr*k*kz(kr)*exp_fac(kr)*j0(kr*k*r),   # 0
        lambda kr: rp(kr) * kr*k*kz(kr)*exp_fac(kr)*jn(2, kr*k*r),# 1
        lambda kr: rp(kr)*(kr*k)**2 * exp_fac(kr)*j1(kr*k*r),     # 2
        lambda kr: rp(kr)*(kr*k)**3/kz(kr)*exp_fac(kr)*j0(kr*k*r),# 3
        lambda kr: rs(kr)*kr*k/kz(kr)*exp_fac(kr)*jn(2, kr*k*r),  # 4
        lambda kr: rs(kr)*kr*k/kz(kr)*exp_fac(kr)*j0(kr*k*r),     # 5
        lambda kr: rp(kr)*kr*k*exp_fac(kr)*jn(2, kr*k*r),         # 6
        lambda kr: rp(kr)*kr*k*exp_fac(kr)*j0(kr*k*r),            # 7
        lambda kr: (kr*k)**2/kz(kr)*rp(kr)*exp_fac(kr)*j1(kr*k*r),# 8
        lambda kr: rs(kr)*kr*k*exp_fac(kr)*jn(2, kr*k*r),         # 9
        lambda kr: rs(kr)*kr*k*exp_fac(kr)*j0(kr*k*r),            # 10
        lambda kr: rs(kr)*(kr*k)**2/kz(kr)*exp_fac(kr)*j1(kr*k*r) # 11
    ]

    res = []
    for idx, f in enumerate(funcs):
        res.append(integrator(f, field_type)*k)
    result = tuple(res)
    
    # Сохраняем в кэш
    _integrals_cache[cache_key] = result
    return result


def build_matrices(wl, phi, integrals, polarization=None):
    (int_rp_kr_kz_exp_j0, int_rp_kr_kz_exp_j2, int_rp_kr2_exp_j1,
     int_rp_kr3_kz_exp_j0, int_rs_kr_kz_exp_j2, int_rs_kr_kz_exp_j0,
     int_rp_kr_exp_j2, int_rp_kr_exp_j0, int_rp_kr2_kz_exp_j1,
     int_rs_kr_exp_j2, int_rs_kr_exp_j0, int_rs_kr2_kz_exp_j1) = integrals

    nm_to_m = 1e-9
    k = 2*np.pi/wl

    # --- GEp ---
    GEp = np.zeros((3,3), dtype=complex)
    GEp[0,0] = -1j/(8*pi*k**2*nm_to_m)*(int_rp_kr_kz_exp_j0 - int_rp_kr_kz_exp_j2*cos(2*phi))
    GEp[0,1] = sin(2*phi)*1j/(8*pi*k**2*nm_to_m)*int_rp_kr_kz_exp_j2
    GEp[0,2] = cos(phi)/(4*pi*k**2*nm_to_m)*int_rp_kr2_exp_j1
    GEp[1,1] = -1j/(8*pi*k**2*nm_to_m)*(int_rp_kr_kz_exp_j0+int_rp_kr_kz_exp_j2*cos(2*phi))
    GEp[1,2] = sin(phi)/(4*pi*k**2*nm_to_m)*int_rp_kr2_exp_j1
    GEp[2,2] = 1j/(4*pi*k**2*nm_to_m)*int_rp_kr3_kz_exp_j0
    GEp[1,0] = GEp[0,1]; GEp[2,0] = -GEp[0,2]; GEp[2,1] = -GEp[1,2]

    # --- GEs ---
    GEs = np.zeros((3,3), dtype=complex)
    GEs[0,0] = 1j/(8*pi*nm_to_m)*(int_rs_kr_kz_exp_j2*cos(2*phi)+int_rs_kr_kz_exp_j0)
    GEs[0,1] = 1j*sin(2*phi)/(8*pi*nm_to_m)*int_rs_kr_kz_exp_j2
    GEs[1,1] = 1j/(8*pi*nm_to_m)*(int_rs_kr_kz_exp_j0 - cos(2*phi)*int_rs_kr_kz_exp_j2)
    GEs[1,0] = GEs[0,1]

    # --- rotGHs ---
    rotGHs = np.zeros((3,3), dtype=complex)
    rotGHs[0,0] = sin(2*phi)/(8*pi)*int_rp_kr_exp_j2/nm_to_m**2
    rotGHs[0,1] = 1/(8*pi)*(int_rp_kr_exp_j0 - cos(2*phi)*int_rp_kr_exp_j2)/nm_to_m**2
    rotGHs[1,0] = -1/(8*pi)*(int_rp_kr_exp_j0 + cos(2*phi)*int_rp_kr_exp_j2)/nm_to_m**2
    rotGHs[1,1] = -rotGHs[0,0]
    rotGHs[2,0] = 1j*sin(phi)/(4*pi)*int_rp_kr2_kz_exp_j1/nm_to_m**2
    rotGHs[2,1] = -1j*cos(phi)/(4*pi)*int_rp_kr2_kz_exp_j1/nm_to_m**2

    # --- rotGHp ---
    rotGHp = np.zeros((3,3), dtype=complex)
    rotGHp[0,0] = sin(2*phi)/(8*pi)*int_rs_kr_exp_j2/nm_to_m**2
    rotGHp[0,1] = -1/(8*pi)*(int_rs_kr_exp_j0 + cos(2*phi)*int_rs_kr_exp_j2)/nm_to_m**2
    rotGHp[0,2] = -1j*sin(phi)/(4*pi)*int_rs_kr2_kz_exp_j1/nm_to_m**2
    rotGHp[1,0] = 1/(8*pi)*(int_rs_kr_exp_j0 - cos(2*phi)*int_rs_kr_exp_j2)/nm_to_m**2
    rotGHp[1,1] = -rotGHp[0,0]
    rotGHp[1,2] = 1j*cos(phi)/(4*pi)*int_rs_kr2_kz_exp_j1/nm_to_m**2
    

    if polarization == 'p':
        return GEp, rotGHp
    elif polarization == 's':
        return GEs, rotGHs
    else: 
        return GEp+GEs, rotGHp+rotGHs


def getG(wl, eps_interp, h, r, phi, field_type=None):
    
    eps_val = eps_interp(wl)  


    ints_E_evas = precompute_integrals(wl, h, r, False, eps_val, 'spp')
    ints_H_evas = precompute_integrals(wl, h, r, True, eps_val, 'spp')
    ints_E_reg =  precompute_integrals(wl, h, r, False, eps_val, 'reg')
    ints_H_reg =  precompute_integrals(wl, h, r, True, eps_val, 'reg') 
    
    
    GErp_reg, rotGHrs_reg = build_matrices(wl, phi, ints_E_reg, 'p')
    GErp_evas, rotGHrs_evas = build_matrices(wl, phi, ints_E_evas, 'p')
    
    GErs_reg, rotGHrp_reg = build_matrices(wl, phi, ints_E_reg, 's')
    GErs_evas, rotGHrp_evas = build_matrices(wl, phi, ints_E_evas, 's')
    
    
    
    GHrs_reg, rotGErp_reg = build_matrices(wl, phi, ints_H_reg, 'p')
    GHrs_evas, rotGErp_evas=build_matrices(wl, phi, ints_H_evas, 'p')
    
    GHrp_reg, rotGErs_reg = build_matrices(wl, phi, ints_H_reg, 's')
    GHrp_evas, rotGErs_evas =build_matrices(wl, phi, ints_H_evas, 's')
    
    
    if field_type == 'spp':
        return [GErp_evas, rotGHrp_evas, GHrp_evas, rotGErp_evas]
    elif field_type == 'reg':
        return [GErp_reg+GErs_reg+GErs_evas, rotGHrs_reg+rotGHrp_reg+rotGHrs_evas, GHrs_reg+GHrp_reg+GHrs_evas, rotGErp_reg+rotGErs_reg+rotGErs_evas]
    else :
        GE = GErp_reg + GErp_evas + GErs_reg+GErs_evas
        rotGH = rotGHrs_reg+rotGHrs_evas+rotGHrp_reg+rotGHrp_evas
        
        GH = GHrs_reg+GHrs_evas+GHrp_reg+GHrp_evas
        rotGE = rotGErp_reg+ rotGErp_evas+rotGErs_reg+rotGErs_evas
        return [GE, rotGH, GH, rotGE]




def G0(wl,z0, r, phi, z):
    nm_to_m = 1e-9
    I = np.eye(3, dtype=complex)
    k = 2*pi/wl
    
    x0 =0 
    y0 =0 
    x = r*cos(phi)
    y = r*sin(phi)
    
    r0 = np.array([[x0],
                  [y0],
                  [z0]])
    r = np.array([[x],
                    [y],
                [z]])
    
    R = r - r0
    
    if np.all(R==0):
        return np.zeros((3,3), dtype=complex), np.zeros((3,3), dtype=complex)
    
    Rabs = np.linalg.norm(R)
    
    RR = np.outer(R,R)
    
    exp_fac = exp(1j*k*Rabs)
    
    G0 = exp_fac/(4*pi*Rabs) * ((1+(1j*k*Rabs-1)/(k**2*Rabs**2)) * I + (3-3*1j*k*Rabs-k**2*Rabs**2)/(k**2*Rabs**2) * RR/Rabs**2)/nm_to_m
    
    Rx, Ry, Rz = R[:,0]
    RxI = np.array([
        [0,   -Rz,  Ry],
        [Rz,   0,  -Rx],
        [-Ry, Rx,   0]
    ], dtype=complex)
    
    rotG0 = k * exp_fac/(4*pi*Rabs) * RxI/Rabs * ( 1j - 1/(k*Rabs))/nm_to_m**2
    
    return G0, rotG0