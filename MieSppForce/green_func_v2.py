# from functools import lru_cache
# import hashlib, os, pickle
# import numpy as np
# from scipy import integrate
# from scipy.special import jn, j0, j1
# from numpy import sqrt, exp, sin, cos, tan, pi
# from MieSppForce import frenel
# from filelock import FileLock

# # CACHE_FILE = "integrator_cache.pkl"
# # CACHE_LOCK = FileLock(CACHE_FILE + ".lock")

# # _mem_cache = {}

# # def clear_cache(): 
# #     if os.path.exists(CACHE_FILE):  
# #         os.remove(CACHE_FILE)

        
# def integrator(f, key, field_type=None):
    
#     # if key in _mem_cache:
#     #     return _mem_cache[key]
    
#     # with CACHE_LOCK:
#     #     if os.path.exists(CACHE_FILE):
#     #         with open(CACHE_FILE, "rb") as fh:
#     #             cache = pickle.load(fh)
#     #         if key in cache:
#     #             _mem_cache[key] = cache[key]
#     #             return cache[key]

#     def f_subst_reg(t):
#         kr = tan(t)
#         return f(kr) * (1 / cos(t)**2)
    
#     def f_subst_spp(t):
#         kr = 1/t
#         return f(kr) /(t**2)

#     if field_type == 'spp':
#         start, end = 0+1e-12, 1
#         points=[0]
#         f_subst = f_subst_spp
#     elif field_type == 'reg':
#         start, end= 0, pi/4
#         points=[pi/4]
#         f_subst = f_subst_reg
#     else:
#         start, end = 0, pi/2
#         points=[pi/4]
#         f_subst = f_subst_reg
        
        
#     I, err = integrate.quad(lambda t: f_subst(t), start, end, points=points, limit=4000, complex_func=True, epsrel=1e-8)
    
#     # _mem_cache[key] = I
#     # with CACHE_LOCK:
#     #     cache = {}
#     #     if os.path.exists(CACHE_FILE):
#     #         with open(CACHE_FILE, "rb") as fh:
#     #             cache = pickle.load(fh)
#     #     cache[key] = I
#     #     with open(CACHE_FILE, "wb") as fh:
#     #         pickle.dump(cache, fh)

#     return I


# @lru_cache(maxsize=None)
# def precompute_integrals(wl, h, r, forH, eps_val, field_type=None):
#     k = 2*np.pi/wl

#     def rp(kr): return frenel.reflection_coeff(wl, lambda _: eps_val, kr)[2 if forH else 0]
#     def rs(kr): return frenel.reflection_coeff(wl, lambda _: eps_val, kr)[0 if forH else 2]
    
#     def kz(kr): return k*sqrt(1 - kr**2+0j)
#     def exp_fac(kr): return exp(1j*kz(kr)*h)
    
#     funcs = [
#         lambda kr: rp(kr) * kr*k*kz(kr)*exp_fac(kr)*j0(kr*k*r),   # 0
#         lambda kr: rp(kr) * kr*k*kz(kr)*exp_fac(kr)*jn(2, kr*k*r),# 1
#         lambda kr: rp(kr)*(kr*k)**2 * exp_fac(kr)*j1(kr*k*r),     # 2
#         lambda kr: rp(kr)*(kr*k)**3/kz(kr)*exp_fac(kr)*j0(kr*k*r),# 3
#         lambda kr: rs(kr)*kr*k/kz(kr)*exp_fac(kr)*jn(2, kr*k*r),  # 4
#         lambda kr: rs(kr)*kr*k/kz(kr)*exp_fac(kr)*j0(kr*k*r),     # 5
#         lambda kr: rp(kr)*kr*k*exp_fac(kr)*jn(2, kr*k*r),         # 6
#         lambda kr: rp(kr)*kr*k*exp_fac(kr)*j0(kr*k*r),            # 7
#         lambda kr: (kr*k)**2/kz(kr)*rp(kr)*exp_fac(kr)*j1(kr*k*r),# 8
#         lambda kr: rs(kr)*kr*k*exp_fac(kr)*jn(2, kr*k*r),         # 9
#         lambda kr: rs(kr)*kr*k*exp_fac(kr)*j0(kr*k*r),            # 10
#         lambda kr: rs(kr)*(kr*k)**2/kz(kr)*exp_fac(kr)*j1(kr*k*r) # 11
#     ]

#     res = []
#     for idx, f in enumerate(funcs):
#         key = hashlib.md5(f"{wl}_{eps_val}_{h}_{r}_{idx}_{forH}_{field_type}".encode()).hexdigest()
#         res.append(integrator(f, key, field_type)*k)
#     return tuple(res)


# def build_matrices(wl, phi, integrals, polarization=None):
#     (int_rp_kr_kz_exp_j0, int_rp_kr_kz_exp_j2, int_rp_kr2_exp_j1,
#      int_rp_kr3_kz_exp_j0, int_rs_kr_kz_exp_j2, int_rs_kr_kz_exp_j0,
#      int_rp_kr_exp_j2, int_rp_kr_exp_j0, int_rp_kr2_kz_exp_j1,
#      int_rs_kr_exp_j2, int_rs_kr_exp_j0, int_rs_kr2_kz_exp_j1) = integrals

#     nm_to_m = 1e-9
#     k = 2*np.pi/wl

#     # --- GEp ---
#     GEp = np.zeros((3,3), dtype=complex)
#     GEp[0,0] = -1j/(8*pi*k**2*nm_to_m)*(int_rp_kr_kz_exp_j0 - int_rp_kr_kz_exp_j2*cos(2*phi))
#     GEp[0,1] = sin(2*phi)*1j/(8*pi*k**2*nm_to_m)*int_rp_kr_kz_exp_j2
#     GEp[0,2] = cos(phi)/(4*pi*k**2*nm_to_m)*int_rp_kr2_exp_j1
#     GEp[1,1] = -1j/(8*pi*k**2*nm_to_m)*(int_rp_kr_kz_exp_j0+int_rp_kr_kz_exp_j2*cos(2*phi))
#     GEp[1,2] = sin(phi)/(4*pi*k**2*nm_to_m)*int_rp_kr2_exp_j1
#     GEp[2,2] = 1j/(4*pi*k**2*nm_to_m)*int_rp_kr3_kz_exp_j0
#     GEp[1,0] = GEp[0,1]; GEp[2,0] = -GEp[0,2]; GEp[2,1] = -GEp[1,2]

#     # --- GEs ---
#     GEs = np.zeros((3,3), dtype=complex)
#     GEs[0,0] = 1j/(8*pi*nm_to_m)*(int_rs_kr_kz_exp_j2*cos(2*phi)+int_rs_kr_kz_exp_j0)
#     GEs[0,1] = 1j*sin(2*phi)/(8*pi*nm_to_m)*int_rs_kr_kz_exp_j2
#     GEs[1,1] = 1j/(8*pi*nm_to_m)*(int_rs_kr_kz_exp_j0 - cos(2*phi)*int_rs_kr_kz_exp_j2)
#     GEs[1,0] = GEs[0,1]

#     # --- rotGHs ---
#     rotGHs = np.zeros((3,3), dtype=complex)
#     rotGHs[0,0] = sin(2*phi)/(8*pi)*int_rp_kr_exp_j2/nm_to_m**2
#     rotGHs[0,1] = 1/(8*pi)*(int_rp_kr_exp_j0 - cos(2*phi)*int_rp_kr_exp_j2)/nm_to_m**2
#     rotGHs[1,0] = -1/(8*pi)*(int_rp_kr_exp_j0 + cos(2*phi)*int_rp_kr_exp_j2)/nm_to_m**2
#     rotGHs[1,1] = -rotGHs[0,0]
#     rotGHs[2,0] = 1j*sin(phi)/(4*pi)*int_rp_kr2_kz_exp_j1/nm_to_m**2
#     rotGHs[2,1] = -1j*cos(phi)/(4*pi)*int_rp_kr2_kz_exp_j1/nm_to_m**2

#     # --- rotGHp ---
#     rotGHp = np.zeros((3,3), dtype=complex)
#     rotGHp[0,0] = sin(2*phi)/(8*pi)*int_rs_kr_exp_j2/nm_to_m**2
#     rotGHp[0,1] = -1/(8*pi)*(int_rs_kr_exp_j0 + cos(2*phi)*int_rs_kr_exp_j2)/nm_to_m**2
#     rotGHp[0,2] = -1j*sin(phi)/(4*pi)*int_rs_kr2_kz_exp_j1/nm_to_m**2
#     rotGHp[1,0] = 1/(8*pi)*(int_rs_kr_exp_j0 - cos(2*phi)*int_rs_kr_exp_j2)/nm_to_m**2
#     rotGHp[1,1] = -rotGHp[0,0]
#     rotGHp[1,2] = 1j*cos(phi)/(4*pi)*int_rs_kr2_kz_exp_j1/nm_to_m**2
    
    
#     if polarization == 'p':
#         return GEp, rotGHp
#     elif polarization == 's':
#         return GEs, rotGHs
#     else: 
#         return GEp+GEs, rotGHp+rotGHs


# def getG(wl, eps_interp, h, r, phi, field_type=None):
    
#     eps_val = eps_interp(wl)  


#     ints_E_evas = precompute_integrals(wl, h, r, False, eps_val, 'spp')
#     ints_H_evas = precompute_integrals(wl, h, r, True, eps_val, 'spp')
#     ints_E_reg =  precompute_integrals(wl, h, r, False, eps_val, 'reg')
#     ints_H_reg =  precompute_integrals(wl, h, r, True, eps_val, 'reg') 
    
#     GErp_evas, rotGHrs_evas = build_matrices(wl, phi, ints_E_evas, 'p')
#     GHrs_evas, rotGHrp_evas = build_matrices(wl, phi, ints_E_evas, 's')
    
#     GErs_evas, rotGErp_evas = build_matrices(wl, phi, ints_H_evas, 'p')
#     GHrp_evas, rotGErs_evas = build_matrices(wl, phi, ints_H_evas, 's')
    
#     GErp_reg, rotGHrs_reg = build_matrices(wl, phi, ints_E_reg, 'p')
#     GErs_reg, rotGHrp_reg = build_matrices(wl, phi, ints_E_reg, 's')
    
#     GHrs_reg, rotGErp_reg = build_matrices(wl, phi, ints_H_reg, 'p')
#     GHrp_reg, rotGErs_reg = build_matrices(wl, phi, ints_H_reg, 's')
#     # ints_E = precompute_integrals(wl, h, r, False, eps_val)
#     # ints_H = precompute_integrals(wl, h, r, True, eps_val)
    
#     # _, rotGHrs = build_matrices(wl, phi, ints_E, 'p')
#     # GErs, _ = build_matrices(wl, phi, ints_E, 's')
    
#     # GHrs, _ = build_matrices(wl, phi, ints_H, 'p')
#     # _, rotGErs = build_matrices(wl, phi, ints_H, 's')
    
#     if field_type == 'spp':
#         return [GErp_evas, rotGHrp_evas, GHrp_evas, rotGErp_evas]
        
#     elif field_type=='reg':
#         return [GErp_reg+GErs_reg+GErs_evas, rotGHrp_reg+rotGHrs_reg+rotGHrs_evas, GHrp_reg+GHrs_reg+GHrs_evas, rotGErp_reg+rotGErs_reg+rotGErs_evas]
#     else:
#         GE = GErp_evas + GErp_reg +  GErs_evas + GErs_reg
#         rotGH = rotGHrp_evas +rotGHrp_reg+ rotGHrs_evas + rotGHrs_reg
#         GH = GHrp_evas + GHrp_reg + GHrs_evas + GHrs_reg
#         rotGE = rotGErp_evas + rotGErp_reg + rotGErs_evas + rotGErs_reg

    
#     return [GE, rotGH, GH, rotGE]


from functools import lru_cache
import hashlib, os, pickle
import numpy as np
from scipy import integrate
from scipy.special import jn, j0, j1
from numpy import sqrt, exp, sin, cos, tan, pi
from MieSppForce import frenel

CACHE_FILE = "integrator_cache.pkl"


# === Кэш интегралов на диск ===
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
        
def clear_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

def integrator(f, key, limit=3000):
    cache = load_cache()
    if key in cache:
        return cache[key]

    def f_subst(t):
        kr = tan(t)
        return f(kr) * (1 / cos(t)**2)

    I, err = integrate.quad(lambda t: f_subst(t), 0, pi/2,
                            points=[pi/4], complex_func=True,
                            limit=limit)

    cache[key] = I
    save_cache(cache)
    return I


# === вычисляем интегралы (НЕ зависят от phi) ===
@lru_cache(maxsize=None)
def precompute_integrals(wl, h, r, forH, eps_val):
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
        key = hashlib.md5(f"{wl}_{eps_val}_{h}_{r}_{idx}_{forH}".encode()).hexdigest()
        res.append(integrator(f, key)*k)
    return tuple(res)


# === строим матрицы уже по phi ===
def build_matrices(wl, h, r, nm_to_m, phi, integrals, polarization=None):
    (int_rp_kr_kz_exp_j0, int_rp_kr_kz_exp_j2, int_rp_kr2_exp_j1,
     int_rp_kr3_kz_exp_j0, int_rs_kr_kz_exp_j2, int_rs_kr_kz_exp_j0,
     int_rp_kr_exp_j2, int_rp_kr_exp_j0, int_rp_kr2_kz_exp_j1,
     int_rs_kr_exp_j2, int_rs_kr_exp_j0, int_rs_kr2_kz_exp_j1) = integrals

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
    # return GEp+GEs, rotGHp+rotGHs



# === основной интерфейс ===
def getG(wl, eps_interp, h, r, phi, polarization=None):
    nm_to_m = 1e-9
    eps_val = eps_interp(wl)  # значение на wl
    # интегралы посчитали ровно один раз
    ints_E = precompute_integrals(wl, h, r, False, eps_val)
    ints_H = precompute_integrals(wl, h, r, True, eps_val)
    
    if polarization == 'p':
        GE, rotGH = build_matrices(wl, h, r, nm_to_m, phi, ints_E, 'p')
        GH, rotGE = build_matrices(wl, h, r, nm_to_m, phi, ints_H, 'p')
    elif polarization =='s':
        GE, rotGH = build_matrices(wl, h, r, nm_to_m, phi, ints_E, 's')
        GH, rotGE = build_matrices(wl, h, r, nm_to_m, phi, ints_H, 's')
    else:
        GE, rotGH = build_matrices(wl, h, r, nm_to_m, phi, ints_E)
        GH, rotGE = build_matrices(wl, h, r, nm_to_m, phi, ints_H)
    
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