import numpy as np
from MieSppForce import frenel, dipoles, force
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar

c_const = 299792458

def get_intensity(wl, P, M, eps_Au):
    px, py, pz = P
    mx, my, mz = M
    ka = -1j * np.sqrt(1 / (eps_Au(wl) + 1))
    kspp = np.sqrt(eps_Au(wl) / (eps_Au(wl) + 1))
    return lambda phi: np.abs((mx / c_const + 1j * ka * py) * np.sin(phi) +
                              (my / c_const - 1j * ka * px) * np.cos(phi) - kspp * pz) ** 2

eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')
STOP = 45
R = 100
dist = 2
point = [0, 0, dist + R]
ANGLE = 25 * np.pi / 180


wls_values = np.linspace(600, 800, 20)
phase_values = np.linspace(0, 2 * np.pi, 20)
a_values = np.linspace(0, np.pi/2, 20)
angles = np.linspace(0, 2 * np.pi, 20)

def compute_for_angle(angle_index):
    max_D = 0
    best_params = (0, 0, 0)
    best_p = np.array([None, None, None])
    best_m = np.array([None, None, None])
    
    best_angle_F, best_angle_D = 0, 0

    for wl in wls_values:
        for pha in phase_values:
            for a_i in a_values:

                p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
                                            amplitude=1, phase=pha, a_angle=a_i, stop=STOP)
                f =  force.F(wl, eps_Au, point, R, eps_Si, ANGLE, amplitude=1,phase=pha,a_angle=a_i, stop=STOP, full_output=True)
                F_x = f[0]
                F_y = f[1]
                F_z = f[2]
                
                angle_F = np.arctan2(F_y[0], F_x[0] - F_x[1] - F_x[4])
                
                p, m = p[:, 0], m[:, 0]
                intensity = get_intensity(wl, p, m, eps_Au)
                res = minimize_scalar(lambda phi: -intensity(phi), bounds=(-np.pi, np.pi), method='bounded')
                angle_D = res.x
                maxD = 2*np.pi*intensity(angle_D)/quad(lambda phi: intensity(phi), -np.pi, np.pi)[0]
                
                delta_angle_D_F = np.abs(angle_D - angle_F)
                delta_angle_F = np.abs(angle_F - angles[angle_index])
                
                if 5*np.pi/6<delta_angle_D_F < 7*np.pi/6 and delta_angle_F < np.pi / 16 and maxD>1:
                    max_D = maxD
                    best_params = (wl, pha, a_i)
                    best_p, best_m = p, m

    return angle_index, max_D, best_params, best_angle_F, best_angle_D, best_p, best_m

# Параллельный запуск
results = Parallel(n_jobs=16, backend="loky")(delayed(compute_for_angle)(i) for i in tqdm(range(len(angles))))

# Сохранение результатов
Ds = np.zeros(len(angles))
params = np.empty((len(angles), 3))
dipoles_p = []
dipoles_m = []
angleF = []
amgleD = []

for i, D, param, angle_F, angle_D, p, m in results:
    Ds[i] = D
    params[i] = param
    angleF.append(angle_F)
    amgleD.append(angle_D)
    dipoles_p.append(p)
    dipoles_m.append(m)

# Сохранение в файл .npz
np.savez("optimize_spp_f.npz", Ds=Ds, params=params, dipoles_p=dipoles_p, dipoles_m=dipoles_m, amgleD=amgleD, angle_F=angleF ,angles = angles)

# def compute_for_wavelength(wl_index):
#     wl = wls_values[wl_index]
#     max_D = 0
#     best_params = (0, 0, 0)
#     best_p = np.array([None, None, None])
#     best_m = np.array([None, None, None])
#     best_angle_F, best_angle_D = 0, 0

#     for angle_index, angle in enumerate(angles):
#         for pha in phase_values:
#             for a_i in a_values:

#                 p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
#                                                amplitude=1, phase=pha, a_angle=a_i, stop=STOP)
#                 f = force.F(wl, eps_Au, point, R, eps_Si, ANGLE, amplitude=1, phase=pha, a_angle=a_i, stop=STOP, full_output=True)
#                 F_x = f[0]
#                 F_y = f[1]
#                 F_z = f[2]

#                 angle_F = np.arctan2(F_y[0], F_x[0] - F_x[1] - F_x[4])

#                 p, m = p[:, 0], m[:, 0]
#                 intensity = get_intensity(wl, p, m, eps_Au)
#                 res = minimize_scalar(lambda phi: -intensity(phi), bounds=(-np.pi, np.pi), method='bounded')
#                 angle_D = res.x
#                 maxD = 2 * np.pi * intensity(angle_D) / quad(lambda phi: intensity(phi), -np.pi, np.pi)[0]

#                 delta_angle_D_F = np.abs(angle_D - angle_F)
#                 delta_angle_F = np.abs(angle_F - angle)

#                 if 5 * np.pi / 6 < delta_angle_D_F < 7 * np.pi / 6 and delta_angle_F < np.pi / 16 and maxD > 1:
#                     max_D = maxD
#                     best_params = (wl, pha, a_i)
#                     best_p, best_m = p, m

#     return wl_index, max_D, best_params, best_angle_F, best_angle_D, best_p, best_m, angle

# # Параллельный запуск
# results = Parallel(n_jobs=16, backend="loky")(delayed(compute_for_wavelength)(i) for i in tqdm(range(len(wls_values))))

# # Сохранение результатов
# Ds = np.zeros(len(wls_values))
# params = np.empty((len(wls_values), 3))
# dipoles_p = []
# dipoles_m = []
# angleF = []
# amgleD = []
# angles = []

# for i, D, param, angle_F, angle_D, p, m, angle in results:
#     Ds[i] = D
#     params[i] = param
#     angleF.append(angle_F)
#     amgleD.append(angle_D)
#     dipoles_p.append(p)
#     dipoles_m.append(m)
#     angles.append(angle)

# # Сохранение в файл .npz
# np.savez("optimize_spp_f.npz", Ds=Ds, params=params, dipoles_p=dipoles_p, dipoles_m=dipoles_m, angles=angles, wls_values=wls_values)