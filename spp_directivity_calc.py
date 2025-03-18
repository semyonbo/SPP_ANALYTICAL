import numpy as np
from MieSppForce import frenel, dipoles
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from joblib import Parallel, delayed

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
dist = 20
point = [0, 0, dist + R]
ANGLE = 25 * np.pi / 180
wl = 850

phase_values = np.linspace(0, 2 * np.pi, 100)
a_values = np.linspace(0, np.pi/2, 100)
angles = np.linspace(0, 2 * np.pi, 100)

def compute_for_angle(angle_index):
    max_D = 0
    best_params = (0, 0)
    best_p, best_m = None, None

    for pha in phase_values:
        for a_i in a_values:
            p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
                                           amplitude=1, phase=pha, a_angle=a_i, stop=STOP)
            p, m = p[:, 0], m[:, 0]
            intensity = get_intensity(wl, p, m, eps_Au)
            Imax = quad(intensity, 0, 2 * np.pi)[0]
            D = 2 * np.pi * intensity(angles[angle_index]) / Imax

            if D > max_D:
                max_D = D
                best_params = (pha, a_i)
                best_p, best_m = p, m

    return angle_index, max_D, best_params, best_p, best_m

# Параллельный запуск
results = Parallel(n_jobs=16, backend="loky")(delayed(compute_for_angle)(i) for i in tqdm(range(len(angles))))

# Сохранение результатов
Ds = np.zeros(len(angles))
params = np.empty((len(angles), 2))
dipoles_p = []
dipoles_m = []

for i, D, param, p, m in results:
    Ds[i] = D
    params[i] = param
    dipoles_p.append(p)
    dipoles_m.append(m)

# Сохранение в файл .npz
np.savez("results.npz", Ds=Ds, params=params, dipoles_p=dipoles_p, dipoles_m=dipoles_m, angles=angles)
