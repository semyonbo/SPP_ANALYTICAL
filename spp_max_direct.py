import numpy as np
from MieSppForce import frenel, dipoles
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit

c_const = 299792458


@njit
def compute_intensity(phi, mx, my, px, py, pz, ka, kspp):
    return np.abs((mx / c_const + 1j * ka * py) * np.sin(phi) +
                  (my / c_const - 1j * ka * px) * np.cos(phi) - kspp * pz) ** 2




def get_intensity(wl, P, M, eps_Au):
    px, py, pz = P
    mx, my, mz = M
    ka = -1j * np.sqrt(1 / (eps_Au(wl) + 1))
    kspp = np.sqrt(eps_Au(wl) / (eps_Au(wl) + 1))
    return lambda phi: compute_intensity(phi, mx, my, px, py, pz, ka, kspp)


eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')

STOP = 45
R = 146
dist = 20
point = [0, 0, dist + R]
ANGLE = 25 * np.pi / 180
wl = 850

phase_values = np.linspace(0, 2*np.pi, 100)
a_angle = np.linspace(0, np.pi/2, 100)
angles = np.linspace(0, 2 * np.pi, 100)

def compute_for_params(angle_index, pha, a_i):
    p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
                                   amplitude=1, phase=pha, a_angle=a_i, stop=STOP)
    p, m = p[:, 0], m[:, 0]
    intensity = get_intensity(wl, p, m, eps_Au)

    Imax, _ = quad(intensity, 0, 2 * np.pi)
    D = 2 * np.pi * intensity(angles[angle_index]) / Imax

    return angle_index, pha, a_i, D, p, m

if __name__ == '__main__':
    # Параллельный запуск с 16 потоками
    results = []
    # with ProcessPoolExecutor(max_workers=16) as executor:
    #     futures = [
    #         executor.submit(compute_for_params, angle_index, pha, a_i)
    #         for angle_index in range(len(angles))
    #         for pha in phase_values
    #         for a_i in a_angle
    #     ]
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         results.append(future.result())
    
    for angle_index in tqdm(range(len(angles))):
        for pha in phase_values:
            for a_i in a_angle:
                result = compute_for_params(angle_index, pha, a_i)
                results.append(result)

    # Инициализация пустых массивов для хранения результатов
    best_D = np.zeros(len(angles))
    best_params = np.empty((len(angles), 2))
    dipoles_p = np.empty((len(angles), 3), dtype=np.complex128)
    dipoles_m = np.empty((len(angles), 3), dtype=np.complex128)

    # Поиск лучших параметров для каждого угла
    for i, pha, a_i, D, p, m in results:
        if D > best_D[i]:
            best_D[i] = D
            best_params[i] = (pha, a_i)
            dipoles_p[i] = p
            dipoles_m[i] = m

    # Сохранение в файл .npz
    np.savez("results_optimized_3_indexes_new.npz", Ds=best_D, params=best_params, dipoles_p=dipoles_p, dipoles_m=dipoles_m, angles=angles)
