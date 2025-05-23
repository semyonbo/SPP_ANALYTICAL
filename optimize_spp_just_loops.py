import numpy as np
from MieSppForce import frenel, dipoles
from tqdm import tqdm
from MieSppForce.directivity import get_directivity

# === Константы и параметры ===
c_const = 299792458
R = 295/2  # радиус
point = [0, 0, 2+R]  # точка наблюдения
grid_wl = np.linspace(800, 1100, 50)      # длины волн
grid_phase = np.linspace(-np.pi, np.pi, 50)   # фазы
grid_ai = np.linspace(0, np.pi/2, 50)         # угол поляризации

# дисперсии
eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')

# углы оптимизации
theta_list = np.linspace(0, 2*np.pi, 50)

# массивы результатов
results = []

for theta in tqdm(theta_list, desc="Оптимизация по углам"):
    best = {'D': -np.inf}
    for wl in grid_wl:
        for pha in grid_phase:
            for ai in grid_ai:
                # считаем диполи
                P, M = dipoles.calc_dipoles_v2(
                    wl, eps_Au, point, R, eps_Si, theta,
                    amplitude=1, phase=pha, a_angle=ai, stop=45
                )
                P, M = P[:,0], M[:,0]
                D = get_directivity(wl, P, M, eps_Au)
                D_val = D(theta)
                if D_val > best['D']:
                    best.update({
                        'wl': wl,
                        'phase': pha,
                        'a_angle': ai,
                        'D': D_val,
                        'P': P,
                        'M': M
                    })
    results.append(best)

# сохранение
np.savez('spp_optim_simple.npz', results=results, thetas=theta_list)
print("Готово. Результаты сохранены в spp_optim_simple.npz")
