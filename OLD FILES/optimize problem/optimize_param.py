import numpy as np
from MieSppForce import frenel, dipoles, force
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import quad
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# === Константы и параметры ===
c_const = 299792458
STOP = 45
R = 110
dist = 2
point = [0, 0, dist + R]
ANGLE = 25 * np.pi / 180

# Диапазоны
angles = np.linspace(0, 2 * np.pi, 100)
wavelength_range = np.linspace(500, 1300, 100)

# Дисперсии
eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')


def get_intensity(wl, P, M, eps_Au):
    px, py, pz = P
    mx, my, mz = M
    ka = -1j * np.sqrt(1 / (eps_Au(wl) + 1))
    kspp = np.sqrt(eps_Au(wl) / (eps_Au(wl) + 1))
    return lambda phi: np.abs((mx / c_const + 1j * ka * py) * np.sin(phi) +
                              (my / c_const - 1j * ka * px) * np.cos(phi) - kspp * pz) ** 2


# === Основной цикл по углам ===
params = np.zeros((len(angles), 3))  # wl, phase, a_i
losses = np.zeros(len(angles))
angle_F = np.zeros(len(angles))
angle_D = np.zeros(len(angles))

for i, angle_target in enumerate(tqdm(angles, desc="Оптимизация по углам")):
    best_loss = 1e6
    best_params = (np.nan, np.nan, np.nan)
    best_angle_F = np.nan
    best_angle_D = np.nan

    for wl in wavelength_range:
        def objective_phase_pol(params_vec):
            pha, a_i = params_vec
            try:
                p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
                                               amplitude=1, phase=pha, a_angle=a_i, stop=STOP)
                f = force.F(wl, eps_Au, point, R, eps_Si, ANGLE,
                            amplitude=1, phase=pha, a_angle=a_i, stop=STOP, full_output=True)
                Fx = f[0][0] - f[0][1] - f[0][4]
                Fy = f[1][0]
                angle_F_now = np.arctan2(Fy, Fx)

                p, m = p[:, 0], m[:, 0]
                intensity = get_intensity(wl, p, m, eps_Au)
                res = minimize_scalar(lambda phi: -intensity(phi), bounds=(-np.pi, np.pi), method='bounded')
                angle_D_now = res.x

                directivity = 2 * np.pi * intensity(angle_D_now) / quad(lambda phi: intensity(phi), -np.pi, np.pi)[0]
                if directivity < 1.5:
                    return 100

                delta_F = np.abs(np.angle(np.exp(1j * (angle_F_now - angle_target))))
                delta_DF = np.abs(np.angle(np.exp(1j * (angle_D_now - angle_F_now))))
                return delta_F + 0.5 * np.abs(np.pi - delta_DF)
            except:
                return 1e3

        res = minimize(
            objective_phase_pol,
            x0=[0.0, 0.1],
            bounds=[(-np.pi, np.pi), (0, np.pi / 2)],
            method="L-BFGS-B"
        )

        loss = res.fun
        pha_opt, a_i_opt = res.x

        if loss < best_loss:
            try:
                p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
                                               amplitude=1, phase=pha_opt, a_angle=a_i_opt, stop=STOP)
                f = force.F(wl, eps_Au, point, R, eps_Si, ANGLE,
                            amplitude=1, phase=pha_opt, a_angle=a_i_opt, stop=STOP, full_output=True)
                Fx = f[0][0] - f[0][1] - f[0][4]
                Fy = f[1][0]
                angle_F_now = np.arctan2(Fy, Fx)

                p, m = p[:, 0], m[:, 0]
                intensity = get_intensity(wl, p, m, eps_Au)
                res_int = minimize_scalar(lambda phi: -intensity(phi), bounds=(-np.pi, np.pi), method='bounded')
                angle_D_now = res_int.x

                best_loss = loss
                best_params = (wl, pha_opt, a_i_opt)
                best_angle_F = angle_F_now
                best_angle_D = angle_D_now
            except:
                continue

    params[i] = best_params
    losses[i] = best_loss
    angle_F[i] = best_angle_F
    angle_D[i] = best_angle_D

# === Расчёт отклонений ===
delta_F = np.abs(angle_F - angles)
delta_DF = np.abs(angle_D - angle_F)

# === Сохранение результатов ===
np.savez("opt_split_wavelength_phasepol.npz",
         params=params,
         losses=losses,
         angles=angles,
         angle_F=angle_F,
         angle_D=angle_D,
         delta_F=delta_F,
         delta_DF=delta_DF)

print("✅ Завершено. Результаты сохранены в 'opt_split_wavelength_phasepol.npz'")
