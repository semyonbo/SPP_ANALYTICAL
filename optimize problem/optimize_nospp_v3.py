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
wavelength_range = np.linspace(600, 1300, 100)

# Дисперсии
eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')


# === Основной цикл по углам ===
params = np.zeros((len(angles), 3))  # wl, phase, a_i
losses = np.zeros(len(angles))
angle_F = np.zeros(len(angles))


for i, angle_target in enumerate(tqdm(angles, desc="Оптимизация по углам")):
    best_loss = 1e6
    best_params = (np.nan, np.nan, np.nan)
    best_angle_F = np.nan
    best_angle_D = np.nan

    for wl in wavelength_range:
        def objective_phase_pol(params_vec):
            pha, a_i = params_vec
            try:
               
                f = force.F(wl, eps_Au, point, R, eps_Si, ANGLE,
                            amplitude=1, phase=pha, a_angle=a_i, stop=STOP, full_output=True)
                f0 = force.F(wl, 1, [0,0,0], R, eps_Si, 0,
                            amplitude=1, phase=0, a_angle=0, stop=STOP, full_output=False)
                
                
                Fx = f[0,0] - f[0,1] - f[0,4]
                Fy = f[1,0]
                normF = np.sqrt(Fx**2 + Fy**2)
                
                angle_F_now = np.arctan2(Fy, Fx)

                delta_F = np.abs(np.angle(np.exp(1j * (angle_F_now - angle_target))))
                
                if normF/np.abs(f0[2]) < 0.1:
                    return 1e3
                
                return delta_F 
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

        if loss < best_loss :
            try:
                f0 = force.F(wl, 1, [0,0,0], R, eps_Si, 0,
                            amplitude=1, phase=0, a_angle=0, stop=STOP, full_output=True)
                f = force.F(wl, eps_Au, point, R, eps_Si, ANGLE,
                            amplitude=1, phase=pha_opt, a_angle=a_i_opt, stop=STOP, full_output=True)
                Fx = f[0,0] - f[0,1] - f[0,4]
                Fy = f[1,0]
                normF = np.sqrt(Fx**2 + Fy**2)
                angle_F_now = np.arctan2(Fy, Fx)

                best_loss = loss
                best_params = (wl, pha_opt, a_i_opt)
                best_angle_F = angle_F_now
  
            except:
                continue

    params[i] = best_params
    losses[i] = best_loss
    angle_F[i] = best_angle_F

# === Расчёт отклонений ===
delta_F = np.abs(angle_F - angles)

filename = 'opt_only_forve_v3.npz'

# === Сохранение результатов ===
np.savez(filename,
         params=params,
         losses=losses,
         angles=angles,
         angle_F=angle_F,
         delta_F=delta_F)

print(f"✅ Завершено. Результаты сохранены в {filename}")
