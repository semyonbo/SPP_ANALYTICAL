import numpy as np
from MieSppForce import frenel, dipoles, force
from scipy.optimize import minimize
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm
import warnings
from MieSppForce.directivity import get_directivity

warnings.filterwarnings("ignore")

# === Константы и параметры ===
c_const = 299792458
STOP = 45
R = 110
dist = 2
point = [0, 0, dist + R]
ANGLE = 25 * np.pi / 180

D_treshold = 1.5

# Диапазоны
wls_values = np.linspace(600, 1100, 100)
angles = np.linspace(0, 2 * np.pi, 100)

# Дисперсии
eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')



def compute_for_wavelength(wl, target_angle):
    def objective(params):
        pha, a_i = params
        try:
            # p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
            #                                amplitude=1, phase=pha, a_angle=a_i, stop=STOP)
            f = force.F(wl, eps_Au, point, R, eps_Si, ANGLE,
                        amplitude=1, phase=pha, a_angle=a_i, stop=STOP, full_output=True)
            # f0 = force.F(wl, 1, [0,0,0], R, eps_Si, 0,
            #             amplitude=1, phase=0, a_angle=0, stop=STOP, full_output=False)[3]
            
            F_x, F_y = f[0], f[1]
            Fxy_norm= np.sqrt((F_x[0]-F_x[1] - F_x[4])**2 + F_y[0]**2)
            angle_F = np.arctan2(F_y[0], F_x[0] - F_x[1] - F_x[4])

            #p, m = p[:, 0], m[:, 0]
            #Dir = get_directivity(wl, p, m, eps_Au)
            #res = minimize_scalar(lambda phi: -Dir(phi), bounds=(-np.pi, np.pi), method='bounded')
            #angle_D = res.x

            delta_F = np.abs(np.angle(np.exp(1j * (angle_F - target_angle))))
            #delta_DF = np.abs(np.angle(np.exp(1j * (angle_D - angle_F))))
            #maxD =Dir(angle_D)
            # if maxD < D_treshold:
            #     return 10
            # return delta_F + (np.abs(np.pi -delta_DF))
            # if Fxy_norm/f0 < 0.1:
            #     return 10
            return delta_F
        except:
            return 100
    
    res = minimize(objective, x0=[0.0, 0.1],
                   bounds=[(-np.pi, np.pi), (0, np.pi / 2)],
                   method='L-BFGS-B')
    return wl, res.fun, res.x


def optimize_for_angle(angle_index):
    angle = angles[angle_index]
    print(f"Оптимизация для угла {np.degrees(angle):.1f}°")

    results = Parallel(n_jobs=16, backend="loky")(
        delayed(compute_for_wavelength)(wl, angle) for wl in wls_values
    )

    best_loss = 1e6
    best_params = None
    best_angle_F = None
    best_angle_D = None

    for wl, loss, x in results:
        pha, a_i = x
        try:
            # p, m = dipoles.calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, ANGLE,
            #                                amplitude=1, phase=pha, a_angle=a_i, stop=STOP)
            f = force.F(wl, eps_Au, point, R, eps_Si, ANGLE,
                        amplitude=1, phase=pha, a_angle=a_i, stop=STOP, full_output=True)
            # f0 = force.F(wl, 1, [0,0,0], R, eps_Si, 0,
            #             amplitude=1, phase=0, a_angle=0, stop=STOP, full_output=False)[3]
            
            F_x, F_y = f[0], f[1]
            #Fxy_norm= np.sqrt((F_x[0]-F_x[1] - F_x[4])**2 + F_y[0]**2)
            angle_F = np.arctan2(F_y[0], F_x[0] - F_x[1] - F_x[4])

            # p, m = p[:, 0], m[:, 0]
            # Dir = get_directivity(wl, p, m, eps_Au)
            # res = minimize_scalar(lambda phi: -Dir(phi), bounds=(-np.pi, np.pi), method='bounded')
            # angle_D = res.x
            # maxD = Dir(angle_D)

            # if maxD >= D_treshold and loss < best_loss:
            #     best_loss = loss
            #     best_params = (wl, pha, a_i)
            #     best_angle_F = angle_F
            #     best_angle_D = angle_D
            if loss < best_loss:
                best_loss = loss
                best_params = (wl, pha, a_i)
                best_angle_F = angle_F
                #best_angle_D = angle_D
        except:
            continue

    if best_params is None:
        best_params = (np.nan, np.nan, np.nan)
        best_angle_F = np.nan
        #best_angle_D = np.nan
        best_loss = 1000

    return angle_index, best_params, best_loss, best_angle_F, best_angle_D


# === Запуск оптимизации ===
results = [optimize_for_angle(i) for i in tqdm(range(len(angles)))]

# === Обработка результатов ===
params = np.zeros((len(angles), 3))
losses = np.zeros(len(angles))
angle_F = np.zeros(len(angles))
#angle_D = np.zeros(len(angles))

for i, param, loss, aF, aD in results:
    params[i] = param
    losses[i] = loss
    angle_F[i] = aF
    #angle_D[i] = aD

delta_F = np.abs( angle_F - angles)
#delta_DF = np.abs(angle_D - angle_F)

save_path = "optimize_no_spp_dir_600_900.npz"

# === Сохранение ===
np.savez(save_path,
         params=params,
         losses=losses,
         angles=angles,
         angle_F=angle_F,
         #angle_D=angle_D,
         delta_F=delta_F,
         #delta_DF=delta_DF)
)

print(f"✅ Готово. Результаты сохранены в '{save_path}'")
