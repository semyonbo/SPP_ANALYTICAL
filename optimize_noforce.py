import numpy as np
from MieSppForce import frenel, dipoles
from MieSppForce.directivity import get_directivity
from scipy.optimize import brute, fmin
from tqdm import tqdm
from joblib import Parallel, delayed
from functools import lru_cache
import warnings
from scipy.integrate import IntegrationWarning

# Отключаем IntegrationWarning
warnings.simplefilter("ignore", IntegrationWarning)

c_const = 299792458
STOP = 35
ANGLE = 25 * np.pi / 180
R = 100
dist = 2
point = [0, 0, dist]

theta_list = np.linspace(0, 2 * np.pi, 100)

eps_Si = frenel.get_interpolate('Si')
eps_Au = frenel.get_interpolate('Au')


dipole_cache = {}

def get_dipoles_cached(wl, phase, ai):
    key = (round(wl, 5), round(phase, 5), round(ai, 5))
    if key not in dipole_cache:
        P, M = dipoles.calc_dipoles_v2(
            wl, eps_Au, point, R, eps_Si, ANGLE,
            amplitude=1, phase=phase, a_angle=ai, stop=STOP
        )
        dipole_cache[key] = (P, M)
    return dipole_cache[key]

def global_objective_with_theta(x, theta):
    wl, phase, ai = x
    try:
        P, M = get_dipoles_cached(wl, phase, ai)
        Dir_func = get_directivity(wl, P, M, eps_Au)
        val_forward = Dir_func(theta)
        val_backward = Dir_func((theta + np.pi) % (2 * np.pi))
        return -(val_forward - val_backward)
    except Exception:
        return 1e6

def optimize_for_theta(theta):
    ranges = (
        slice(600, 1100, 10),
        slice(-np.pi, np.pi, np.pi/18),
        slice(0, np.pi/2, np.pi/18),
    )

    result = brute(
        global_objective_with_theta,
        ranges,
        args=(theta,),
        full_output=True,
        finish=fmin,
        workers=1  # без параллелизма внутри brute, чтобы избежать проблем с pickle
    )

    (wl_opt, phase_opt, ai_opt), _, _, _ = result

    try:
        P_opt, M_opt = get_dipoles_cached(wl_opt, phase_opt, ai_opt)
        Dir_func_opt = get_directivity(wl_opt, P_opt, M_opt, eps_Au)
        D = Dir_func_opt(theta)
        return {
            'theta': theta,
            'wl': wl_opt,
            'phase': phase_opt,
            'a_angle': ai_opt,
            'D': D,
            'P': P_opt,
            'M': M_opt,
        }
    except Exception:
        return None

def main():
    results = Parallel(n_jobs=16, verbose=10)(
        delayed(optimize_for_theta)(theta) for theta in theta_list
    )
    # Убираем None из результатов:
    results = [r for r in results if r is not None]

    np.savez('spp_optim_smooth_parallel.npz', results=results, thetas=theta_list)
    print("✅ Оптимизация завершена и результаты сохранены")

if __name__ == '__main__':
    main()