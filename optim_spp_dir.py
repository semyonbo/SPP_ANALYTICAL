import numpy as np
from tqdm import tqdm
from MieSppForce import frenel, dipoles
from MieSppForce.directivity import get_directivity
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product

c = 299792458
RADIUS = 295/2
OBS_POINT = [0, 0, 2+RADIUS]
ANGLE = 25*np.pi/180

# Parameter grids
grid_wl = np.linspace(800, 1000, 50)
grid_phase = np.linspace(-np.pi, np.pi, 200)
grid_ai = np.linspace(0, np.pi / 2, 200)
theta_list = np.linspace(0, 2 * np.pi, 200)

# Material dispersions
eps_Au = frenel.get_interpolate('Au')
eps_Si = frenel.get_interpolate('Si')

dipole_cache = {}

# @lru_cache(maxsize=None)
def get_dipoles(wl, phase, ai):
    key = (float(wl), float(phase), float(ai))
    if key not in dipole_cache:
            P, M = dipoles.calc_dipoles_v2(
                wl,
                eps_Au,
                OBS_POINT,
                RADIUS,
                eps_Si,
                ANGLE,
                amplitude=1,
                phase=phase,
                a_angle=ai,
                stop=45,
            )
            dipole_cache[key] = (P[:, 0], M[:, 0])
    return dipole_cache[key]

def process_wavelength(wl):
    n = len(theta_list)
    best_D = np.full(n, -np.inf)
    best_params = np.zeros((n, 3))  # wl, phase, ai
    best_P = np.zeros((n, 3), dtype=complex)
    best_M = np.zeros((n, 3), dtype=complex)

    for phase, ai in product(grid_phase, grid_ai):
        P, M = get_dipoles(wl, phase, ai)
        Dir_func = get_directivity(wl, P, M, eps_Au)
        D_vals_fwd = Dir_func(theta_list)
        #D_vals_bwd = Dir_func(theta_list+np.pi)
        D_vals = D_vals_fwd

        mask = D_vals > best_D
        if mask.any():
            idxs = np.where(mask)[0]
            best_D[idxs] = D_vals[idxs]
            best_params[idxs] = (wl, phase, ai)
            best_P[idxs] = P
            best_M[idxs] = M

    return best_D, best_params, best_P, best_M

def main():
    n = len(theta_list)
    global_best_D = np.full(n, -np.inf)
    global_params = np.zeros((n, 3))
    global_P = np.zeros((n, 3), dtype=complex)
    global_M = np.zeros((n, 3), dtype=complex)

    # Parallel processing over wavelengths using loky (multiprocessing)
    results = Parallel(n_jobs=-1, backend='loky', verbose=5)(
        delayed(process_wavelength)(wl) for wl in grid_wl
    )

    # Aggregate results
    for D, params, P_arr, M_arr in results:
        mask = D > global_best_D
        if mask.any():
            idxs = np.where(mask)[0]
            global_best_D[idxs] = D[idxs]
            global_params[idxs] = params[idxs]
            global_P[idxs] = P_arr[idxs]
            global_M[idxs] = M_arr[idxs]

    # Assemble and save results
    output = [
        {
            'theta': theta_list[i],
            'wl': global_params[i][0],
            'phase': global_params[i][1],
            'a_angle': global_params[i][2],
            'D': global_best_D[i],
            'P': global_P[i],
            'M': global_M[i],
        }
        for i in range(n)
    ]

    np.savez('spp_optim_clean_v3.npz', results=output, thetas=theta_list)
    print("✅ Done: results saved to spp_optim_clean_v3.npz")


if __name__ == '__main__':
    main()
