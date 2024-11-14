import frenel
from scipy.integrate import quad
import numpy as np
from green_func import rot_green_ref_00


wls = np.linspace(400, 900, 50)
z0 = (146+20)*1e-9
eps_Au = frenel.get_interpolate('Au')
for wli in wls:
    #print(quad(lambda kr: green_ref_00_integrand(kr, wli, z0, eps_Au)
    #      [0][0, 0], 0, np.inf, epsrel=1e-3, complex_func=True))
    print(rot_green_ref_00(wli, z0, eps_Au))